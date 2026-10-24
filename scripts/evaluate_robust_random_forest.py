from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from crypto_volatility.evaluation.robust_validation import (
    expanding_validation_splits,
)
from crypto_volatility.features.build_features import (
    CORE_PLUS_DOWNSIDE_FEATURES,
    TARGET_COLUMN,
)
from crypto_volatility.models.random_forest import (
    fit_random_forest_model,
    predict_random_forest,
)
from crypto_volatility.utils.config import load_config

INITIAL_TRAIN_SIZE = 1200
VALIDATION_SIZE = 120
STEP_SIZE = 120
PURGE_ROWS = 1


RANDOM_FOREST_PARAMS = {
    "BTC": {
        "max_depth": 8,
        "min_samples_leaf": 20,
    },
    "ETH": {
        "max_depth": 3,
        "min_samples_leaf": 10,
    },
    "SOL": {
        "max_depth": 5,
        "min_samples_leaf": 20,
    },
    "BNB": {
        "max_depth": 3,
        "min_samples_leaf": 5,
    },
}


def load_model_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config[
        "interval"
    ]

    datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
        path = processed_dir / (
            f"{symbol.lower()}_{interval}_model.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        df = pd.read_parquet(
            path
        )

        if "timestamp" not in df.columns:
            raise ValueError(
                f"{asset} dataset is missing 'timestamp'"
            )

        datasets[
            asset
        ] = (
            df
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

    return datasets


def build_development_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
]:
    outer_split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=PURGE_ROWS,
        )
    )

    if outer_split.test.empty:
        raise ValueError(
            "Outer test split is empty"
        )

    final_test_start = (
        outer_split.test[
            "timestamp"
        ].iloc[0]
    )

    matching_positions = df.index[
        df[
            "timestamp"
        ].eq(
            final_test_start
        )
    ].tolist()

    if len(
        matching_positions
    ) != 1:
        raise ValueError(
            "Could not uniquely identify final test start"
        )

    test_start_position = (
        matching_positions[
            0
        ]
    )

    development_end = (
        test_start_position
        - PURGE_ROWS
    )

    development_df = (
        df
        .iloc[
            :development_end
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        development_df,
        final_test_start,
    )


def evaluate_asset(
    asset: str,
    development_df: pd.DataFrame,
) -> pd.DataFrame:
    if asset not in RANDOM_FOREST_PARAMS:
        raise ValueError(
            f"Missing Random Forest parameters for {asset}"
        )

    params = RANDOM_FOREST_PARAMS[
        asset
    ]

    splits = expanding_validation_splits(
        development_df,
        initial_train_size=INITIAL_TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        step_size=STEP_SIZE,
        purge_rows=PURGE_ROWS,
    )

    rows = []

    for split in splits:
        model = fit_random_forest_model(
            train_df=split.train,
            feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
            target_column=TARGET_COLUMN,
            n_estimators=300,
            max_depth=params[
                "max_depth"
            ],
            min_samples_leaf=params[
                "min_samples_leaf"
            ],
            max_features=1.0,
            random_state=42,
        )

        predictions = predict_random_forest(
            model=model,
            df=split.validation,
            feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        )

        y_true = split.validation[
            TARGET_COLUMN
        ]

        mae = mean_absolute_error(
            y_true,
            predictions,
        )

        rmse = root_mean_squared_error(
            y_true,
            predictions,
        )

        rows.append(
            {
                "asset": asset,
                "fold": split.fold,
                "model": "Random Forest",
                "train_rows": len(
                    split.train
                ),
                "validation_rows": len(
                    split.validation
                ),
                "validation_start": (
                    split.validation[
                        "timestamp"
                    ].iloc[0]
                ),
                "validation_end": (
                    split.validation[
                        "timestamp"
                    ].iloc[-1]
                ),
                "mae": mae,
                "rmse": rmse,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        fold_metrics
        .groupby(
            [
                "asset",
                "model",
            ],
            as_index=False,
        )
        .agg(
            mean_mae=(
                "mae",
                "mean",
            ),
            std_mae=(
                "mae",
                "std",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
            min_rmse=(
                "rmse",
                "min",
            ),
            max_rmse=(
                "rmse",
                "max",
            ),
            folds=(
                "fold",
                "count",
            ),
        )
    )

    return (
        summary
        .sort_values(
            [
                "asset",
                "mean_rmse",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    config = load_config()

    full_datasets = (
        load_model_datasets(
            config
        )
    )

    development_datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    test_start_dates: dict[
        str,
        pd.Timestamp,
    ] = {}

    for asset, df in (
        full_datasets.items()
    ):
        (
            development_df,
            final_test_start,
        ) = build_development_dataset(
            df
        )

        development_datasets[
            asset
        ] = development_df

        test_start_dates[
            asset
        ] = final_test_start

    unique_test_starts = set(
        test_start_dates.values()
    )

    if len(
        unique_test_starts
    ) != 1:
        raise ValueError(
            "Assets do not share the same final test start date"
        )

    final_test_start = next(
        iter(
            unique_test_starts
        )
    )

    print(
        "\nPhase 7 Random Forest robust validation"
    )

    print(
        "Final sealed test begins:",
        final_test_start,
    )

    asset_results = []

    for asset, development_df in (
        development_datasets.items()
    ):
        results = evaluate_asset(
            asset=asset,
            development_df=development_df,
        )

        asset_results.append(
            results
        )

    fold_metrics = pd.concat(
        asset_results,
        ignore_index=True,
    )

    summary = build_summary(
        fold_metrics
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics_path = (
        output_dir
        / "robust_random_forest_fold_metrics.csv"
    )

    summary_path = (
        output_dir
        / "robust_random_forest_summary.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nFold metrics:\n"
    )

    print(
        fold_metrics[
            [
                "asset",
                "fold",
                "mae",
                "rmse",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRandom Forest robust summary:\n"
    )

    print(
        summary[
            [
                "asset",
                "mean_mae",
                "std_mae",
                "mean_rmse",
                "std_rmse",
                "min_rmse",
                "max_rmse",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved fold metrics to:",
        fold_metrics_path,
    )

    print(
        "Saved summary to:",
        summary_path,
    )


if __name__ == "__main__":
    main()