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
from crypto_volatility.models.har import (
    fit_har_model,
    predict_har,
)
from crypto_volatility.models.lasso import (
    fit_lasso_model,
    predict_lasso,
)
from crypto_volatility.models.ridge import (
    fit_ridge_model,
    predict_ridge,
)
from crypto_volatility.utils.config import load_config

INITIAL_TRAIN_SIZE = 1200
VALIDATION_SIZE = 120
STEP_SIZE = 120
PURGE_ROWS = 1


RIDGE_ALPHAS = {
    "BTC": 100.0,
    "ETH": 10.0,
    "SOL": 0.001,
    "BNB": 100.0,
}


LASSO_ALPHAS = {
    "BTC": 0.01,
    "ETH": 0.001,
    "SOL": 0.005,
    "BNB": 0.01,
}


def load_model_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Load processed model datasets for all configured assets.
    """
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

        df = (
            df
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        datasets[
            asset
        ] = df

    return datasets


def build_development_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
]:
    """
    Build the Phase 7 development period while keeping the final
    outer test partition sealed.

    The row immediately before the final test set is also excluded
    because its t+1 target crosses into the test period.
    """
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

    if development_end <= 0:
        raise ValueError(
            "Development period is empty"
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


def validate_common_timestamps(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    """
    Confirm that all asset datasets use the same timestamps.

    This ensures every model is evaluated over exactly the same
    Phase 7 validation dates.
    """
    assets = list(
        datasets.keys()
    )

    if not assets:
        raise ValueError(
            "No datasets supplied"
        )

    reference_asset = assets[
        0
    ]

    reference_timestamps = (
        datasets[
            reference_asset
        ][
            "timestamp"
        ]
        .reset_index(
            drop=True
        )
    )

    for asset in assets[
        1:
    ]:
        timestamps = (
            datasets[
                asset
            ][
                "timestamp"
            ]
            .reset_index(
                drop=True
            )
        )

        if not reference_timestamps.equals(
            timestamps
        ):
            raise ValueError(
                "Asset datasets do not share identical timestamps: "
                f"{reference_asset} vs {asset}"
            )


def evaluate_har_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    float,
    float,
]:
    """
    Fit and evaluate HAR-GK on one robust-validation fold.
    """
    model = fit_har_model(
        train_df=train,
        target_column=TARGET_COLUMN,
    )

    predictions = predict_har(
        model=model,
        df=validation,
    )

    y_true = validation[
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

    return (
        mae,
        rmse,
    )


def evaluate_ridge_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    alpha: float,
) -> tuple[
    float,
    float,
]:
    """
    Fit and evaluate Ridge on one robust-validation fold.
    """
    model = fit_ridge_model(
        train_df=train,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        alpha=alpha,
    )

    predictions = predict_ridge(
        model=model,
        df=validation,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
    )

    y_true = validation[
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

    return (
        mae,
        rmse,
    )


def evaluate_lasso_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    alpha: float,
) -> tuple[
    float,
    float,
]:
    """
    Fit and evaluate Lasso on one robust-validation fold.
    """
    model = fit_lasso_model(
        train_df=train,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        alpha=alpha,
    )

    predictions = predict_lasso(
        model=model,
        df=validation,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
    )

    y_true = validation[
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

    return (
        mae,
        rmse,
    )


def evaluate_asset(
    asset: str,
    development_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate HAR, Ridge, and Lasso over all locked Phase 7 folds.
    """
    if asset not in RIDGE_ALPHAS:
        raise ValueError(
            f"Missing Ridge alpha for {asset}"
        )

    if asset not in LASSO_ALPHAS:
        raise ValueError(
            f"Missing Lasso alpha for {asset}"
        )

    splits = expanding_validation_splits(
        development_df,
        initial_train_size=INITIAL_TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        step_size=STEP_SIZE,
        purge_rows=PURGE_ROWS,
    )

    rows = []

    for split in splits:
        har_mae, har_rmse = (
            evaluate_har_fold(
                train=split.train,
                validation=split.validation,
            )
        )

        ridge_mae, ridge_rmse = (
            evaluate_ridge_fold(
                train=split.train,
                validation=split.validation,
                alpha=RIDGE_ALPHAS[
                    asset
                ],
            )
        )

        lasso_mae, lasso_rmse = (
            evaluate_lasso_fold(
                train=split.train,
                validation=split.validation,
                alpha=LASSO_ALPHAS[
                    asset
                ],
            )
        )

        validation_start = (
            split.validation[
                "timestamp"
            ].iloc[0]
        )

        validation_end = (
            split.validation[
                "timestamp"
            ].iloc[-1]
        )

        model_results = [
            {
                "model": "HAR-GK",
                "mae": har_mae,
                "rmse": har_rmse,
            },
            {
                "model": "Ridge",
                "mae": ridge_mae,
                "rmse": ridge_rmse,
            },
            {
                "model": "Lasso",
                "mae": lasso_mae,
                "rmse": lasso_rmse,
            },
        ]

        for result in model_results:
            rows.append(
                {
                    "asset": asset,
                    "fold": split.fold,
                    "model": result[
                        "model"
                    ],
                    "train_rows": len(
                        split.train
                    ),
                    "validation_rows": len(
                        split.validation
                    ),
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "mae": result[
                        "mae"
                    ],
                    "rmse": result[
                        "rmse"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


def add_fold_winners(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mark the model with the lowest RMSE within each asset/fold.

    Exact ties are allowed to produce more than one winner.
    """
    result = fold_metrics.copy()

    best_rmse = (
        result
        .groupby(
            [
                "asset",
                "fold",
            ]
        )[
            "rmse"
        ]
        .transform(
            "min"
        )
    )

    result[
        "fold_winner"
    ] = (
        result[
            "rmse"
        ]
        == best_rmse
    )

    return result


def build_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Phase 7 performance across robust-validation folds.
    """
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
            folds_won=(
                "fold_winner",
                "sum",
            ),
        )
    )

    summary[
        "folds_won"
    ] = (
        summary[
            "folds_won"
        ]
        .astype(
            int
        )
    )

    summary = (
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

    return summary


def print_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> None:
    """
    Print fold-level results grouped by asset.
    """
    for asset in (
        fold_metrics[
            "asset"
        ].unique()
    ):
        asset_results = (
            fold_metrics[
                fold_metrics[
                    "asset"
                ]
                == asset
            ]
            .sort_values(
                [
                    "fold",
                    "rmse",
                ]
            )
        )

        print(
            f"\n{asset} fold results:\n"
        )

        print(
            asset_results[
                [
                    "fold",
                    "model",
                    "mae",
                    "rmse",
                    "fold_winner",
                ]
            ].to_string(
                index=False
            )
        )


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """
    Print aggregate Phase 7 results.
    """
    print(
        "\nRobust validation summary:\n"
    )

    print(
        summary[
            [
                "asset",
                "model",
                "mean_mae",
                "std_mae",
                "mean_rmse",
                "std_rmse",
                "min_rmse",
                "max_rmse",
                "folds_won",
            ]
        ].to_string(
            index=False
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

    validate_common_timestamps(
        development_datasets
    )

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
        "\nPhase 7 robust validation"
    )

    print(
        "Initial train size:",
        INITIAL_TRAIN_SIZE,
    )

    print(
        "Validation size:",
        VALIDATION_SIZE,
    )

    print(
        "Step size:",
        STEP_SIZE,
    )

    print(
        "Purge rows:",
        PURGE_ROWS,
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

    fold_metrics = (
        add_fold_winners(
            fold_metrics
        )
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
        / "robust_linear_fold_metrics.csv"
    )

    summary_path = (
        output_dir
        / "robust_linear_summary.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_fold_metrics(
        fold_metrics
    )

    print_summary(
        summary
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