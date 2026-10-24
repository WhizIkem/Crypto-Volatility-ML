from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.data.var_panel import (
    build_var_volatility_panel,
)
from crypto_volatility.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from crypto_volatility.evaluation.robust_validation import (
    expanding_validation_splits,
)
from crypto_volatility.features.build_features import (
    TARGET_COLUMN,
)
from crypto_volatility.models.var import (
    fit_var_model,
    forecast_var_one_step,
)
from crypto_volatility.utils.config import load_config

INITIAL_TRAIN_SIZE = 1200
VALIDATION_SIZE = 120
STEP_SIZE = 120
PURGE_ROWS = 1

VAR_LAG = 7


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

        required_columns = {
            "timestamp",
            "log_gk_volatility",
            TARGET_COLUMN,
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{asset} dataset missing columns: "
                f"{sorted(missing_columns)}"
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


def build_development_datasets(
    datasets: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.Timestamp,
]:
    """
    Restrict every asset to the Phase 7 development period.

    The row whose t+1 target crosses into the final test period
    remains excluded.
    """
    development_datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    test_starts = []

    for asset, df in (
        datasets.items()
    ):
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
                f"{asset} outer test split is empty"
            )

        final_test_start = (
            outer_split.test[
                "timestamp"
            ].iloc[0]
        )

        positions = df.index[
            df[
                "timestamp"
            ].eq(
                final_test_start
            )
        ].tolist()

        if len(
            positions
        ) != 1:
            raise ValueError(
                f"Could not uniquely locate "
                f"{asset} final test start"
            )

        test_start_position = (
            positions[
                0
            ]
        )

        development_end = (
            test_start_position
            - PURGE_ROWS
        )

        development_datasets[
            asset
        ] = (
            df
            .iloc[
                :development_end
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_starts.append(
            final_test_start
        )

    unique_test_starts = set(
        test_starts
    )

    if len(
        unique_test_starts
    ) != 1:
        raise ValueError(
            "Assets do not share the same final test start date"
        )

    return (
        development_datasets,
        next(
            iter(
                unique_test_starts
            )
        ),
    )


def build_var_research_panel(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the joint volatility panel plus aligned t+1 targets.

    Columns such as BTC/ETH/SOL/BNB contain volatility observed
    at day t.

    Columns such as BTC_target contain the true volatility at t+1.
    """
    panel = build_var_volatility_panel(
        datasets=datasets,
        time_column="timestamp",
        volatility_column="log_gk_volatility",
    )

    for asset, df in (
        datasets.items()
    ):
        target_frame = (
            df[
                [
                    "timestamp",
                    TARGET_COLUMN,
                ]
            ]
            .copy()
            .rename(
                columns={
                    TARGET_COLUMN: (
                        f"{asset}_target"
                    ),
                }
            )
        )

        panel = panel.merge(
            target_frame,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

    return (
        panel
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


def evaluate_var_fold(
    train: pd.DataFrame,
    context: pd.DataFrame,
    validation: pd.DataFrame,
    asset_columns: list[str],
) -> pd.DataFrame:
    """
    Evaluate one fixed-lag VAR fold on the true t+1 target.

    For validation row t:

    1. volatility at t is now observed;
    2. append that observation to VAR history;
    3. fit using information available through t;
    4. forecast t+1;
    5. compare against target_log_gk_vol_t_plus_1 stored on row t.
    """
    history = pd.concat(
        [
            train[
                asset_columns
            ],
            context[
                asset_columns
            ],
        ],
        ignore_index=True,
    )

    rows = []

    for validation_index in range(
        len(validation)
    ):
        current_row = validation.iloc[
            validation_index
        ]

        current_observation = pd.DataFrame(
            [
                {
                    asset: current_row[
                        asset
                    ]
                    for asset in asset_columns
                }
            ]
        )

        # Day t is observed before making the t+1 forecast.
        history = pd.concat(
            [
                history,
                current_observation,
            ],
            ignore_index=True,
        )

        model = fit_var_model(
            train_df=history,
            columns=asset_columns,
            maxlags=VAR_LAG,
            ic=None,
        )

        forecast = forecast_var_one_step(
            model=model,
            history_df=history,
            columns=asset_columns,
        )

        row = {
            "validation_index": validation_index,
            "forecast_origin": current_row[
                "timestamp"
            ],
        }

        for asset in asset_columns:
            row[
                f"{asset}_actual"
            ] = float(
                current_row[
                    f"{asset}_target"
                ]
            )

            row[
                f"{asset}_forecast"
            ] = float(
                forecast[
                    asset
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def calculate_fold_metrics(
    forecasts: pd.DataFrame,
    asset_columns: list[str],
    fold: int,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Calculate per-asset metrics for one corrected VAR fold.
    """
    rows = []

    for asset in asset_columns:
        y_true = forecasts[
            f"{asset}_actual"
        ]

        y_pred = forecasts[
            f"{asset}_forecast"
        ]

        rows.append(
            {
                "asset": asset,
                "fold": fold,
                "model": "VAR",
                "lag": VAR_LAG,
                "validation_rows": len(
                    forecasts
                ),
                "validation_start": validation_start,
                "validation_end": validation_end,
                "mae": mean_absolute_error(
                    y_true,
                    y_pred,
                ),
                "rmse": root_mean_squared_error(
                    y_true,
                    y_pred,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate corrected VAR performance across all folds.
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

    full_datasets = load_model_datasets(
        config
    )

    (
        development_datasets,
        final_test_start,
    ) = build_development_datasets(
        full_datasets
    )

    panel = build_var_research_panel(
        development_datasets
    )

    asset_columns = list(
        development_datasets.keys()
    )

    splits = expanding_validation_splits(
        panel,
        initial_train_size=INITIAL_TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        step_size=STEP_SIZE,
        purge_rows=PURGE_ROWS,
    )

    print(
        "\nPhase 7 corrected VAR robust validation"
    )

    print(
        "Forecast task: t -> t+1"
    )

    print(
        "Fixed VAR lag:",
        VAR_LAG,
    )

    print(
        "Number of folds:",
        len(
            splits
        ),
    )

    print(
        "Final sealed test begins:",
        final_test_start,
    )

    all_metrics = []

    for split in splits:
        forecasts = evaluate_var_fold(
            train=split.train,
            context=split.context,
            validation=split.validation,
            asset_columns=asset_columns,
        )

        metrics = calculate_fold_metrics(
            forecasts=forecasts,
            asset_columns=asset_columns,
            fold=split.fold,
            validation_start=(
                split.validation[
                    "timestamp"
                ].iloc[0]
            ),
            validation_end=(
                split.validation[
                    "timestamp"
                ].iloc[-1]
            ),
        )

        all_metrics.append(
            metrics
        )

    fold_metrics = pd.concat(
        all_metrics,
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

    fold_path = (
        output_dir
        / "robust_var_fold_metrics.csv"
    )

    summary_path = (
        output_dir
        / "robust_var_summary.csv"
    )

    fold_metrics.to_csv(
        fold_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nCorrected VAR fold metrics:\n"
    )

    print(
        fold_metrics[
            [
                "asset",
                "fold",
                "mae",
                "rmse",
            ]
        ]
        .sort_values(
            [
                "asset",
                "fold",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nCorrected VAR robust summary:\n"
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


if __name__ == "__main__":
    main()