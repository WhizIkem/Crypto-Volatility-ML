from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.data.var_panel import (
    build_var_volatility_panel,
)
from crypto_volatility.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from crypto_volatility.models.var import (
    extract_var_coefficients,
    fit_var_model,
    forecast_var_one_step,
)
from crypto_volatility.utils.config import load_config

MAX_LAGS = 10
LAG_SELECTION_CRITERION = "aic"


def load_processed_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config["interval"]

    datasets = {}

    for asset, symbol in (
        data_config["symbols"].items()
    ):
        path = processed_dir / (
            f"{symbol.lower()}_{interval}_model.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        datasets[asset] = pd.read_parquet(
            path
        )

    return datasets


def split_var_panel(
    panel: pd.DataFrame,
    time_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create outer train and validation periods while preserving the
    purged boundary row as observable forecasting context.

    For a one-day-ahead VAR forecast, the row immediately before
    validation is legitimate historical information and must be
    included in the forecasting history, but it is not scored as a
    validation observation.
    """
    if time_column not in panel.columns:
        raise ValueError(
            f"Missing time column: {time_column}"
        )

    n_rows = len(panel)

    train_boundary = int(
        n_rows * 0.70
    )

    validation_boundary = int(
        n_rows * 0.85
    )

    train_target_end = (
        train_boundary - 1
    )

    validation_start = (
        train_boundary
    )

    validation_end = (
        validation_boundary - 1
    )

    train = (
        panel
        .iloc[
            :train_target_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation_context = (
        panel
        .iloc[
            train_target_end:
            validation_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        panel
        .iloc[
            validation_start:
            validation_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    if train.empty:
        raise ValueError(
            "VAR training partition is empty"
        )

    if validation_context.empty:
        raise ValueError(
            "VAR validation context is empty"
        )

    if validation.empty:
        raise ValueError(
            "VAR validation partition is empty"
        )

    return (
        train,
        validation_context,
        validation,
    )

def evaluate_var_validation(
    train: pd.DataFrame,
    validation_context: pd.DataFrame,
    validation: pd.DataFrame,
    asset_columns: list[str],
) -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Evaluate expanding one-step-ahead VAR forecasts.

    Lag order is selected once using outer training data only.

    At each validation step, the model is refitted using all market
    information observed up to that date, then forecasts the next row.
    """
    initial_model = fit_var_model(
        train_df=train,
        columns=asset_columns,
        maxlags=MAX_LAGS,
        ic=LAG_SELECTION_CRITERION,
    )

    selected_lag = (
        initial_model.k_ar
    )

    if selected_lag < 1:
        raise ValueError(
            "VAR lag selection returned zero lags"
        )

    history = pd.concat(
        [
            train[
                asset_columns
            ],
            validation_context[
                asset_columns
            ],
        ],
        ignore_index=True,
    )

    forecast_rows = []

    for validation_index in range(
        len(validation)
    ):
        current_row = validation.iloc[
            validation_index
        ]

        model = fit_var_model(
            train_df=history,
            columns=asset_columns,
            maxlags=selected_lag,
            ic=None,
        )

        forecast = forecast_var_one_step(
            model=model,
            history_df=history,
            columns=asset_columns,
        )

        row = {
            "validation_index": validation_index,
        }

        for asset in asset_columns:
            row[
                f"{asset}_actual"
            ] = float(
                current_row[
                    asset
                ]
            )

            row[
                f"{asset}_forecast"
            ] = float(
                forecast[
                    asset
                ]
            )

        forecast_rows.append(
            row
        )

        observed_row = pd.DataFrame(
            [
                {
                    asset: current_row[
                        asset
                    ]
                    for asset in asset_columns
                }
            ]
        )

        history = pd.concat(
            [
                history,
                observed_row,
            ],
            ignore_index=True,
        )

    return (
        pd.DataFrame(
            forecast_rows
        ),
        selected_lag,
    )


def summarize_forecasts(
    forecasts: pd.DataFrame,
    asset_columns: list[str],
    selected_lag: int,
) -> pd.DataFrame:
    """
    Calculate validation MAE and RMSE for each VAR equation.
    """
    rows = []

    for asset in asset_columns:
        actual = forecasts[
            f"{asset}_actual"
        ]

        predicted = forecasts[
            f"{asset}_forecast"
        ]

        rows.append(
            {
                "asset": asset,
                "model": "var",
                "selected_lag": selected_lag,
                "n_observations": len(
                    forecasts
                ),
                "mae": mean_absolute_error(
                    actual,
                    predicted,
                ),
                "rmse": root_mean_squared_error(
                    actual,
                    predicted,
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "rmse",
            "asset",
        ]
    ).reset_index(
        drop=True
    )


def run_granger_causality(
    train: pd.DataFrame,
    asset_columns: list[str],
    selected_lag: int,
) -> pd.DataFrame:
    """
    Test whether lagged volatility in one asset helps explain
    another asset inside the outer training period.

    The tests are descriptive research results, not final test-set
    model-selection results.
    """
    model = fit_var_model(
        train_df=train,
        columns=asset_columns,
        maxlags=selected_lag,
        ic=None,
    )

    rows = []

    for caused_asset in asset_columns:
        for causing_asset in asset_columns:
            if caused_asset == causing_asset:
                continue

            result = model.test_causality(
                caused=caused_asset,
                causing=[
                    causing_asset,
                ],
                kind="f",
            )

            rows.append(
                {
                    "causing_asset": causing_asset,
                    "caused_asset": caused_asset,
                    "selected_lag": selected_lag,
                    "test_statistic": float(
                        result.test_statistic
                    ),
                    "p_value": float(
                        result.pvalue
                    ),
                    "significant_5pct": (
                        result.pvalue
                        < 0.05
                    ),
                }
            )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "p_value",
            "causing_asset",
            "caused_asset",
        ]
    ).reset_index(
        drop=True
    )


def main() -> None:
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    panel = build_var_volatility_panel(
        datasets=datasets,
        time_column="timestamp",
        volatility_column="log_gk_volatility",
    )

    asset_columns = list(
        datasets.keys()
    )

    (
        train,
        validation_context,
        validation,
    ) = split_var_panel(
        panel=panel,
        time_column="timestamp",
    )

    (
        forecasts,
        selected_lag,
    ) = evaluate_var_validation(
        train=train,
        validation_context=validation_context,
        validation=validation,
        asset_columns=asset_columns,
    )

    metrics = summarize_forecasts(
        forecasts=forecasts,
        asset_columns=asset_columns,
        selected_lag=selected_lag,
    )

    granger = run_granger_causality(
        train=train,
        asset_columns=asset_columns,
        selected_lag=selected_lag,
    )

    fitted_train_model = fit_var_model(
        train_df=train,
        columns=asset_columns,
        maxlags=selected_lag,
        ic=None,
    )

    coefficients = (
        extract_var_coefficients(
            fitted_train_model
        )
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "var_validation_metrics.csv"
    )

    forecasts_path = (
        output_dir
        / "var_validation_forecasts.csv"
    )

    granger_path = (
        output_dir
        / "var_granger_causality.csv"
    )

    coefficients_path = (
        output_dir
        / "var_coefficients.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    forecasts.to_csv(
        forecasts_path,
        index=False,
    )

    granger.to_csv(
        granger_path,
        index=False,
    )

    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    print(
        "\nSelected VAR lag order:",
        selected_lag,
    )

    print(
        "\nVAR validation metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nGranger-causality results:\n"
    )

    print(
        granger.to_string(
            index=False
        )
    )

    print(
        "\nSaved VAR metrics to:",
        metrics_path,
    )

    print(
        "Saved VAR Granger results to:",
        granger_path,
    )


if __name__ == "__main__":
    main()