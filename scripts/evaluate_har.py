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
from crypto_volatility.features.build_features import (
    TARGET_COLUMN,
)
from crypto_volatility.models.har import (
    fit_har_model,
    predict_har,
)
from crypto_volatility.utils.config import load_config


def load_processed_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Load processed model-ready datasets.
    """
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config["interval"]
    symbols = data_config["symbols"]

    datasets = {}

    for asset, symbol in symbols.items():
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


def evaluate_forecast(
    asset: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, str | float | int]:
    """
    Evaluate one validation forecast.
    """
    aligned = pd.concat(
        [
            y_true.rename("y_true"),
            y_pred.rename("y_pred"),
        ],
        axis=1,
    ).dropna()

    return {
        "asset": asset,
        "model": model_name,
        "n_observations": len(aligned),
        "mae": mean_absolute_error(
            aligned["y_true"],
            aligned["y_pred"],
        ),
        "rmse": root_mean_squared_error(
            aligned["y_true"],
            aligned["y_pred"],
        ),
    }


def evaluate_asset(
    asset: str,
    df: pd.DataFrame,
) -> tuple[
    list[dict[str, str | float | int]],
    dict[str, str | float],
]:
    """
    Fit HAR on training data and evaluate HAR plus the
    7-day rolling benchmark on validation data.
    """
    split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=1,
        )
    )

    train = (
        split.train
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        split.validation
        .copy()
        .reset_index(drop=True)
    )

    y_validation = (
        validation[TARGET_COLUMN]
        .copy()
    )

    har_model = fit_har_model(
        train
    )

    har_prediction = predict_har(
        har_model,
        validation,
    )

    results = [
        evaluate_forecast(
            asset=asset,
            model_name="har_gk",
            y_true=y_validation,
            y_pred=har_prediction,
        )
    ]

    validation_rolling_prediction = (
        validation["har_vol_weekly"]
        .copy()
    )

    results.append(
        evaluate_forecast(
            asset=asset,
            model_name="rolling_mean_7d",
            y_true=y_validation,
            y_pred=validation_rolling_prediction,
        )
    )

    coefficients = {
        "asset": asset,
        "const": float(
            har_model.params["const"]
        ),
        "daily": float(
            har_model.params[
                "log_gk_volatility"
            ]
        ),
        "weekly": float(
            har_model.params[
                "har_vol_weekly"
            ]
        ),
        "monthly": float(
            har_model.params[
                "har_vol_monthly"
            ]
        ),
        "r_squared": float(
            har_model.rsquared
        ),
        "adjusted_r_squared": float(
            har_model.rsquared_adj
        ),
    }

    return (
        results,
        coefficients,
    )


def main() -> None:
    """
    Evaluate HAR-GK on the leakage-safe validation period.
    """
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    metric_rows = []
    coefficient_rows = []

    for asset, df in datasets.items():
        (
            asset_metrics,
            asset_coefficients,
        ) = evaluate_asset(
            asset,
            df,
        )

        metric_rows.extend(
            asset_metrics
        )

        coefficient_rows.append(
            asset_coefficients
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics = metrics.sort_values(
        [
            "asset",
            "rmse",
        ]
    ).reset_index(
        drop=True
    )

    coefficients = pd.DataFrame(
        coefficient_rows
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
        / "har_validation_metrics.csv"
    )

    coefficients_path = (
        output_dir
        / "har_coefficients.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    print(
        "\nHAR validation metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nHAR coefficients:\n"
    )

    print(
        coefficients.to_string(
            index=False
        )
    )

    print(
        "\nSaved metrics to:",
        metrics_path,
    )

    print(
        "Saved coefficients to:",
        coefficients_path,
    )


if __name__ == "__main__":
    main()
