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
from crypto_volatility.features.build_features import TARGET_COLUMN
from crypto_volatility.models.baselines import (
    historical_mean_forecast,
    persistence_forecast,
)
from crypto_volatility.utils.config import load_config


def load_processed_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Load processed model-ready datasets."""
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
    """Calculate validation metrics for one forecast."""
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
) -> list[dict[str, str | float | int]]:
    """
    Evaluate leakage-safe naive baselines on validation data.

    The chronological splitter purges one supervised row at
    train/validation and validation/test boundaries.

    Precomputed HAR rolling features are valid baseline forecasts
    because they use only information available through time t.
    """
    split = chronological_train_validation_test_split(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    train = split.train
    validation = split.validation

    y_validation = (
        validation[TARGET_COLUMN]
        .copy()
    )

    results = []

    persistence_prediction = persistence_forecast(
        validation["log_gk_volatility"]
    )

    results.append(
        evaluate_forecast(
            asset=asset,
            model_name="persistence",
            y_true=y_validation,
            y_pred=persistence_prediction,
        )
    )

    historical_mean_prediction = (
        historical_mean_forecast(
            train[TARGET_COLUMN],
            validation.index,
        )
    )

    results.append(
        evaluate_forecast(
            asset=asset,
            model_name="historical_mean",
            y_true=y_validation,
            y_pred=historical_mean_prediction,
        )
    )

    rolling_7_prediction = (
        validation[
            "har_vol_weekly"
        ].copy()
    )

    results.append(
        evaluate_forecast(
            asset=asset,
            model_name="rolling_mean_7d",
            y_true=y_validation,
            y_pred=rolling_7_prediction,
        )
    )

    rolling_30_prediction = (
        validation[
            "har_vol_monthly"
        ].copy()
    )

    results.append(
        evaluate_forecast(
            asset=asset,
            model_name="rolling_mean_30d",
            y_true=y_validation,
            y_pred=rolling_30_prediction,
        )
    )

    return results


def main() -> None:
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    rows = []

    for asset, df in datasets.items():
        rows.extend(
            evaluate_asset(
                asset,
                df,
            )
        )

    metrics = pd.DataFrame(
        rows
    )

    metrics = metrics.sort_values(
        [
            "asset",
            "rmse",
        ]
    ).reset_index(
        drop=True
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "baseline_validation_metrics.csv"
    )

    metrics.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nLeakage-safe baseline validation metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nSaved metrics to:",
        output_path,
    )


if __name__ == "__main__":
    main()