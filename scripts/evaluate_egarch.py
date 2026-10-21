from __future__ import annotations

from pathlib import Path

import numpy as np
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
from crypto_volatility.models.egarch import (
    expanding_window_egarch_forecasts,
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
    Calculate validation metrics for one forecast.
    """
    aligned = pd.concat(
        [
            y_true.rename("y_true"),
            y_pred.rename("y_pred"),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No aligned observations available for evaluation"
        )

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


def get_pre_validation_returns(
    df: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.Series:
    """
    Return all observed returns strictly before validation starts.

    This deliberately includes the purged train/validation
    boundary observation.

    The purged row cannot be used as a supervised training sample,
    because its target crosses the split boundary. Its return,
    however, is historical market information and is legitimately
    available before the first validation forecast.
    """
    validation_start = (
        validation["timestamp"]
        .iloc[0]
    )

    historical_returns = (
        df.loc[
            df["timestamp"] < validation_start,
            "log_return",
        ]
        .dropna()
        .copy()
        .reset_index(drop=True)
    )

    if historical_returns.empty:
        raise ValueError(
            "No historical returns available before validation"
        )

    if not np.isfinite(
        historical_returns.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Historical returns contain non-finite values"
        )

    return historical_returns


def evaluate_asset(
    asset: str,
    df: pd.DataFrame,
) -> dict[str, str | float | int]:
    """
    Run leakage-safe expanding-window EGARCH validation.
    """
    split = chronological_train_validation_test_split(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    validation = (
        split.validation
        .copy()
        .reset_index(drop=True)
    )

    historical_returns = (
        get_pre_validation_returns(
            df,
            validation,
        )
    )

    validation_returns = (
        validation["log_return"]
        .copy()
    )

    y_validation = (
        validation[TARGET_COLUMN]
        .copy()
    )

    print(
        f"\n{asset}: starting "
        f"{len(validation)} expanding-window forecasts..."
    )

    predictions = (
        expanding_window_egarch_forecasts(
            historical_returns=historical_returns,
            validation_returns=validation_returns,
        )
    )

    print(
        f"{asset}: completed "
        f"{len(predictions)} forecasts."
    )

    return evaluate_forecast(
        asset=asset,
        model_name="egarch_111_t",
        y_true=y_validation,
        y_pred=predictions,
    )


def main() -> None:
    """
    Evaluate EGARCH(1,1,1) with Student-t innovations using a
    leakage-safe expanding-window validation procedure.
    """
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    metric_rows = []

    for asset, df in datasets.items():
        result = evaluate_asset(
            asset,
            df,
        )

        metric_rows.append(
            result
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics = metrics.sort_values(
        [
            "rmse",
            "asset",
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
        / "egarch_validation_metrics.csv"
    )

    metrics.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nLeakage-safe EGARCH validation metrics:\n"
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