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
from crypto_volatility.evaluation.walk_forward import (
    tune_ridge_alpha,
)
from crypto_volatility.features.build_features import (
    CORE_PLUS_DOWNSIDE_FEATURES,
    TARGET_COLUMN,
)
from crypto_volatility.models.ridge import (
    fit_ridge_model,
    predict_ridge,
)
from crypto_volatility.utils.config import load_config

RIDGE_ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

N_CV_SPLITS = 5
CV_GAP = 1


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
    best_alpha: float,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, str | float | int]:
    """
    Evaluate tuned Ridge on outer validation data.
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
        "model": "ridge_tuned",
        "feature_set": "core_plus_downside_7",
        "n_features": len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        "best_alpha": best_alpha,
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


def extract_coefficients(
    asset: str,
    model,
) -> list[dict[str, str | float]]:
    """
    Extract standardized Ridge coefficients.
    """
    ridge = model.named_steps[
        "ridge"
    ]

    rows = []

    for feature, coefficient in zip(
        CORE_PLUS_DOWNSIDE_FEATURES,
        ridge.coef_,
        strict=True,
    ):
        rows.append(
            {
                "asset": asset,
                "feature": feature,
                "coefficient": float(
                    coefficient
                ),
                "absolute_coefficient": abs(
                    float(coefficient)
                ),
            }
        )

    return rows


def evaluate_asset(
    asset: str,
    df: pd.DataFrame,
) -> tuple[
    dict[str, str | float | int],
    list[dict[str, str | float]],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Tune Ridge inside outer training data, then evaluate once
    on outer validation data.
    """
    outer_split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=1,
        )
    )

    train = (
        outer_split.train
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        outer_split.validation
        .copy()
        .reset_index(drop=True)
    )

    cv_result = tune_ridge_alpha(
        train_df=train,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        alphas=RIDGE_ALPHAS,
        n_splits=N_CV_SPLITS,
        gap=CV_GAP,
    )

    best_alpha = (
        cv_result.best_alpha
    )

    model = fit_ridge_model(
        train_df=train,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        alpha=best_alpha,
    )

    predictions = predict_ridge(
        model=model,
        df=validation,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
    )

    metrics = evaluate_forecast(
        asset=asset,
        best_alpha=best_alpha,
        y_true=validation[TARGET_COLUMN],
        y_pred=predictions,
    )

    coefficients = extract_coefficients(
        asset=asset,
        model=model,
    )

    fold_metrics = (
        cv_result.fold_metrics
        .copy()
    )

    fold_metrics.insert(
        0,
        "asset",
        asset,
    )

    cv_summary = (
        cv_result.summary
        .copy()
    )

    cv_summary.insert(
        0,
        "asset",
        asset,
    )

    return (
        metrics,
        coefficients,
        fold_metrics,
        cv_summary,
    )


def main() -> None:
    """
    Tune Ridge alpha using training-only purged time-series CV,
    then evaluate the selected model once on outer validation.
    """
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    metric_rows = []
    coefficient_rows = []
    fold_frames = []
    summary_frames = []

    for asset, df in datasets.items():
        print(
            f"\nTuning Ridge for {asset}..."
        )

        (
            metrics,
            coefficients,
            fold_metrics,
            cv_summary,
        ) = evaluate_asset(
            asset,
            df,
        )

        metric_rows.append(
            metrics
        )

        coefficient_rows.extend(
            coefficients
        )

        fold_frames.append(
            fold_metrics
        )

        summary_frames.append(
            cv_summary
        )

        print(
            f"{asset} best alpha: "
            f"{metrics['best_alpha']}"
        )

    metrics = pd.DataFrame(
        metric_rows
    ).sort_values(
        [
            "rmse",
            "asset",
        ]
    ).reset_index(
        drop=True
    )

    coefficients = pd.DataFrame(
        coefficient_rows
    ).sort_values(
        [
            "asset",
            "absolute_coefficient",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    fold_metrics = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    cv_summary = pd.concat(
        summary_frames,
        ignore_index=True,
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
        / "ridge_tuned_validation_metrics.csv"
    )

    coefficients_path = (
        output_dir
        / "ridge_tuned_coefficients.csv"
    )

    folds_path = (
        output_dir
        / "ridge_cv_fold_metrics.csv"
    )

    summary_path = (
        output_dir
        / "ridge_cv_summary.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    fold_metrics.to_csv(
        folds_path,
        index=False,
    )

    cv_summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nTuned Ridge validation metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nTraining-only Ridge CV summary:\n"
    )

    print(
        cv_summary.to_string(
            index=False
        )
    )

    print(
        "\nTuned Ridge coefficients:\n"
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
        "Saved CV summary to:",
        summary_path,
    )


if __name__ == "__main__":
    main()