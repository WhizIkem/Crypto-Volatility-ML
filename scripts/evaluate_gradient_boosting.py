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
    tune_gradient_boosting,
)
from crypto_volatility.features.build_features import (
    CORE_PLUS_DOWNSIDE_FEATURES,
    TARGET_COLUMN,
)
from crypto_volatility.models.gradient_boosting import (
    fit_gradient_boosting_model,
    predict_gradient_boosting,
)
from crypto_volatility.utils.config import load_config

LEARNING_RATES = [
    0.03,
    0.05,
    0.1,
]

MAX_DEPTH_VALUES = [
    1,
    2,
    3,
]

MIN_SAMPLES_LEAF_VALUES = [
    5,
    10,
    20,
]

N_ESTIMATORS = 200
N_CV_SPLITS = 5
CV_GAP = 1
RANDOM_STATE = 42


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
    best_learning_rate: float,
    best_max_depth: int,
    best_min_samples_leaf: int,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, str | float | int]:
    """
    Evaluate tuned Gradient Boosting on outer validation data.
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
        "model": "gradient_boosting_tuned",
        "feature_set": "core_plus_downside_7",
        "n_features": len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        "best_learning_rate": best_learning_rate,
        "best_max_depth": best_max_depth,
        "best_min_samples_leaf": best_min_samples_leaf,
        "n_estimators": N_ESTIMATORS,
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


def extract_feature_importances(
    asset: str,
    model,
) -> list[dict[str, str | float]]:
    """
    Extract Gradient Boosting impurity-based feature importances.
    """
    rows = []

    for feature, importance in zip(
        CORE_PLUS_DOWNSIDE_FEATURES,
        model.feature_importances_,
        strict=True,
    ):
        rows.append(
            {
                "asset": asset,
                "feature": feature,
                "importance": float(
                    importance
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
    Tune Gradient Boosting inside the outer training partition
    and evaluate the selected model once on outer validation.
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

    cv_result = tune_gradient_boosting(
        train_df=train,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        learning_rates=LEARNING_RATES,
        max_depth_values=MAX_DEPTH_VALUES,
        min_samples_leaf_values=MIN_SAMPLES_LEAF_VALUES,
        n_estimators=N_ESTIMATORS,
        n_splits=N_CV_SPLITS,
        gap=CV_GAP,
        random_state=RANDOM_STATE,
    )

    best_learning_rate = (
        cv_result.best_learning_rate
    )

    best_max_depth = (
        cv_result.best_max_depth
    )

    best_min_samples_leaf = (
        cv_result.best_min_samples_leaf
    )

    model = fit_gradient_boosting_model(
        train_df=train,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        n_estimators=N_ESTIMATORS,
        learning_rate=best_learning_rate,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples_leaf,
        random_state=RANDOM_STATE,
    )

    predictions = predict_gradient_boosting(
        model=model,
        df=validation,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
    )

    metrics = evaluate_forecast(
        asset=asset,
        best_learning_rate=best_learning_rate,
        best_max_depth=best_max_depth,
        best_min_samples_leaf=best_min_samples_leaf,
        y_true=validation[TARGET_COLUMN],
        y_pred=predictions,
    )

    importances = extract_feature_importances(
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
        importances,
        fold_metrics,
        cv_summary,
    )


def main() -> None:
    """
    Tune Gradient Boosting using training-only purged
    time-series CV, then evaluate once on outer validation.
    """
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    metric_rows = []
    importance_rows = []
    fold_frames = []
    summary_frames = []

    for asset, df in datasets.items():
        print(
            f"\nTuning Gradient Boosting for {asset}..."
        )

        (
            metrics,
            importances,
            fold_metrics,
            cv_summary,
        ) = evaluate_asset(
            asset,
            df,
        )

        metric_rows.append(
            metrics
        )

        importance_rows.extend(
            importances
        )

        fold_frames.append(
            fold_metrics
        )

        summary_frames.append(
            cv_summary
        )

        print(
            f"{asset} best learning_rate: "
            f"{metrics['best_learning_rate']}"
        )

        print(
            f"{asset} best max_depth: "
            f"{metrics['best_max_depth']}"
        )

        print(
            f"{asset} best min_samples_leaf: "
            f"{metrics['best_min_samples_leaf']}"
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

    importances = pd.DataFrame(
        importance_rows
    ).sort_values(
        [
            "asset",
            "importance",
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
        / "gradient_boosting_tuned_validation_metrics.csv"
    )

    importance_path = (
        output_dir
        / "gradient_boosting_feature_importances.csv"
    )

    folds_path = (
        output_dir
        / "gradient_boosting_cv_fold_metrics.csv"
    )

    summary_path = (
        output_dir
        / "gradient_boosting_cv_summary.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    importances.to_csv(
        importance_path,
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
        "\nTuned Gradient Boosting validation metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nTraining-only Gradient Boosting CV summary:\n"
    )

    print(
        cv_summary.to_string(
            index=False
        )
    )

    print(
        "\nGradient Boosting feature importances:\n"
    )

    print(
        importances.to_string(
            index=False
        )
    )

    print(
        "\nSaved metrics to:",
        metrics_path,
    )

    print(
        "Saved feature importances to:",
        importance_path,
    )

if __name__ == "__main__":
    main()