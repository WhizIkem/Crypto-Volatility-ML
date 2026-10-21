from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from crypto_volatility.evaluation.metrics import (
    root_mean_squared_error,
)
from crypto_volatility.models.gradient_boosting import (
    fit_gradient_boosting_model,
    predict_gradient_boosting,
)
from crypto_volatility.models.lasso import (
    fit_lasso_model,
    predict_lasso,
)
from crypto_volatility.models.random_forest import (
    fit_random_forest_model,
    predict_random_forest,
)
from crypto_volatility.models.ridge import (
    fit_ridge_model,
    predict_ridge,
)


@dataclass(frozen=True)
class RidgeCVResult:
    best_alpha: float
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class LassoCVResult:
    best_alpha: float
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame

@dataclass(frozen=True)
class RandomForestCVResult:
    best_max_depth: int | None
    best_min_samples_leaf: int
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame

@dataclass(frozen=True)
class GradientBoostingCVResult:
    best_max_depth: int
    best_learning_rate: float
    best_min_samples_leaf: int
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame

def purged_time_series_splits(
    n_rows: int,
    n_splits: int = 5,
    gap: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build expanding-window chronological CV folds.

    A gap is placed between each training fold and validation
    fold to prevent forward targets from crossing the boundary.

    For the project's one-day-ahead target, gap=1 is appropriate.
    """
    if n_rows <= 0:
        raise ValueError(
            "n_rows must be greater than zero"
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2"
        )

    if gap < 0:
        raise ValueError(
            "gap cannot be negative"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
    )

    positions = np.arange(
        n_rows
    )

    return list(
        splitter.split(
            positions
        )
    )


def tune_ridge_alpha(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alphas: list[float],
    n_splits: int = 5,
    gap: int = 1,
) -> RidgeCVResult:
    """
    Select Ridge alpha using training-only purged time-series CV.
    """
    if train_df.empty:
        raise ValueError(
            "Training data cannot be empty"
        )

    if not alphas:
        raise ValueError(
            "alphas cannot be empty"
        )

    if any(
        alpha <= 0
        for alpha in alphas
    ):
        raise ValueError(
            "All alpha values must be greater than zero"
        )

    splits = purged_time_series_splits(
        n_rows=len(train_df),
        n_splits=n_splits,
        gap=gap,
    )

    rows = []

    for alpha in alphas:
        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            splits,
            start=1,
        ):
            fold_train = (
                train_df
                .iloc[train_index]
                .copy()
                .reset_index(drop=True)
            )

            fold_validation = (
                train_df
                .iloc[validation_index]
                .copy()
                .reset_index(drop=True)
            )

            model = fit_ridge_model(
                train_df=fold_train,
                feature_columns=feature_columns,
                target_column=target_column,
                alpha=alpha,
            )

            predictions = predict_ridge(
                model=model,
                df=fold_validation,
                feature_columns=feature_columns,
            )

            rmse = root_mean_squared_error(
                fold_validation[
                    target_column
                ],
                predictions,
            )

            rows.append(
                {
                    "alpha": alpha,
                    "fold": fold_number,
                    "train_observations": len(
                        fold_train
                    ),
                    "validation_observations": len(
                        fold_validation
                    ),
                    "rmse": rmse,
                }
            )

    fold_metrics = pd.DataFrame(
        rows
    )

    summary = (
        fold_metrics
        .groupby(
            "alpha",
            as_index=False,
        )
        .agg(
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
        )
        .sort_values(
            [
                "mean_rmse",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )

    best_alpha = float(
        summary.iloc[0][
            "alpha"
        ]
    )

    return RidgeCVResult(
        best_alpha=best_alpha,
        fold_metrics=fold_metrics,
        summary=summary,
    )


def tune_lasso_alpha(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alphas: list[float],
    n_splits: int = 5,
    gap: int = 1,
) -> LassoCVResult:
    """
    Select Lasso alpha using training-only purged time-series CV.

    Each inner fold fits its own StandardScaler and Lasso model,
    so preprocessing statistics never use inner-validation data.
    """
    if train_df.empty:
        raise ValueError(
            "Training data cannot be empty"
        )

    if not alphas:
        raise ValueError(
            "alphas cannot be empty"
        )

    if any(
        alpha <= 0
        for alpha in alphas
    ):
        raise ValueError(
            "All alpha values must be greater than zero"
        )

    splits = purged_time_series_splits(
        n_rows=len(train_df),
        n_splits=n_splits,
        gap=gap,
    )

    rows = []

    for alpha in alphas:
        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            splits,
            start=1,
        ):
            fold_train = (
                train_df
                .iloc[train_index]
                .copy()
                .reset_index(drop=True)
            )

            fold_validation = (
                train_df
                .iloc[validation_index]
                .copy()
                .reset_index(drop=True)
            )

            model = fit_lasso_model(
                train_df=fold_train,
                feature_columns=feature_columns,
                target_column=target_column,
                alpha=alpha,
            )

            predictions = predict_lasso(
                model=model,
                df=fold_validation,
                feature_columns=feature_columns,
            )

            rmse = root_mean_squared_error(
                fold_validation[
                    target_column
                ],
                predictions,
            )

            rows.append(
                {
                    "alpha": alpha,
                    "fold": fold_number,
                    "train_observations": len(
                        fold_train
                    ),
                    "validation_observations": len(
                        fold_validation
                    ),
                    "rmse": rmse,
                }
            )

    fold_metrics = pd.DataFrame(
        rows
    )

    summary = (
        fold_metrics
        .groupby(
            "alpha",
            as_index=False,
        )
        .agg(
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
        )
        .sort_values(
            [
                "mean_rmse",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )

    best_alpha = float(
        summary.iloc[0][
            "alpha"
        ]
    )

    return LassoCVResult(
        best_alpha=best_alpha,
        fold_metrics=fold_metrics,
        summary=summary,
    )

def tune_random_forest(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_depth_values: list[int | None],
    min_samples_leaf_values: list[int],
    n_estimators: int = 300,
    n_splits: int = 5,
    gap: int = 1,
    random_state: int = 42,
) -> RandomForestCVResult:
    """
    Tune Random Forest hyperparameters using training-only
    purged time-series cross-validation.
    """
    if train_df.empty:
        raise ValueError(
            "Training data cannot be empty"
        )

    if not max_depth_values:
        raise ValueError(
            "max_depth_values cannot be empty"
        )

    if not min_samples_leaf_values:
        raise ValueError(
            "min_samples_leaf_values cannot be empty"
        )

    if any(
        value is not None
        and value < 1
        for value in max_depth_values
    ):
        raise ValueError(
            "max_depth values must be positive or None"
        )

    if any(
        value < 1
        for value in min_samples_leaf_values
    ):
        raise ValueError(
            "min_samples_leaf values must be at least 1"
        )

    splits = purged_time_series_splits(
        n_rows=len(train_df),
        n_splits=n_splits,
        gap=gap,
    )

    rows = []

    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:
            for fold_number, (
                train_index,
                validation_index,
            ) in enumerate(
                splits,
                start=1,
            ):
                fold_train = (
                    train_df
                    .iloc[train_index]
                    .copy()
                    .reset_index(drop=True)
                )

                fold_validation = (
                    train_df
                    .iloc[validation_index]
                    .copy()
                    .reset_index(drop=True)
                )

                model = fit_random_forest_model(
                    train_df=fold_train,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=1.0,
                    random_state=random_state,
                )

                predictions = predict_random_forest(
                    model=model,
                    df=fold_validation,
                    feature_columns=feature_columns,
                )

                rmse = root_mean_squared_error(
                    fold_validation[
                        target_column
                    ],
                    predictions,
                )

                rows.append(
                    {
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "fold": fold_number,
                        "train_observations": len(
                            fold_train
                        ),
                        "validation_observations": len(
                            fold_validation
                        ),
                        "rmse": rmse,
                    }
                )

    fold_metrics = pd.DataFrame(
        rows
    )

    summary = (
        fold_metrics
        .groupby(
            [
                "max_depth",
                "min_samples_leaf",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
        )
        .sort_values(
            [
                "mean_rmse",
                "std_rmse",
            ]
        )
        .reset_index(drop=True)
    )

    best_row = summary.iloc[0]

    best_max_depth = (
        None
        if pd.isna(
            best_row["max_depth"]
        )
        else int(
            best_row["max_depth"]
        )
    )

    best_min_samples_leaf = int(
        best_row[
            "min_samples_leaf"
        ]
    )

    return RandomForestCVResult(
        best_max_depth=best_max_depth,
        best_min_samples_leaf=best_min_samples_leaf,
        fold_metrics=fold_metrics,
        summary=summary,
    )

def tune_gradient_boosting(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    learning_rates: list[float],
    max_depth_values: list[int],
    min_samples_leaf_values: list[int],
    n_estimators: int = 200,
    n_splits: int = 5,
    gap: int = 1,
    random_state: int = 42,
) -> GradientBoostingCVResult:
    """
    Tune Gradient Boosting using training-only purged
    time-series cross-validation.
    """
    if train_df.empty:
        raise ValueError(
            "Training data cannot be empty"
        )

    if not learning_rates:
        raise ValueError(
            "learning_rates cannot be empty"
        )

    if not max_depth_values:
        raise ValueError(
            "max_depth_values cannot be empty"
        )

    if not min_samples_leaf_values:
        raise ValueError(
            "min_samples_leaf_values cannot be empty"
        )

    if any(
        value <= 0
        for value in learning_rates
    ):
        raise ValueError(
            "learning_rates must be greater than zero"
        )

    if any(
        value < 1
        for value in max_depth_values
    ):
        raise ValueError(
            "max_depth values must be at least 1"
        )

    if any(
        value < 1
        for value in min_samples_leaf_values
    ):
        raise ValueError(
            "min_samples_leaf values must be at least 1"
        )

    splits = purged_time_series_splits(
        n_rows=len(train_df),
        n_splits=n_splits,
        gap=gap,
    )

    rows = []

    for learning_rate in learning_rates:
        for max_depth in max_depth_values:
            for min_samples_leaf in min_samples_leaf_values:
                for fold_number, (
                    train_index,
                    validation_index,
                ) in enumerate(
                    splits,
                    start=1,
                ):
                    fold_train = (
                        train_df
                        .iloc[train_index]
                        .copy()
                        .reset_index(drop=True)
                    )

                    fold_validation = (
                        train_df
                        .iloc[validation_index]
                        .copy()
                        .reset_index(drop=True)
                    )

                    model = fit_gradient_boosting_model(
                        train_df=fold_train,
                        feature_columns=feature_columns,
                        target_column=target_column,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                    )

                    predictions = predict_gradient_boosting(
                        model=model,
                        df=fold_validation,
                        feature_columns=feature_columns,
                    )

                    rmse = root_mean_squared_error(
                        fold_validation[
                            target_column
                        ],
                        predictions,
                    )

                    rows.append(
                        {
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "fold": fold_number,
                            "train_observations": len(
                                fold_train
                            ),
                            "validation_observations": len(
                                fold_validation
                            ),
                            "rmse": rmse,
                        }
                    )

    fold_metrics = pd.DataFrame(
        rows
    )

    summary = (
        fold_metrics
        .groupby(
            [
                "learning_rate",
                "max_depth",
                "min_samples_leaf",
            ],
            as_index=False,
        )
        .agg(
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
        )
        .sort_values(
            [
                "mean_rmse",
                "std_rmse",
            ]
        )
        .reset_index(drop=True)
    )

    best_row = summary.iloc[0]

    return GradientBoostingCVResult(
        best_learning_rate=float(
            best_row[
                "learning_rate"
            ]
        ),
        best_max_depth=int(
            best_row[
                "max_depth"
            ]
        ),
        best_min_samples_leaf=int(
            best_row[
                "min_samples_leaf"
            ]
        ),
        fold_metrics=fold_metrics,
        summary=summary,
    )