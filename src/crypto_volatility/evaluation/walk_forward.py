from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from crypto_volatility.evaluation.metrics import (
    root_mean_squared_error,
)
from crypto_volatility.models.lasso import (
    fit_lasso_model,
    predict_lasso,
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