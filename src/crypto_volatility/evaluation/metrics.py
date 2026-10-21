from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate mean absolute error.
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
            "No aligned observations available for MAE"
        )

    errors = (
        aligned["y_true"]
        - aligned["y_pred"]
    ).abs()

    return float(
        errors.mean()
    )


def root_mean_squared_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate root mean squared error.
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
            "No aligned observations available for RMSE"
        )

    squared_errors = (
        aligned["y_true"]
        - aligned["y_pred"]
    ).pow(2)

    return float(
        np.sqrt(
            squared_errors.mean()
        )
    )