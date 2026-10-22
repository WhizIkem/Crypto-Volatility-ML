from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper


def fit_var_model(
    train_df: pd.DataFrame,
    columns: list[str],
    maxlags: int = 5,
    ic: str | None = "aic",
) -> VARResultsWrapper:
    """
    Fit a Vector Autoregression model to multiple time series.

    Parameters
    ----------
    train_df:
        Chronologically ordered training data.

    columns:
        Variables included in the VAR system.

    maxlags:
        Maximum lag order considered.

    ic:
        Information criterion used to select lag order.
        Common choices are "aic", "bic", "hqic", or None.

    Returns
    -------
    VARResultsWrapper
        Fitted statsmodels VAR result.
    """
    if not columns:
        raise ValueError(
            "columns cannot be empty"
        )

    if maxlags < 1:
        raise ValueError(
            "maxlags must be at least 1"
        )

    missing_columns = (
        set(columns)
        - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing VAR columns: "
            f"{sorted(missing_columns)}"
        )

    data = train_df[
        columns
    ].copy()

    if data.isna().any().any():
        raise ValueError(
            "VAR training data cannot contain missing values"
        )

    if not np.isfinite(
        data.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "VAR training data must contain only finite values"
        )

    if len(data) <= maxlags:
        raise ValueError(
            "VAR training data must contain more rows than maxlags"
        )

    model = VAR(
        data
    )

    result = model.fit(
        maxlags=maxlags,
        ic=ic,
    )

    return result


def forecast_var_one_step(
    model: VARResultsWrapper,
    history_df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Produce a one-step-ahead VAR forecast.

    The history must contain at least the fitted lag order.
    """
    if not columns:
        raise ValueError(
            "columns cannot be empty"
        )

    missing_columns = (
        set(columns)
        - set(history_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing VAR forecast columns: "
            f"{sorted(missing_columns)}"
        )

    lag_order = model.k_ar

    if lag_order < 1:
        raise ValueError(
            "Fitted VAR model has no autoregressive lags"
        )

    if len(history_df) < lag_order:
        raise ValueError(
            "Insufficient history for VAR forecast"
        )

    history = (
        history_df[
            columns
        ]
        .iloc[
            -lag_order:
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        history
    ).all():
        raise ValueError(
            "VAR forecast history must contain only finite values"
        )

    forecast = model.forecast(
        history,
        steps=1,
    )[0]

    return pd.Series(
        forecast,
        index=columns,
        name="var_forecast",
        dtype=float,
    )


def extract_var_coefficients(
    model: VARResultsWrapper,
) -> pd.DataFrame:
    """
    Return VAR coefficients in tidy tabular form.
    """
    params = model.params.copy()

    params.index.name = "term"

    return (
        params
        .reset_index()
        .melt(
            id_vars="term",
            var_name="equation",
            value_name="coefficient",
        )
    )