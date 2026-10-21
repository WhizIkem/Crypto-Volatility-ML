from __future__ import annotations

import pandas as pd


def persistence_forecast(
    current_volatility: pd.Series,
) -> pd.Series:
    """
    Forecast next-day volatility using today's volatility.

    Parameters
    ----------
    current_volatility:
        Volatility observed at time t.

    Returns
    -------
    pd.Series
        Forecast for volatility at time t+1.
    """
    return current_volatility.copy()


def historical_mean_forecast(
    train_target: pd.Series,
    forecast_index: pd.Index,
) -> pd.Series:
    """
    Forecast using the mean target observed in the training set.

    Parameters
    ----------
    train_target:
        Training target values only.

    forecast_index:
        Index for the forecast period.

    Returns
    -------
    pd.Series
        Constant forecast equal to the training-target mean.
    """
    clean_target = train_target.dropna()

    if clean_target.empty:
        raise ValueError(
            "train_target cannot be empty"
        )

    mean_value = clean_target.mean()

    return pd.Series(
        mean_value,
        index=forecast_index,
        dtype=float,
    )


def rolling_mean_forecast(
    volatility: pd.Series,
    window: int,
) -> pd.Series:
    """
    Forecast next-day volatility using a trailing rolling mean.

    The forecast at time t uses volatility observations available
    up to and including time t.

    Parameters
    ----------
    volatility:
        Volatility observed through time.

    window:
        Trailing rolling-window length.

    Returns
    -------
    pd.Series
        Rolling-mean volatility forecast.
    """
    if window < 1:
        raise ValueError(
            "window must be at least 1"
        )

    return (
        volatility
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )