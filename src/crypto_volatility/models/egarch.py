from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

RETURN_SCALE = 100.0


def fit_egarch_model(
    returns: pd.Series,
) -> ARCHModelResult:
    """
    Fit an EGARCH(1,1,1) model to daily log returns.

    Returns are multiplied by 100 during estimation for
    numerical stability. Forecasts are later converted back
    to decimal-return units.

    The model specification is:

    - constant conditional mean;
    - EGARCH(1,1,1) variance;
    - Student-t innovations.

    Parameters
    ----------
    returns:
        Daily log-return series.

    Returns
    -------
    ARCHModelResult
        Fitted EGARCH model.

    Raises
    ------
    ValueError
        If the return series is empty, too short, or contains
        non-finite values.
    """
    clean_returns = returns.dropna()

    if clean_returns.empty:
        raise ValueError(
            "Return series cannot be empty"
        )

    if len(clean_returns) < 100:
        raise ValueError(
            "EGARCH requires at least 100 return observations"
        )

    if not np.isfinite(
        clean_returns.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Return series must contain only finite values"
        )

    scaled_returns = (
        clean_returns
        * RETURN_SCALE
    )

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
        rescale=False,
    )

    return model.fit(
        disp="off",
        options={
            "maxiter": 2000,  # Maximum number of iterations for the optimizer
        }
    )


def forecast_egarch_variance(
    model: ARCHModelResult,
) -> float:
    """
    Produce a one-step-ahead variance forecast.

    The returned variance is converted from percentage-return
    units back to decimal-return units.

    Parameters
    ----------
    model:
        Fitted EGARCH model.

    Returns
    -------
    float
        One-step-ahead conditional return variance.
    """
    forecast = model.forecast(
        horizon=1,
        reindex=False,
    )

    scaled_variance = float(
        forecast.variance.iloc[
            -1,
            0,
        ]
    )

    variance = (
        scaled_variance
        / RETURN_SCALE**2
    )

    if (
        not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError(
            "EGARCH produced an invalid variance forecast"
        )

    return variance


def forecast_egarch_volatility(
    model: ARCHModelResult,
) -> float:
    """
    Produce a one-step-ahead volatility forecast.

    Returns
    -------
    float
        Conditional volatility in decimal-return units.
    """
    variance = forecast_egarch_variance(
        model
    )

    return float(
        np.sqrt(variance)
    )


def forecast_egarch_log_volatility(
    model: ARCHModelResult,
) -> float:
    """
    Produce one-step-ahead log volatility.

    This transformation places the EGARCH forecast on a scale
    comparable to the project's log-volatility forecasting target.

    Returns
    -------
    float
        Natural logarithm of forecast conditional volatility.
    """
    volatility = (
        forecast_egarch_volatility(
            model
        )
    )

    return float(
        np.log(volatility)
    )

def expanding_window_egarch_forecasts(
    historical_returns: pd.Series,
    validation_returns: pd.Series,
) -> pd.Series:
    """
    Generate leakage-safe expanding-window one-step-ahead
    EGARCH forecasts.

    For each validation observation at time t:

    1. the return observed at t becomes available;
    2. it is appended to the historical information set;
    3. EGARCH is fitted using only returns known through t;
    4. volatility for t + 1 is forecast;
    5. the forecast is returned on the log-volatility scale.
    """
    if validation_returns.empty:
        raise ValueError(
            "Validation return series cannot be empty"
        )

    history = (
        historical_returns
        .dropna()
        .astype(float)
        .reset_index(drop=True)
    )

    if len(history) < 100:
        raise ValueError(
            "Historical return series must contain at least "
            "100 observations"
        )

    if not np.isfinite(
        history.to_numpy()
    ).all():
        raise ValueError(
            "Historical returns must contain only finite values"
        )

    prediction_index = (
        validation_returns.index.copy()
    )

    predictions = []

    for current_return in validation_returns:
        if not np.isfinite(current_return):
            raise ValueError(
                "Validation returns must contain only "
                "finite values"
            )

        history = pd.concat(
            [
                history,
                pd.Series(
                    [float(current_return)]
                ),
            ],
            ignore_index=True,
        )

        model = fit_egarch_model(
            history
        )

        if model.convergence_flag != 0:
            raise RuntimeError(
                "EGARCH failed to converge during "
                "walk-forward evaluation"
            )

        prediction = (
            forecast_egarch_log_volatility(
                model
            )
        )

        predictions.append(
            prediction
        )

    return pd.Series(
        predictions,
        index=prediction_index,
        name="egarch_prediction",
        dtype=float,
    )