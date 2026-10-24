from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_volatility.portfolio.risk import (
    apply_max_weight_cap,
    apply_total_exposure,
    build_forecast_covariance,
    calculate_cash_weight,
    calculate_portfolio_volatility,
    calculate_volatility_target_exposure,
)


@dataclass(frozen=True)
class PortfolioAllocation:
    """
    Final daily portfolio allocation.

    risky_weights:
        Final BTC/ETH/SOL/BNB portfolio weights after all
        risk controls.

    cash_weight:
        Remaining capital held in cash.

    unconstrained_risky_weights:
        Raw inverse-volatility weights before concentration
        and total-exposure controls.

    capped_risky_weights:
        Fully invested crypto weights after the per-asset cap.

    forecast_portfolio_volatility:
        Forecast portfolio volatility before total-exposure scaling.

    risky_exposure:
        Total fraction of capital allocated to crypto.
    """

    risky_weights: pd.Series
    cash_weight: float
    unconstrained_risky_weights: pd.Series
    capped_risky_weights: pd.Series
    forecast_portfolio_volatility: float
    risky_exposure: float

  
def log_volatility_to_volatility(
    log_volatility: pd.Series,
) -> pd.Series:
    """
    Convert log-volatility forecasts back to volatility levels.
    """
    if log_volatility.empty:
        raise ValueError(
            "log_volatility cannot be empty"
        )

    values = log_volatility.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "log_volatility must contain only finite values"
        )

    volatility = np.exp(
        values
    )

    return pd.Series(
        volatility,
        index=log_volatility.index,
        name="predicted_volatility",
        dtype=float,
    )


def inverse_volatility_weights(
    volatility: pd.Series,
    minimum_volatility: float = 1e-8,
) -> pd.Series:
    """
    Convert positive volatility forecasts into normalized
    inverse-volatility portfolio weights.

    Lower forecast volatility receives a larger portfolio weight.

    Weights sum to 1.
    """
    if volatility.empty:
        raise ValueError(
            "volatility cannot be empty"
        )

    if minimum_volatility <= 0:
        raise ValueError(
            "minimum_volatility must be greater than zero"
        )

    values = volatility.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "volatility must contain only finite values"
        )

    if (
        values <= 0
    ).any():
        raise ValueError(
            "volatility must contain only positive values"
        )

    safe_volatility = volatility.clip(
        lower=minimum_volatility
    )

    inverse_volatility = (
        1.0
        / safe_volatility
    )

    total_inverse_volatility = (
        inverse_volatility.sum()
    )

    if (
        not np.isfinite(
            total_inverse_volatility
        )
        or total_inverse_volatility <= 0
    ):
        raise ValueError(
            "Could not normalize inverse-volatility weights"
        )

    weights = (
        inverse_volatility
        / total_inverse_volatility
    )

    return pd.Series(
        weights,
        index=volatility.index,
        name="portfolio_weight",
        dtype=float,
    )


def weights_from_log_volatility_forecasts(
    log_volatility_forecasts: pd.Series,
) -> pd.Series:
    """
    Convert model forecasts in log-volatility units directly
    into inverse-volatility portfolio weights.
    """
    volatility = (
        log_volatility_to_volatility(
            log_volatility_forecasts
        )
    )

    return inverse_volatility_weights(
        volatility
    )

def build_portfolio_allocation(
    log_volatility_forecasts: pd.Series,
    historical_returns: pd.DataFrame,
    *,
    max_asset_weight: float = 0.40,
    target_portfolio_volatility: float = 0.03,
    correlation_lookback: int = 30,
    max_risky_exposure: float = 1.0,
) -> PortfolioAllocation:
    """
    Build the complete volatility-driven portfolio allocation.

    Parameters
    ----------
    log_volatility_forecasts:
        Next-period log-volatility forecasts indexed by asset.

    historical_returns:
        Historical asset returns available at the forecast date.
        One column per asset.

    max_asset_weight:
        Maximum fully invested crypto weight per asset.

    target_portfolio_volatility:
        Desired next-period portfolio volatility.

    correlation_lookback:
        Historical return lookback used for correlation estimation.

    max_risky_exposure:
        Maximum total crypto exposure.
        Use 1.0 for no leverage.

    Returns
    -------
    PortfolioAllocation
        Final risky weights, cash allocation, and intermediate
        portfolio-risk information.
    """
    volatility_forecasts = (
        log_volatility_to_volatility(
            log_volatility_forecasts
        )
    )

    unconstrained_weights = (
        inverse_volatility_weights(
            volatility_forecasts
        )
    )

    capped_weights = (
        apply_max_weight_cap(
            unconstrained_weights,
            max_weight=max_asset_weight,
        )
    )

    covariance = (
        build_forecast_covariance(
            forecast_volatility=volatility_forecasts,
            historical_returns=historical_returns,
            lookback=correlation_lookback,
        )
    )

    forecast_portfolio_volatility = (
        calculate_portfolio_volatility(
            weights=capped_weights,
            covariance=covariance,
        )
    )

    risky_exposure = (
        calculate_volatility_target_exposure(
            forecast_portfolio_volatility=(
                forecast_portfolio_volatility
            ),
            target_volatility=target_portfolio_volatility,
            max_exposure=max_risky_exposure,
        )
    )

    final_risky_weights = (
        apply_total_exposure(
            risky_weights=capped_weights,
            exposure=risky_exposure,
        )
    )

    cash_weight = (
        calculate_cash_weight(
            final_risky_weights
        )
    )

    return PortfolioAllocation(
        risky_weights=final_risky_weights,
        cash_weight=cash_weight,
        unconstrained_risky_weights=unconstrained_weights,
        capped_risky_weights=capped_weights,
        forecast_portfolio_volatility=(
            forecast_portfolio_volatility
        ),
        risky_exposure=risky_exposure,
    )