from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_volatility.models.production import (
    ProductionForecasts,
)
from crypto_volatility.portfolio.allocation import (
    build_portfolio_allocation,
)
from crypto_volatility.portfolio.backtest import (
    calculate_turnover,
)

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]


@dataclass(frozen=True)
class TradingDecision:
    """
    Final portfolio decision produced by the production pipeline.
    """

    forecast_log_volatility: pd.Series
    target_weights: pd.Series
    current_weights: pd.Series
    trade_weights: pd.Series
    turnover: float
    should_rebalance: bool
    risky_exposure: float
    cash_weight: float
    forecast_portfolio_volatility: float


def build_trading_decision(
    *,
    forecasts: ProductionForecasts,
    historical_returns: pd.DataFrame,
    current_weights: pd.Series,
    annual_volatility_target: float = 0.20,
    max_asset_weight: float = 0.40,
    correlation_lookback: int = 30,
    max_risky_exposure: float = 1.0,
    turnover_threshold: float = 0.05,
) -> TradingDecision:
    """
    Convert production volatility forecasts into an executable
    portfolio decision.
    """
    if annual_volatility_target <= 0:
        raise ValueError(
            "annual_volatility_target must be positive"
        )

    if not 0 < max_asset_weight <= 1:
        raise ValueError(
            "max_asset_weight must be in (0, 1]"
        )

    if correlation_lookback < 2:
        raise ValueError(
            "correlation_lookback must be at least 2"
        )

    if not 0 < max_risky_exposure <= 1:
        raise ValueError(
            "max_risky_exposure must be in (0, 1]"
        )

    if not 0 <= turnover_threshold <= 1:
        raise ValueError(
            "turnover_threshold must be in [0, 1]"
        )

    forecast_series = (
        forecasts
        .to_series()
        .reindex(
            ASSETS
        )
    )

    if forecast_series.isna().any():
        raise ValueError(
            "Forecasts missing required assets"
        )

    missing_returns = (
        set(
            ASSETS
        )
        - set(
            historical_returns.columns
        )
    )

    if missing_returns:
        raise ValueError(
            "Historical returns missing assets: "
            f"{sorted(missing_returns)}"
        )

    if len(
        historical_returns
    ) < correlation_lookback:
        raise ValueError(
            "Not enough historical returns for correlation lookback"
        )

    missing_current = (
        set(
            ASSETS
        )
        - set(
            current_weights.index
        )
    )

    if missing_current:
        raise ValueError(
            "Current weights missing assets: "
            f"{sorted(missing_current)}"
        )

    current_risky_weights = (
        current_weights[
            ASSETS
        ]
        .astype(
            float
        )
    )

    if not np.isfinite(
        current_risky_weights.to_numpy()
    ).all():
        raise ValueError(
            "Current weights contain non-finite values"
        )

    if (
        current_risky_weights
        < 0
    ).any():
        raise ValueError(
            "Current weights cannot be negative"
        )

    daily_target = (
        annual_volatility_target
        / np.sqrt(
            365.0
        )
    )

    allocation = build_portfolio_allocation(
        log_volatility_forecasts=(
            forecast_series
        ),
        historical_returns=(
            historical_returns[
                ASSETS
            ]
        ),
        max_asset_weight=(
            max_asset_weight
        ),
        target_portfolio_volatility=(
            daily_target
        ),
        correlation_lookback=(
            correlation_lookback
        ),
        max_risky_exposure=(
            max_risky_exposure
        ),
    )

    target_weights = (
        allocation
        .risky_weights
        .reindex(
            ASSETS
        )
        .astype(
            float
        )
    )

    current_cash_weight = float(
        max(
            1.0
            - current_risky_weights.sum(),
            0.0,
        )
    )

    target_cash_weight = float(
        allocation.cash_weight
    )

    current_full = pd.Series(
        {
            **current_risky_weights.to_dict(),
            "CASH": current_cash_weight,
        },
        dtype=float,
    )

    target_full = pd.Series(
        {
            **target_weights.to_dict(),
            "CASH": target_cash_weight,
        },
        dtype=float,
    )

    turnover = (
        calculate_turnover(
            current_weights=(
                target_full
            ),
            previous_weights=(
                current_full
            ),
        )
    )

    should_rebalance = bool(
        turnover
        >= turnover_threshold
    )

    if should_rebalance:
        trade_weights = (
            target_weights
            - current_risky_weights
        )
    else:
        trade_weights = pd.Series(
            0.0,
            index=ASSETS,
            dtype=float,
        )

    return TradingDecision(
        forecast_log_volatility=(
            forecast_series
        ),
        target_weights=(
            target_weights
        ),
        current_weights=(
            current_risky_weights
        ),
        trade_weights=(
            trade_weights
        ),
        turnover=float(
            turnover
        ),
        should_rebalance=(
            should_rebalance
        ),
        risky_exposure=float(
            allocation.risky_exposure
        ),
        cash_weight=float(
            allocation.cash_weight
        ),
        forecast_portfolio_volatility=float(
            allocation.forecast_portfolio_volatility
        ),
    )