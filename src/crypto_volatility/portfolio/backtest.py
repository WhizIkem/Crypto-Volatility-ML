from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """
    Backtest outputs.

    daily:
        Daily portfolio-level results.

    weights:
        Daily risky-asset weights.

    metrics:
        Summary performance metrics.
    """

    daily: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float]


def calculate_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Turnover is defined as:

        0.5 * sum(abs(w_t - w_{t-1}))

    This avoids double-counting sells and buys.
    """
    if not current_weights.index.equals(
        previous_weights.index
    ):
        raise ValueError(
            "Weight indices must match"
        )

    current = current_weights.to_numpy(
        dtype=float
    )

    previous = previous_weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        current
    ).all():
        raise ValueError(
            "current_weights must contain only finite values"
        )

    if not np.isfinite(
        previous
    ).all():
        raise ValueError(
            "previous_weights must contain only finite values"
        )

    return float(
        0.5
        * np.abs(
            current
            - previous
        ).sum()
    )


def calculate_transaction_cost(
    turnover: float,
    transaction_cost_bps: float,
) -> float:
    """
    Convert turnover into a proportional transaction cost.

    Example
    -------
    10 bps = 0.001

    cost = turnover * 0.001
    """
    if turnover < 0:
        raise ValueError(
            "turnover cannot be negative"
        )

    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative"
        )

    cost_rate = (
        transaction_cost_bps
        / 10_000.0
    )

    return float(
        turnover
        * cost_rate
    )


def calculate_portfolio_return(
    weights: pd.Series,
    asset_returns: pd.Series,
    cash_weight: float = 0.0,
    cash_return: float = 0.0,
) -> float:
    """
    Calculate one-period portfolio return.

    Risky weights are applied to next-period asset returns.

    Cash contributes:

        cash_weight * cash_return
    """
    if not weights.index.equals(
        asset_returns.index
    ):
        raise ValueError(
            "Weight and return indices must match"
        )

    if not 0 <= cash_weight <= 1:
        raise ValueError(
            "cash_weight must be in [0, 1]"
        )

    weight_values = weights.to_numpy(
        dtype=float
    )

    return_values = asset_returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        weight_values
    ).all():
        raise ValueError(
            "weights must contain only finite values"
        )

    if not np.isfinite(
        return_values
    ).all():
        raise ValueError(
            "asset_returns must contain only finite values"
        )

    risky_return = float(
        weight_values
        @ return_values
    )

    return float(
        risky_return
        + (
            cash_weight
            * cash_return
        )
    )


def build_equity_curve(
    returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Convert simple period returns into an equity curve.
    """
    if initial_value <= 0:
        raise ValueError(
            "initial_value must be greater than zero"
        )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    if not np.isfinite(
        returns.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "returns must contain only finite values"
        )

    equity = (
        initial_value
        * (
            1.0
            + returns
        ).cumprod()
    )

    return pd.Series(
        equity,
        index=returns.index,
        name="equity",
        dtype=float,
    )


def calculate_max_drawdown(
    equity: pd.Series,
) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Returned value is negative or zero.
    """
    if equity.empty:
        raise ValueError(
            "equity cannot be empty"
        )

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    return float(
        drawdown.min()
    )