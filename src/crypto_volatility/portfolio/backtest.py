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

    Turnover is:

        0.5 * sum(abs(w_t - w_{t-1}))
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
    Convert turnover into proportional transaction cost.
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

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
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


def calculate_drawdown_series(
    equity: pd.Series,
) -> pd.Series:
    """
    Calculate the drawdown series from an equity curve.

    Values are zero at new peaks and negative below the
    previous running maximum.
    """
    if equity.empty:
        raise ValueError(
            "equity cannot be empty"
        )

    values = equity.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "equity must contain only finite values"
        )

    if (
        values <= 0
    ).any():
        raise ValueError(
            "equity must contain only positive values"
        )

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    return pd.Series(
        drawdown,
        index=equity.index,
        name="drawdown",
        dtype=float,
    )


def calculate_max_drawdown(
    equity: pd.Series,
) -> float:
    """
    Calculate maximum drawdown.

    Returned value is negative or zero.
    """
    drawdown = (
        calculate_drawdown_series(
            equity
        )
    )

    return float(
        drawdown.min()
    )


def calculate_max_drawdown_duration(
    equity: pd.Series,
) -> int:
    """
    Calculate maximum drawdown duration in observations.

    Duration counts consecutive observations below the
    previous equity peak.
    """
    drawdown = (
        calculate_drawdown_series(
            equity
        )
    )

    max_duration = 0
    current_duration = 0

    for value in drawdown:
        if value < 0:
            current_duration += 1

            max_duration = max(
                max_duration,
                current_duration,
            )
        else:
            current_duration = 0

    return int(
        max_duration
    )


def calculate_hit_rate(
    returns: pd.Series,
) -> float:
    """
    Calculate fraction of observations with positive return.

    Zero-return observations count as non-winning periods.
    """
    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "returns must contain only finite values"
        )

    return float(
        (
            returns
            > 0
        ).mean()
    )


def calculate_sortino_ratio(
    returns: pd.Series,
    *,
    annualization_factor: int = 365,
    target_return: float = 0.0,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Downside deviation is calculated using returns below the
    target return only.

    The target return is expressed per observation.
    """
    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be positive"
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "returns must contain only finite values"
        )

    downside = np.minimum(
        values
        - target_return,
        0.0,
    )

    downside_deviation = float(
        np.sqrt(
            np.mean(
                downside**2
            )
        )
    )

    mean_excess_return = float(
        np.mean(
            values
            - target_return
        )
    )

    if downside_deviation == 0:
        return float(
            "nan"
        )

    return float(
        mean_excess_return
        / downside_deviation
        * np.sqrt(
            annualization_factor
        )
    )


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Calmar ratio.

    Calmar ratio compares annualized return with the absolute
    value of maximum drawdown.
    """
    if not np.isfinite(
        annualized_return
    ):
        raise ValueError(
            "annualized_return must be finite"
        )

    if not np.isfinite(
        max_drawdown
    ):
        raise ValueError(
            "max_drawdown must be finite"
        )

    if max_drawdown > 0:
        raise ValueError(
            "max_drawdown must be zero or negative"
        )

    if max_drawdown == 0:
        return float(
            "nan"
        )

    return float(
        annualized_return
        / abs(
            max_drawdown
        )
    )


def calculate_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate historical Value at Risk.

    Returns a positive loss magnitude.

    Example:
        VaR = 0.03 means the historical 95% one-period loss
        threshold is approximately 3%.
    """
    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be in (0, 1)"
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "returns must contain only finite values"
        )

    quantile_return = float(
        np.quantile(
            values,
            1.0
            - confidence_level,
        )
    )

    return float(
        max(
            -quantile_return,
            0.0,
        )
    )


def calculate_expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate historical Expected Shortfall.

    Expected Shortfall is the average loss among returns at or
    below the historical VaR quantile.

    Returns a positive loss magnitude.
    """
    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be in (0, 1)"
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "returns must contain only finite values"
        )

    quantile_return = float(
        np.quantile(
            values,
            1.0
            - confidence_level,
        )
    )

    tail_returns = values[
        values
        <= quantile_return
    ]

    if len(
        tail_returns
    ) == 0:
        return 0.0

    expected_tail_return = float(
        np.mean(
            tail_returns
        )
    )

    return float(
        max(
            -expected_tail_return,
            0.0,
        )
    )