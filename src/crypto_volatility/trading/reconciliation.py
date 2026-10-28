from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.trading.portfolio_state import (
    ASSETS,
    PortfolioState,
)

DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ReconciliationReport:
    """
    Result of paper-trading state/log reconciliation.
    """

    forecast_origin: str

    equity_from_log: float
    equity_reconstructed: float
    state_equity: float

    cash_from_log: float
    state_cash: float

    latest_portfolio_transaction_cost: float
    latest_trade_transaction_cost: float

    cumulative_portfolio_transaction_cost: float
    cumulative_trade_transaction_cost: float

    state_matches_history: bool
    equity_matches: bool
    latest_transaction_cost_matches: bool
    cumulative_transaction_cost_matches: bool

    @property
    def passed(self) -> bool:
        return (
            self.state_matches_history
            and self.equity_matches
            and self.latest_transaction_cost_matches
            and self.cumulative_transaction_cost_matches
        )


def _is_close(
    left: float,
    right: float,
    *,
    tolerance: float,
) -> bool:
    """
    Numerically compare two scalar values.
    """
    return bool(
        np.isclose(
            left,
            right,
            atol=tolerance,
            rtol=tolerance,
        )
    )


def load_portfolio_history(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load and validate portfolio-history CSV.
    """
    source = Path(
        path
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Portfolio history not found: {source}"
        )

    frame = pd.read_csv(
        source
    )

    if frame.empty:
        raise ValueError(
            "Portfolio history is empty"
        )

    required_columns = {
        "forecast_origin",
        "equity_after",
        "cash",
        "transaction_cost",
    }

    for asset in ASSETS:
        required_columns.add(
            f"{asset}_quantity"
        )

        required_columns.add(
            f"{asset}_price"
        )

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Portfolio history missing columns: "
            f"{sorted(missing)}"
        )

    return frame


def load_trade_history(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load trade-history CSV.

    Missing trade history is valid when no trades have yet been
    executed.
    """
    source = Path(
        path
    )

    if not source.exists():
        return pd.DataFrame(
            columns=[
                "forecast_origin",
                "asset",
                "side",
                "quantity",
                "price",
                "notional",
                "transaction_cost",
            ]
        )

    frame = pd.read_csv(
        source
    )

    if frame.empty:
        return frame

    required_columns = {
        "forecast_origin",
        "asset",
        "side",
        "quantity",
        "price",
        "notional",
        "transaction_cost",
    }

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Trade history missing columns: "
            f"{sorted(missing)}"
        )

    return frame


def reconstruct_equity_from_history_row(
    row: pd.Series,
) -> float:
    """
    Reconstruct portfolio equity from logged cash, quantities,
    and prices.
    """
    risky_value = 0.0

    for asset in ASSETS:
        quantity = float(
            row[
                f"{asset}_quantity"
            ]
        )

        price = float(
            row[
                f"{asset}_price"
            ]
        )

        risky_value += (
            quantity
            * price
        )

    return float(
        float(
            row[
                "cash"
            ]
        )
        + risky_value
    )


def _normalize_origins(
    values: pd.Series,
) -> pd.Series:
    """
    Normalize forecast-origin timestamps to UTC.
    """
    return pd.to_datetime(
        values,
        utc=True,
    )


def trade_cost_for_forecast_origin(
    *,
    trade_history: pd.DataFrame,
    forecast_origin: str,
) -> float:
    """
    Sum trade costs for one forecast cycle.
    """
    if trade_history.empty:
        return 0.0

    origins = _normalize_origins(
        trade_history[
            "forecast_origin"
        ]
    )

    target = pd.Timestamp(
        forecast_origin
    )

    if target.tzinfo is None:
        target = target.tz_localize(
            "UTC"
        )
    else:
        target = target.tz_convert(
            "UTC"
        )

    matching = (
        trade_history.loc[
            origins
            == target
        ]
    )

    if matching.empty:
        return 0.0

    return float(
        matching[
            "transaction_cost"
        ]
        .astype(
            float
        )
        .sum()
    )


def cumulative_portfolio_cost(
    portfolio_history: pd.DataFrame,
) -> float:
    """
    Sum transaction costs recorded across all portfolio cycles.
    """
    if portfolio_history.empty:
        return 0.0

    return float(
        portfolio_history[
            "transaction_cost"
        ]
        .astype(
            float
        )
        .sum()
    )


def cumulative_trade_cost(
    trade_history: pd.DataFrame,
) -> float:
    """
    Sum transaction costs recorded across all individual trades.
    """
    if trade_history.empty:
        return 0.0

    return float(
        trade_history[
            "transaction_cost"
        ]
        .astype(
            float
        )
        .sum()
    )


def reconcile_latest_paper_state(
    *,
    state: PortfolioState,
    portfolio_history: pd.DataFrame,
    trade_history: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ReconciliationReport:
    """
    Reconcile persisted state, latest cycle, and cumulative
    transaction-cost journals.
    """
    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative"
        )

    if portfolio_history.empty:
        raise ValueError(
            "Portfolio history is empty"
        )

    latest = (
        portfolio_history.iloc[
            -1
        ]
    )

    forecast_origin = str(
        latest[
            "forecast_origin"
        ]
    )

    equity_from_log = float(
        latest[
            "equity_after"
        ]
    )

    equity_reconstructed = (
        reconstruct_equity_from_history_row(
            latest
        )
    )

    state_equity = float(
        state.equity()
    )

    cash_from_log = float(
        latest[
            "cash"
        ]
    )

    state_cash = float(
        state.cash
    )

    latest_portfolio_transaction_cost = float(
        latest[
            "transaction_cost"
        ]
    )

    latest_trade_transaction_cost = (
        trade_cost_for_forecast_origin(
            trade_history=trade_history,
            forecast_origin=forecast_origin,
        )
    )

    cumulative_portfolio_transaction_cost = (
        cumulative_portfolio_cost(
            portfolio_history
        )
    )

    cumulative_trade_transaction_cost = (
        cumulative_trade_cost(
            trade_history
        )
    )

    equity_matches = (
        _is_close(
            equity_from_log,
            equity_reconstructed,
            tolerance=tolerance,
        )
        and _is_close(
            equity_from_log,
            state_equity,
            tolerance=tolerance,
        )
    )

    quantities_match = True
    prices_match = True

    for asset in ASSETS:
        logged_quantity = float(
            latest[
                f"{asset}_quantity"
            ]
        )

        state_quantity = float(
            state.quantities[
                asset
            ]
        )

        if not _is_close(
            logged_quantity,
            state_quantity,
            tolerance=tolerance,
        ):
            quantities_match = False

        logged_price = float(
            latest[
                f"{asset}_price"
            ]
        )

        state_price = float(
            state.last_prices[
                asset
            ]
        )

        if not _is_close(
            logged_price,
            state_price,
            tolerance=tolerance,
        ):
            prices_match = False

    cash_matches = _is_close(
        cash_from_log,
        state_cash,
        tolerance=tolerance,
    )

    state_matches_history = (
        cash_matches
        and quantities_match
        and prices_match
    )

    latest_transaction_cost_matches = (
        _is_close(
            latest_portfolio_transaction_cost,
            latest_trade_transaction_cost,
            tolerance=tolerance,
        )
    )

    cumulative_transaction_cost_matches = (
        _is_close(
            cumulative_portfolio_transaction_cost,
            cumulative_trade_transaction_cost,
            tolerance=tolerance,
        )
    )

    return ReconciliationReport(
        forecast_origin=forecast_origin,
        equity_from_log=equity_from_log,
        equity_reconstructed=equity_reconstructed,
        state_equity=state_equity,
        cash_from_log=cash_from_log,
        state_cash=state_cash,
        latest_portfolio_transaction_cost=(
            latest_portfolio_transaction_cost
        ),
        latest_trade_transaction_cost=(
            latest_trade_transaction_cost
        ),
        cumulative_portfolio_transaction_cost=(
            cumulative_portfolio_transaction_cost
        ),
        cumulative_trade_transaction_cost=(
            cumulative_trade_transaction_cost
        ),
        state_matches_history=(
            state_matches_history
        ),
        equity_matches=(
            equity_matches
        ),
        latest_transaction_cost_matches=(
            latest_transaction_cost_matches
        ),
        cumulative_transaction_cost_matches=(
            cumulative_transaction_cost_matches
        ),
    )


def assert_reconciliation_passed(
    report: ReconciliationReport,
) -> None:
    """
    Raise if paper-trading state and journals do not reconcile.
    """
    if report.passed:
        return

    failures = []

    if not report.state_matches_history:
        failures.append(
            "persisted state does not match latest portfolio history"
        )

    if not report.equity_matches:
        failures.append(
            "portfolio equity does not reconcile"
        )

    if not report.latest_transaction_cost_matches:
        failures.append(
            "latest-cycle transaction costs do not reconcile"
        )

    if not report.cumulative_transaction_cost_matches:
        failures.append(
            "cumulative transaction costs do not reconcile"
        )

    raise ValueError(
        "Paper-trading reconciliation failed: "
        + "; ".join(
            failures
        )
    )