from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ("BTC", "ETH", "SOL", "BNB")

ANNUALIZATION_FACTOR = 365

MIN_RISK_OBSERVATIONS = 30


@dataclass(frozen=True)
class MonitoringSnapshot:
    initial_equity: float
    current_equity: float
    cumulative_pnl: float
    cumulative_return: float
    latest_period_pnl: float
    latest_period_return: float

    annualized_volatility: float | None
    sharpe_ratio: float | None
    risk_metrics_available: bool
    risk_observations: int
    minimum_risk_observations: int

    max_drawdown: float
    current_drawdown: float

    cash: float
    cash_weight: float
    risky_exposure: float
    current_weights: dict[str, float]

    cumulative_transaction_cost: float
    latest_transaction_cost: float

    number_of_cycles: int
    number_of_rebalances: int
    number_of_executed_cycles: int
    number_of_trades: int

    latest_forecast_origin: str | None
    latest_run_timestamp_utc: str | None

    equity_curve: pd.DataFrame


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio history not found: {path}"
        )

    history = pd.read_csv(path)

    if history.empty:
        raise ValueError(
            "Portfolio history is empty."
        )

    required_columns = {
        "run_timestamp_utc",
        "forecast_origin",
        "equity_before",
        "equity_after",
        "cash",
        "transaction_cost",
        "should_rebalance",
        "executed",
    }

    missing = required_columns.difference(
        history.columns
    )

    if missing:
        raise ValueError(
            "Portfolio history is missing required "
            "columns: "
            + ", ".join(sorted(missing))
        )

    history["run_timestamp_utc"] = pd.to_datetime(
        history["run_timestamp_utc"],
        utc=True,
    )

    history["forecast_origin"] = pd.to_datetime(
        history["forecast_origin"],
        utc=True,
    )

    history = history.sort_values(
        [
            "forecast_origin",
            "run_timestamp_utc",
        ]
    ).reset_index(drop=True)

    return history


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    trades = pd.read_csv(path)

    if trades.empty:
        return trades

    if "forecast_origin" in trades.columns:
        trades["forecast_origin"] = pd.to_datetime(
            trades["forecast_origin"],
            utc=True,
        )

    if "run_timestamp_utc" in trades.columns:
        trades["run_timestamp_utc"] = pd.to_datetime(
            trades["run_timestamp_utc"],
            utc=True,
        )

    return trades


def _load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio state not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _build_equity_curve(
    history: pd.DataFrame,
) -> pd.DataFrame:
    curve = history[
        [
            "forecast_origin",
            "run_timestamp_utc",
            "equity_before",
            "equity_after",
            "transaction_cost",
        ]
    ].copy()

    initial_equity = float(
        curve.loc[0, "equity_before"]
    )

    previous_equity = (
        curve["equity_after"].shift(1)
    )

    previous_equity.iloc[0] = initial_equity

    curve["period_pnl"] = (
        curve["equity_after"]
        - previous_equity
    )

    curve["period_return"] = (
        curve["equity_after"]
        / previous_equity
        - 1.0
    )

    curve["running_peak"] = (
        curve["equity_after"].cummax()
    )

    curve["running_peak"] = np.maximum(
        curve["running_peak"],
        initial_equity,
    )

    curve["drawdown"] = (
        curve["equity_after"]
        / curve["running_peak"]
        - 1.0
    )

    curve["cumulative_pnl"] = (
        curve["equity_after"]
        - initial_equity
    )

    curve["cumulative_return"] = (
        curve["equity_after"]
        / initial_equity
        - 1.0
    )

    return curve


def _annualized_volatility(
    returns: pd.Series,
) -> float:
    clean_returns = returns.dropna()

    return float(
        clean_returns.std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
    )


def _sharpe_ratio(
    returns: pd.Series,
) -> float | None:
    clean_returns = returns.dropna()

    volatility = clean_returns.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return None

    return float(
        clean_returns.mean()
        / volatility
        * np.sqrt(ANNUALIZATION_FACTOR)
    )


def _state_equity(state: dict) -> float:
    cash = float(state["cash"])

    quantities = state["quantities"]
    prices = state["last_prices"]

    risky_value = sum(
        float(quantities[asset])
        * float(prices[asset])
        for asset in ASSETS
    )

    return cash + risky_value


def _current_weights(
    state: dict,
    equity: float,
) -> dict[str, float]:
    if equity <= 0:
        return {
            asset: 0.0
            for asset in ASSETS
        }

    quantities = state["quantities"]
    prices = state["last_prices"]

    return {
        asset: (
            float(quantities[asset])
            * float(prices[asset])
            / equity
        )
        for asset in ASSETS
    }


def build_monitoring_snapshot(
    *,
    portfolio_history_path: str | Path,
    trade_history_path: str | Path,
    state_path: str | Path,
) -> MonitoringSnapshot:
    portfolio_history_path = Path(
        portfolio_history_path
    )

    trade_history_path = Path(
        trade_history_path
    )

    state_path = Path(state_path)

    history = _load_history(
        portfolio_history_path
    )

    trades = _load_trades(
        trade_history_path
    )

    state = _load_state(
        state_path
    )

    curve = _build_equity_curve(
        history
    )

    initial_equity = float(
        history.iloc[0]["equity_before"]
    )

    current_equity = _state_equity(
        state
    )

    cumulative_pnl = (
        current_equity - initial_equity
    )

    cumulative_return = (
        current_equity
        / initial_equity
        - 1.0
    )

    latest_period_pnl = float(
        curve.iloc[-1]["period_pnl"]
    )

    latest_period_return = float(
        curve.iloc[-1]["period_return"]
    )

    returns = (
        curve["period_return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    risk_observations = len(returns)

    risk_metrics_available = (
        risk_observations
        >= MIN_RISK_OBSERVATIONS
    )

    if risk_metrics_available:
        annualized_volatility = (
            _annualized_volatility(
                returns
            )
        )

        sharpe_ratio = _sharpe_ratio(
            returns
        )

    else:
        annualized_volatility = None
        sharpe_ratio = None

    max_drawdown = float(
        curve["drawdown"].min()
    )

    historical_peak = max(
        initial_equity,
        float(
            curve["running_peak"].iloc[-1]
        ),
    )

    current_peak = max(
        historical_peak,
        current_equity,
    )

    if current_peak <= 0:
        current_drawdown = 0.0
    else:
        current_drawdown = float(
            current_equity
            / current_peak
            - 1.0
        )

    cash = float(
        state["cash"]
    )

    cash_weight = (
        cash / current_equity
        if current_equity > 0
        else 0.0
    )

    current_weights = _current_weights(
        state,
        current_equity,
    )

    risky_exposure = float(
        sum(
            current_weights.values()
        )
    )

    cumulative_transaction_cost = float(
        history[
            "transaction_cost"
        ].sum()
    )

    latest_transaction_cost = float(
        history.iloc[-1][
            "transaction_cost"
        ]
    )

    number_of_cycles = len(
        history
    )

    number_of_rebalances = int(
        history[
            "should_rebalance"
        ].astype(bool).sum()
    )

    number_of_executed_cycles = int(
        history[
            "executed"
        ].astype(bool).sum()
    )

    number_of_trades = len(
        trades
    )

    latest_forecast_origin = (
        state.get(
            "last_forecast_timestamp"
        )
    )

    latest_run_timestamp_utc = (
        history.iloc[-1][
            "run_timestamp_utc"
        ].isoformat()
    )

    return MonitoringSnapshot(
        initial_equity=initial_equity,
        current_equity=current_equity,
        cumulative_pnl=cumulative_pnl,
        cumulative_return=cumulative_return,
        latest_period_pnl=latest_period_pnl,
        latest_period_return=latest_period_return,
        annualized_volatility=(
            annualized_volatility
        ),
        sharpe_ratio=sharpe_ratio,
        risk_metrics_available=(
            risk_metrics_available
        ),
        risk_observations=(
            risk_observations
        ),
        minimum_risk_observations=(
            MIN_RISK_OBSERVATIONS
        ),
        max_drawdown=max_drawdown,
        current_drawdown=current_drawdown,
        cash=cash,
        cash_weight=cash_weight,
        risky_exposure=risky_exposure,
        current_weights=current_weights,
        cumulative_transaction_cost=(
            cumulative_transaction_cost
        ),
        latest_transaction_cost=(
            latest_transaction_cost
        ),
        number_of_cycles=(
            number_of_cycles
        ),
        number_of_rebalances=(
            number_of_rebalances
        ),
        number_of_executed_cycles=(
            number_of_executed_cycles
        ),
        number_of_trades=(
            number_of_trades
        ),
        latest_forecast_origin=(
            latest_forecast_origin
        ),
        latest_run_timestamp_utc=(
            latest_run_timestamp_utc
        ),
        equity_curve=curve,
    )