from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from crypto_volatility.monitoring.performance import (
    build_monitoring_snapshot,
)
from crypto_volatility.monitoring.runtime_health import (
    check_runtime_health,
)

BASE_DIR = Path(__file__).resolve().parents[1]

LOG_DIR = BASE_DIR / "logs"

STATE_PATH = LOG_DIR / "paper_portfolio_state.json"
PORTFOLIO_HISTORY_PATH = LOG_DIR / "paper_portfolio_history.csv"
TRADE_HISTORY_PATH = LOG_DIR / "paper_trades.csv"
SCHEDULER_LOG_PATH = LOG_DIR / "paper_scheduler.log"

ASSETS = ("BTC", "ETH", "SOL", "BNB")


def money(value: float) -> str:
    return f"${value:,.2f}"


def percentage(value: float) -> str:
    return f"{value:.2%}"


def load_trades() -> pd.DataFrame:
    if not TRADE_HISTORY_PATH.exists():
        return pd.DataFrame()

    trades = pd.read_csv(TRADE_HISTORY_PATH)

    if trades.empty:
        return trades

    trades["forecast_origin"] = pd.to_datetime(
        trades["forecast_origin"],
        utc=True,
    )

    trades["run_timestamp_utc"] = pd.to_datetime(
        trades["run_timestamp_utc"],
        utc=True,
    )

    trades = trades.sort_values(
        "run_timestamp_utc",
        ascending=False,
    ).reset_index(drop=True)

    return trades


def load_portfolio_history() -> pd.DataFrame:
    history = pd.read_csv(PORTFOLIO_HISTORY_PATH)

    history["forecast_origin"] = pd.to_datetime(
        history["forecast_origin"],
        utc=True,
    )

    history["run_timestamp_utc"] = pd.to_datetime(
        history["run_timestamp_utc"],
        utc=True,
    )

    return history.sort_values(
        "forecast_origin"
    ).reset_index(drop=True)


st.set_page_config(
    page_title="Crypto Volatility ML",
    page_icon="📈",
    layout="wide",
)

st.title("Crypto Volatility ML")
st.caption(
    "Live paper-trading monitoring and portfolio observability"
)

snapshot = build_monitoring_snapshot(
    portfolio_history_path=PORTFOLIO_HISTORY_PATH,
    trade_history_path=TRADE_HISTORY_PATH,
    state_path=STATE_PATH,
)

health = check_runtime_health(
    portfolio_history_path=PORTFOLIO_HISTORY_PATH,
    trade_history_path=TRADE_HISTORY_PATH,
    state_path=STATE_PATH,
    scheduler_log_path=SCHEDULER_LOG_PATH,
)

history = load_portfolio_history()
trades = load_trades()

if health.healthy:
    st.success("Runtime healthy")
else:
    st.error("Runtime unhealthy")

if health.issues:
    with st.expander("Runtime issues"):
        for issue in health.issues:
            st.write(f"- {issue}")

st.subheader("Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Equity",
        money(snapshot.current_equity),
        money(snapshot.cumulative_pnl),
        border=True,
    )

with col2:
    st.metric(
        "Cumulative Return",
        percentage(snapshot.cumulative_return),
        border=True,
    )

with col3:
    st.metric(
        "Risky Exposure",
        percentage(snapshot.risky_exposure),
        border=True,
    )

with col4:
    st.metric(
        "Cash Weight",
        percentage(snapshot.cash_weight),
        border=True,
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "Latest Period P&L",
        money(snapshot.latest_period_pnl),
        border=True,
    )

with col6:
    st.metric(
        "Maximum Drawdown",
        percentage(snapshot.max_drawdown),
        border=True,
    )

with col7:
    st.metric(
        "Transaction Costs",
        money(snapshot.cumulative_transaction_cost),
        border=True,
    )

with col8:
    st.metric(
        "Trades",
        snapshot.number_of_trades,
        border=True,
    )

st.divider()

st.subheader("Equity Curve")

equity_curve = snapshot.equity_curve[
    [
        "forecast_origin",
        "equity_after",
    ]
].copy()

equity_curve = equity_curve.set_index(
    "forecast_origin"
)

st.line_chart(
    equity_curve,
    y="equity_after",
    y_label="Equity",
)

st.subheader("Drawdown")

drawdown_curve = snapshot.equity_curve[
    [
        "forecast_origin",
        "drawdown",
    ]
].copy()

drawdown_curve = drawdown_curve.set_index(
    "forecast_origin"
)

st.line_chart(
    drawdown_curve,
    y="drawdown",
    y_label="Drawdown",
)

st.divider()

st.subheader("Current Allocation")

allocation = pd.DataFrame(
    {
        "Asset": [
            *ASSETS,
            "Cash",
        ],
        "Weight": [
            snapshot.current_weights[asset]
            for asset in ASSETS
        ]
        + [
            snapshot.cash_weight
        ],
    }
)

st.bar_chart(
    allocation,
    x="Asset",
    y="Weight",
    y_label="Portfolio Weight",
)

st.divider()

st.subheader("Risk Metrics")

risk_col1, risk_col2, risk_col3 = st.columns(3)

if snapshot.risk_metrics_available:
    with risk_col1:
        st.metric(
            "Annualized Volatility",
            percentage(
                snapshot.annualized_volatility
            ),
            border=True,
        )

    with risk_col2:
        if snapshot.sharpe_ratio is None:
            sharpe_value = "Unavailable"
        else:
            sharpe_value = (
                f"{snapshot.sharpe_ratio:.3f}"
            )

        st.metric(
            "Sharpe Ratio",
            sharpe_value,
            border=True,
        )

else:
    with risk_col1:
        st.metric(
            "Annualized Volatility",
            "Insufficient history",
            border=True,
        )

    with risk_col2:
        st.metric(
            "Sharpe Ratio",
            "Insufficient history",
            border=True,
        )

with risk_col3:
    st.metric(
        "Risk Observations",
        (
            f"{snapshot.risk_observations}"
            f"/{snapshot.minimum_risk_observations}"
        ),
        border=True,
    )

st.divider()

st.subheader("Latest Forecasts")

latest = history.iloc[-1]

forecast_rows = []

for asset in ASSETS:
    forecast_rows.append(
        {
            "Asset": asset,
            "Forecast Log Volatility": float(
                latest[
                    f"{asset}_forecast_log_vol"
                ]
            ),
            "Current Weight": float(
                latest[
                    f"{asset}_current_weight"
                ]
            ),
            "Target Weight": float(
                latest[
                    f"{asset}_target_weight"
                ]
            ),
            "Price": float(
                latest[
                    f"{asset}_price"
                ]
            ),
        }
    )

forecast_table = pd.DataFrame(
    forecast_rows
)

st.dataframe(
    forecast_table,
    width="stretch",
    hide_index=True,
)

st.divider()

st.subheader("Execution")

exec_col1, exec_col2, exec_col3, exec_col4 = (
    st.columns(4)
)

with exec_col1:
    st.metric(
        "Cycles",
        snapshot.number_of_cycles,
        border=True,
    )

with exec_col2:
    st.metric(
        "Rebalance Signals",
        snapshot.number_of_rebalances,
        border=True,
    )

with exec_col3:
    st.metric(
        "Executed Cycles",
        snapshot.number_of_executed_cycles,
        border=True,
    )

with exec_col4:
    st.metric(
        "Latest Cost",
        money(snapshot.latest_transaction_cost),
        border=True,
    )

st.subheader("Recent Trades")

if trades.empty:
    st.info("No trades recorded.")
else:
    display_trades = trades[
        [
            "forecast_origin",
            "asset",
            "side",
            "quantity",
            "price",
            "notional",
            "transaction_cost",
        ]
    ].head(20)

    st.dataframe(
        display_trades,
        width="stretch",
        hide_index=True,
    )

st.divider()

st.subheader("Runtime Health")

runtime_col1, runtime_col2, runtime_col3 = (
    st.columns(3)
)

with runtime_col1:
    st.metric(
        "Reconciliation",
        (
            "Healthy"
            if health.reconciliation_healthy
            else "Failed"
        ),
        border=True,
    )

with runtime_col2:
    st.metric(
        "Scheduler",
        (
            "Healthy"
            if health.scheduler_success_recent
            else "Stale / Missing"
        ),
        border=True,
    )

with runtime_col3:
    st.metric(
        "Runtime",
        (
            "Healthy"
            if health.healthy
            else "Unhealthy"
        ),
        border=True,
    )

st.write(
    "**Latest forecast origin:**",
    snapshot.latest_forecast_origin,
)

st.write(
    "**Latest recorded trading run:**",
    snapshot.latest_run_timestamp_utc,
)

st.write(
    "**Last successful scheduled run:**",
    health.last_successful_scheduled_run,
)

st.caption(
    "The dashboard is read-only and does not execute trades "
    "or modify paper-trading state."
)