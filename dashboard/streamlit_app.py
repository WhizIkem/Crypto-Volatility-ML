from __future__ import annotations

from pathlib import Path

import altair as alt
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

COLORS = {
    "bg": "#080D16",
    "surface": "#0F1724",
    "surface_high": "#151F2E",
    "border": "#263449",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "blue": "#38BDF8",
    "blue_soft": "#60A5FA",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "red": "#FB7185",
    "amber": "#FBBF24",
    "purple": "#A78BFA",
    "slate": "#64748B",
}



def format_timestamp(value: str | None) -> str:
    if not value:
        return "Unavailable"

    timestamp = pd.Timestamp(value)

    return timestamp.strftime(
        "%Y-%m-%d %H:%M UTC"
    )

def money(value: float, decimals: int = 2) -> str:
    if value < 0:
        return f"-${abs(value):,.{decimals}f}"

    return f"${value:,.{decimals}f}"


def compact_money(value: float) -> str:
    absolute = abs(value)

    if absolute >= 1_000_000:
        text = f"${absolute / 1_000_000:.2f}M"
    elif absolute >= 1_000:
        text = f"${absolute / 1_000:.1f}K"
    else:
        text = f"${absolute:,.0f}"

    if value < 0:
        return f"-{text}"

    return text


def percentage(value: float) -> str:
    return f"{value:.2%}"


def signed_percentage(value: float) -> str:
    return f"{value:+.2%}"


def metric_card(
    title: str,
    value: str,
    *,
    subtitle: str | None = None,
    tone: str = "neutral",
) -> None:
    tone_class = {
        "positive": "metric-positive",
        "negative": "metric-negative",
        "neutral": "metric-neutral",
        "info": "metric-info",
    }.get(tone, "metric-neutral")

    subtitle_html = ""

    if subtitle:
        subtitle_html = (
            f'<div class="metric-subtitle {tone_class}">'
            f"{subtitle}"
            "</div>"
        )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    return trades.sort_values(
        "run_timestamp_utc",
        ascending=False,
    ).reset_index(drop=True)


def load_portfolio_history() -> pd.DataFrame:
    history = pd.read_csv(
        PORTFOLIO_HISTORY_PATH
    )

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
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --dashboard-bg: #080D16;
            --dashboard-surface: #0F1724;
            --dashboard-surface-high: #151F2E;
            --dashboard-border: #263449;

            --dashboard-text: #F8FAFC;
            --dashboard-muted: #94A3B8;

            --dashboard-blue: #38BDF8;
            --dashboard-cyan: #22D3EE;
            --dashboard-green: #34D399;
            --dashboard-red: #FB7185;
            --dashboard-amber: #FBBF24;
            --dashboard-purple: #A78BFA;
        }


        /* --------------------------------------------------
          MAIN APPLICATION
        -------------------------------------------------- */

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 85% 0%,
                    rgba(56, 189, 248, 0.07),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 15% 20%,
                    rgba(167, 139, 250, 0.035),
                    transparent 24%
                ),
                #080D16;
        }


        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }


        /* --------------------------------------------------
          TYPOGRAPHY
        -------------------------------------------------- */

        h1 {
            color: #F8FAFC !important;
            font-size: 2.55rem !important;
            font-weight: 760 !important;
            letter-spacing: -0.055rem;
            margin-bottom: 0.2rem !important;
        }

        h2 {
            color: #F8FAFC !important;
            font-size: 1.65rem !important;
            font-weight: 720 !important;
            margin-top: 1rem !important;
        }

        h3 {
            color: #E2E8F0 !important;
            font-size: 1.2rem !important;
            font-weight: 680 !important;
        }


        .hero-subtitle {
            color: #94A3B8;
            font-size: 1rem;
            margin-top: -0.2rem;
            margin-bottom: 1.3rem;
        }


        .section-note {
            color: #7F91A9;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: -0.4rem;
            margin-bottom: 0.8rem;
        }


        /* --------------------------------------------------
          STATUS BANNERS
        -------------------------------------------------- */

        .status-banner {
            border: 1px solid rgba(52, 211, 153, 0.32);
            border-left: 4px solid #34D399;

            background:
                linear-gradient(
                    90deg,
                    rgba(52, 211, 153, 0.12),
                    rgba(52, 211, 153, 0.035)
                );

            border-radius: 12px;

            padding: 0.85rem 1rem;
            margin: 0.7rem 0 1.4rem 0;

            color: #6EE7B7;
            font-weight: 650;
        }


        .status-banner-warning {
            border: 1px solid rgba(251, 113, 133, 0.30);
            border-left: 4px solid #FB7185;

            background:
                linear-gradient(
                    90deg,
                    rgba(251, 113, 133, 0.12),
                    rgba(251, 113, 133, 0.035)
                );

            border-radius: 12px;

            padding: 0.85rem 1rem;
            margin: 0.7rem 0 1.4rem 0;

            color: #FDA4AF;
            font-weight: 650;
        }


        /* --------------------------------------------------
          KPI CARDS
        -------------------------------------------------- */

        .metric-card {
            min-height: 138px;

            padding: 1rem 1.05rem;

            border-radius: 14px;

            border:
                1px solid
                rgba(71, 85, 105, 0.55);

            background:
                linear-gradient(
                    145deg,
                    rgba(21, 31, 46, 0.98),
                    rgba(12, 19, 31, 0.98)
                );

            box-shadow:
                0 10px 30px rgba(0, 0, 0, 0.20);

            overflow: hidden;

            transition:
                transform 0.18s ease,
                border-color 0.18s ease,
                box-shadow 0.18s ease;
        }


        .metric-card:hover {
            transform: translateY(-2px);

            border-color:
                rgba(56, 189, 248, 0.45);

            box-shadow:
                0 14px 34px rgba(0, 0, 0, 0.28),
                0 0 0 1px rgba(56, 189, 248, 0.06);
        }


        .metric-title {
            color: #8FA3BC;

            font-size:
                clamp(
                    0.68rem,
                    0.8vw,
                    0.82rem
                );

            font-weight: 700;

            text-transform: uppercase;

            letter-spacing: 0.055rem;

            line-height: 1.4;

            margin-bottom: 0.45rem;

            overflow-wrap: anywhere;
        }


        .metric-value {
            color: #F8FAFC;

            font-size:
                clamp(
                    1.25rem,
                    1.8vw,
                    1.75rem
                );

            line-height: 1.15;

            font-weight: 750;

            letter-spacing: -0.025rem;

            white-space: normal;
            overflow-wrap: anywhere;
        }


        .metric-subtitle {
            margin-top: 0.55rem;

            font-size:
                clamp(
                    0.7rem,
                    0.75vw,
                    0.84rem
                );

            line-height: 1.45;

            font-weight: 650;

            white-space: normal;
            overflow-wrap: anywhere;
        }


        .metric-positive {
            color: #34D399;
        }


        .metric-negative {
            color: #FB7185;
        }


        .metric-neutral {
            color: #94A3B8;
        }


        .metric-info {
            color: #38BDF8;
        }


        /* --------------------------------------------------
          SIDEBAR
        -------------------------------------------------- */

        [data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #111827 0%,
                    #0D1420 100%
                );

            border-right:
                1px solid
                rgba(71, 85, 105, 0.42);
        }


        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }


        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background:
                rgba(15, 23, 36, 0.42);

            border-radius: 10px;

            padding: 0.4rem 0.65rem;
        }


        /* --------------------------------------------------
          TABS
        -------------------------------------------------- */

        div[data-testid="stTabs"]
        button {
            color: #94A3B8;

            font-size: 0.92rem;

            font-weight: 650;

            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }


        div[data-testid="stTabs"]
        button:hover {
            color: #E2E8F0;
        }


        div[data-testid="stTabs"]
        button[aria-selected="true"] {
            color: #38BDF8 !important;
        }


        div[data-testid="stTabs"]
        [data-baseweb="tab-highlight"] {
            background-color: #38BDF8;
        }


        /* --------------------------------------------------
          TABLES
        -------------------------------------------------- */

        div[data-testid="stDataFrame"] {
            border:
                1px solid
                rgba(71, 85, 105, 0.45);

            border-radius: 12px;

            overflow: hidden;

            box-shadow:
                0 8px 22px rgba(0, 0, 0, 0.12);
        }


        /* --------------------------------------------------
          PROGRESS
        -------------------------------------------------- */

        div[data-testid="stProgress"] > div > div {
            background:
                linear-gradient(
                    90deg,
                    #38BDF8,
                    #22D3EE
                );
        }


        /* --------------------------------------------------
          DIVIDERS
        -------------------------------------------------- */

        hr {
            border-color:
                rgba(71, 85, 105, 0.38);
        }


        /* --------------------------------------------------
          LINKS
        -------------------------------------------------- */

        a {
            color: #38BDF8 !important;
        }


        /* --------------------------------------------------
          SCROLLBAR
        -------------------------------------------------- */

        ::-webkit-scrollbar {
            width: 9px;
            height: 9px;
        }

        ::-webkit-scrollbar-track {
            background: #080D16;
        }

        ::-webkit-scrollbar-thumb {
            background: #263449;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #33465F;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


snapshot = build_monitoring_snapshot(
    portfolio_history_path=(
        PORTFOLIO_HISTORY_PATH
    ),
    trade_history_path=(
        TRADE_HISTORY_PATH
    ),
    state_path=STATE_PATH,
)

health = check_runtime_health(
    portfolio_history_path=(
        PORTFOLIO_HISTORY_PATH
    ),
    trade_history_path=(
        TRADE_HISTORY_PATH
    ),
    state_path=STATE_PATH,
    scheduler_log_path=(
        SCHEDULER_LOG_PATH
    ),
)

history = load_portfolio_history()
trades = load_trades()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("Crypto Volatility ML")

st.markdown(
    """
    <div class="hero-subtitle">
        ML-driven volatility forecasting · Dynamic risk allocation ·
        Automated paper trading
    </div>
    """,
    unsafe_allow_html=True,
)

if health.healthy:
    st.markdown(
        """
        <div class="status-banner">
            ● Production runtime healthy ·
            Scheduler active ·
            Portfolio reconciliation passing
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="status-banner-warning">
            ● Production runtime requires attention
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.header("System")

    if health.healthy:
        st.success("Production healthy")
    else:
        st.error("Attention required")

    st.metric(
        "Current Equity",
        compact_money(
            snapshot.current_equity
        ),
    )

    st.metric(
        "Risky Exposure",
        percentage(
            snapshot.risky_exposure
        ),
    )

    st.metric(
        "Cash",
        compact_money(
            snapshot.cash
        ),
    )

    st.divider()

    st.subheader("Latest Activity")

    st.caption("Forecast origin")
    st.code(
        snapshot.latest_forecast_origin
        or "Unavailable",
        language=None,
    )

    st.caption("Last trading run")
    st.write(
        snapshot.latest_run_timestamp_utc
    )

    st.caption("Scheduler success")
    st.write(
        health.last_successful_scheduled_run
    )

    st.divider()

    st.caption(
        "Read-only production dashboard. "
        "Trading cannot be triggered from this interface."
    )


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

(
    overview_tab,
    portfolio_tab,
    execution_tab,
    health_tab,
) = st.tabs(
    [
        "Overview",
        "Portfolio & Forecasts",
        "Execution",
        "System Health",
    ]
)


# ------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------

with overview_tab:
    st.subheader("Portfolio Snapshot")

    st.markdown(
        """
        <div class="section-note">
            Current paper portfolio state and risk allocation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pnl_tone = (
        "positive"
        if snapshot.cumulative_pnl >= 0
        else "negative"
    )

    period_pnl_tone = (
        "positive"
        if snapshot.latest_period_pnl >= 0
        else "negative"
    )

    return_tone = (
        "positive"
        if snapshot.cumulative_return >= 0
        else "negative"
    )

    row1 = st.columns(
        4,
        gap="medium")

    with row1[0]:
        pnl_arrow = (
            "▲"
            if snapshot.cumulative_pnl >= 0
            else "▼"
        )

        metric_card(
            "Current Equity",
            money(snapshot.current_equity),
            subtitle=(
                f"{pnl_arrow} "
                f"{money(snapshot.cumulative_pnl)} "
                "since inception"
            ),
            tone=pnl_tone,
        )

    with row1[1]:
        metric_card(
            "Cumulative Return",
            percentage(
                snapshot.cumulative_return
            ),
            subtitle="Since paper-trading inception",
            tone=return_tone,
        )

    with row1[2]:
        metric_card(
            "Risky Exposure",
            percentage(
                snapshot.risky_exposure
            ),
            subtitle=(
                f"Cash reserve "
                f"{percentage(snapshot.cash_weight)}"
            ),
            tone="info",
        )

    with row1[3]:
        metric_card(
            "Cash Balance",
            money(snapshot.cash),
            subtitle=(
                percentage(snapshot.cash_weight)
                + " of portfolio"
            ),
            tone="neutral",
        )

    st.write("")

    row2 = st.columns(
        4,
        gap="medium",
    )

    with row2[0]:
        period_arrow = (
            "▲"
            if snapshot.latest_period_pnl >= 0
            else "▼"
        )

        metric_card(
            "Latest Period P&L",
            money(
                snapshot.latest_period_pnl
            ),
            subtitle=(
                f"{period_arrow} "
                f"{signed_percentage(snapshot.latest_period_return)}"
            ),
            tone=period_pnl_tone,
        )

    with row2[1]:
        metric_card(
            "Maximum Drawdown",
            percentage(
                snapshot.max_drawdown
            ),
            subtitle=(
                "Current "
                + percentage(
                    snapshot.current_drawdown
                )
            ),
            tone="negative",
        )

    with row2[2]:
        metric_card(
            "Transaction Costs",
            money(
                snapshot.cumulative_transaction_cost
            ),
            subtitle=(
                "Latest "
                + money(
                    snapshot.latest_transaction_cost
                )
            ),
            tone="neutral",
        )

    with row2[3]:
        metric_card(
            "Executed Trades",
            f"{snapshot.number_of_trades}",
            subtitle=(
                f"{snapshot.number_of_executed_cycles} "
                "executed cycles"
            ),
            tone="neutral",
        )

    st.divider()

    # --------------------------------------------------------------
    # Performance + Allocation
    # --------------------------------------------------------------

    chart_col, allocation_col = st.columns(
        [2.1, 1]
    )

    with chart_col:
        st.subheader("Equity Curve")

        st.markdown(
            """
            <div class="section-note">
                Mark-to-market paper portfolio equity.
            </div>
            """,
            unsafe_allow_html=True,
        )

        equity_chart_data = (
            snapshot.equity_curve[
                [
                    "forecast_origin",
                    "equity_after",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "forecast_origin": "Date",
                    "equity_after": "Equity",
                }
            )
        )

        equity_chart = (
            alt.Chart(equity_chart_data)
            .mark_line(
                point=True,
                strokeWidth=3,
                color=COLORS["cyan"],
            )
            .encode(
                x=alt.X(
                    "Date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%b %d",
                        labelAngle=0,
                        grid=False,
                    ),
                ),
                y=alt.Y(
                    "Equity:Q",
                    title="Portfolio Equity",
                    scale=alt.Scale(
                        zero=False
                    ),
                    axis=alt.Axis(
                        format="$,.0f",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Date:T",
                        title="Date",
                        format="%Y-%m-%d",
                    ),
                    alt.Tooltip(
                        "Equity:Q",
                        title="Equity",
                        format="$,.2f",
                    ),
                ],
            )
            .properties(
                height=350,
            )
            .interactive()
        )

        st.altair_chart(
            equity_chart,
            width="stretch",
        )

    with allocation_col:
        st.subheader("Current Allocation")

        st.markdown(
            """
            <div class="section-note">
                Portfolio exposure after latest valuation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        allocation = pd.DataFrame(
            {
                "Asset": [
                    *ASSETS,
                    "Cash",
                ],
                "Weight": [
                    snapshot.current_weights[
                        asset
                    ]
                    for asset in ASSETS
                ]
                + [
                    snapshot.cash_weight
                ],
            }
        )

        allocation = allocation.sort_values(
            "Weight",
            ascending=True,
        )

        allocation_chart = (
            alt.Chart(allocation)
            .mark_bar(
                cornerRadiusEnd=6,
            )
            .encode(
                y=alt.Y(
                    "Asset:N",
                    sort=None,
                    title=None,
                ),
                x=alt.X(
                    "Weight:Q",
                    title="Portfolio Weight",
                    axis=alt.Axis(
                        format=".0%",
                    ),
                    scale=alt.Scale(
                        domain=[
                            0,
                            max(
                                0.65,
                                float(
                                    allocation[
                                        "Weight"
                                    ].max()
                                )
                                * 1.10,
                            ),
                        ]
                    ),
                ),
                color=alt.Color(
                    "Asset:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=[
                            "BTC",
                            "ETH",
                            "SOL",
                            "BNB",
                            "Cash",
                        ],
                        range=[
                            COLORS["blue"],
                            COLORS["blue"],
                            COLORS["blue"],
                            COLORS["blue"],
                            COLORS["green"],
                        ],
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Asset:N",
                        title="Asset",
                    ),
                    alt.Tooltip(
                        "Weight:Q",
                        title="Weight",
                        format=".2%",
                    ),
                ],
            )
        )

        allocation_labels = (
            alt.Chart(allocation)
            .mark_text(
                align="left",
                baseline="middle",
                dx=6,
                fontWeight="bold",
            )
            .encode(
                y=alt.Y(
                    "Asset:N",
                    sort=None,
                ),
                x="Weight:Q",
                text=alt.Text(
                    "Weight:Q",
                    format=".1%",
                ),
            )
        )

        st.altair_chart(
            allocation_chart
            + allocation_labels,
            width="stretch",
        )

    # --------------------------------------------------------------
    # Drawdown + risk
    # --------------------------------------------------------------

    drawdown_col, risk_col = st.columns(
        [2.1, 1]
    )

    with drawdown_col:
        st.subheader("Drawdown")

        st.markdown(
            """
            <div class="section-note">
                Decline from the portfolio's running equity peak.
            </div>
            """,
            unsafe_allow_html=True,
        )

        drawdown_data = (
            snapshot.equity_curve[
                [
                    "forecast_origin",
                    "drawdown",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "forecast_origin": "Date",
                    "drawdown": "Drawdown",
                }
            )
        )

        drawdown_chart = (
            alt.Chart(drawdown_data)
            .mark_area(
                opacity=0.28,
                color=COLORS["red"],
            )
            .encode(
                x=alt.X(
                    "Date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%b %d",
                        labelAngle=0,
                        grid=False,
                    ),
                ),
                y=alt.Y(
                    "Drawdown:Q",
                    title="Drawdown",
                    axis=alt.Axis(
                        format=".1%",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Date:T",
                        title="Date",
                        format="%Y-%m-%d",
                    ),
                    alt.Tooltip(
                        "Drawdown:Q",
                        title="Drawdown",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=300,
            )
        )

        drawdown_line = (
            alt.Chart(drawdown_data)
            .mark_line(
                strokeWidth=2.5,
                color=COLORS["red"],
            )
            .encode(
                x="Date:T",
                y="Drawdown:Q",
            )
        )

        st.altair_chart(
            drawdown_chart
            + drawdown_line,
            width="stretch",
        )

    with risk_col:
        st.subheader("Risk Readiness")

        if snapshot.risk_metrics_available:
            metric_card(
                "Annualized Volatility",
                percentage(
                    snapshot.annualized_volatility
                ),
                tone="neutral",
            )

            st.write("")

            sharpe_text = (
                "Unavailable"
                if snapshot.sharpe_ratio is None
                else f"{snapshot.sharpe_ratio:.2f}"
            )

            metric_card(
                "Sharpe Ratio",
                sharpe_text,
                tone="neutral",
            )

        else:
            metric_card(
                "Risk History",
                (
                    f"{snapshot.risk_observations}"
                    f" / "
                    f"{snapshot.minimum_risk_observations}"
                ),
                subtitle=(
                    "Daily observations collected"
                ),
                tone="info",
            )

            st.write("")

            progress = (
                snapshot.risk_observations
                / snapshot.minimum_risk_observations
            )

            st.progress(
                min(progress, 1.0)
            )

            st.caption(
                "Annualized volatility and Sharpe "
                "will activate after 30 observations."
            )


# ------------------------------------------------------------------
# PORTFOLIO & FORECASTS
# ------------------------------------------------------------------

with portfolio_tab:
    st.subheader("Portfolio & Forecasts")

    st.markdown(
        """
        <div class="section-note">
            Latest next-day volatility forecasts and the resulting
            portfolio allocation decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = history.iloc[-1]

    forecast_rows = []

    for asset in ASSETS:
        forecast_log_vol = float(
            latest[
                f"{asset}_forecast_log_vol"
            ]
        )

        current_weight = float(
            latest[
                f"{asset}_current_weight"
            ]
        )

        target_weight = float(
            latest[
                f"{asset}_target_weight"
            ]
        )

        weight_change = (
            target_weight
            - current_weight
        )

        forecast_rows.append(
            {
                "Asset": asset,
                "Forecast Log Volatility": (
                    forecast_log_vol
                ),
                "Current Weight": (
                    current_weight
                ),
                "Target Weight": (
                    target_weight
                ),
                "Weight Change": (
                    weight_change
                ),
                "Price": float(
                    latest[
                        f"{asset}_price"
                    ]
                ),
            }
        )

    forecast_data = pd.DataFrame(
        forecast_rows
    )

    # --------------------------------------------------------------
    # Decision summary
    # --------------------------------------------------------------

    highest_volatility_row = (
        forecast_data.loc[
            forecast_data[
                "Forecast Log Volatility"
            ].idxmax()
        ]
    )

    largest_reduction_row = (
        forecast_data.loc[
            forecast_data[
                "Weight Change"
            ].idxmin()
        ]
    )

    largest_target_row = (
        forecast_data.loc[
            forecast_data[
                "Target Weight"
            ].idxmax()
        ]
    )

    summary_columns = st.columns(4)

    with summary_columns[0]:
        metric_card(
            "Forecast Origin",
            latest[
                "forecast_origin"
            ].strftime("%Y-%m-%d"),
            subtitle="Latest closed daily candle",
            tone="neutral",
        )

    with summary_columns[1]:
        metric_card(
            "Highest Forecast Vol",
            str(
                highest_volatility_row[
                    "Asset"
                ]
            ),
            subtitle=(
                "Log vol "
                f"{highest_volatility_row['Forecast Log Volatility']:.4f}"
            ),
            tone="info",
        )

    with summary_columns[2]:
        metric_card(
            "Largest Risk Reduction",
            str(
                largest_reduction_row[
                    "Asset"
                ]
            ),
            subtitle=(
                signed_percentage(
                    float(
                        largest_reduction_row[
                            "Weight Change"
                        ]
                    )
                )
                + " target change"
            ),
            tone="negative",
        )

    with summary_columns[3]:
        metric_card(
            "Largest Target Weight",
            str(
                largest_target_row[
                    "Asset"
                ]
            ),
            subtitle=percentage(
                float(
                    largest_target_row[
                        "Target Weight"
                    ]
                )
            ),
            tone="info",
        )

    st.divider()

    # --------------------------------------------------------------
    # Current vs target allocation
    # --------------------------------------------------------------

    allocation_col, exposure_col = (
        st.columns(
            [2.2, 1]
        )
    )

    with allocation_col:
        st.subheader(
            "Current vs Target Allocation"
        )

        st.markdown(
            """
            <div class="section-note">
                How the volatility forecasts translate into
                portfolio positioning.
            </div>
            """,
            unsafe_allow_html=True,
        )

        allocation_comparison = (
            forecast_data[
                [
                    "Asset",
                    "Current Weight",
                    "Target Weight",
                ]
            ]
            .melt(
                id_vars="Asset",
                var_name="Allocation",
                value_name="Weight",
            )
        )

        allocation_comparison[
            "Allocation"
        ] = (
            allocation_comparison[
                "Allocation"
            ]
            .str.replace(
                " Weight",
                "",
                regex=False,
            )
        )

        comparison_chart = (
            alt.Chart(
                allocation_comparison
            )
            .mark_bar(
                cornerRadiusEnd=4,
            )
            .encode(
                y=alt.Y(
                    "Asset:N",
                    title=None,
                    sort=list(ASSETS),
                ),
                x=alt.X(
                    "Weight:Q",
                    title="Portfolio Weight",
                    axis=alt.Axis(
                        format=".0%",
                    ),
                ),
                yOffset=alt.YOffset(
                    "Allocation:N"
                ),
                color=alt.Color(
                    "Allocation:N",
                    title=None,
                    scale=alt.Scale(
                        domain=[
                            "Current",
                            "Target",
                        ],
                        range=[
                            COLORS["slate"],
                            COLORS["cyan"],
                        ],
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Asset:N",
                        title="Asset",
                    ),
                    alt.Tooltip(
                        "Allocation:N",
                        title="Allocation",
                    ),
                    alt.Tooltip(
                        "Weight:Q",
                        title="Weight",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=330,
            )
        )

        st.altair_chart(
            comparison_chart,
            width="stretch",
        )

    with exposure_col:
        st.subheader(
            "Portfolio Exposure"
        )

        st.markdown(
            """
            <div class="section-note">
                Current risky and defensive positioning.
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_card(
            "Risky Exposure",
            percentage(
                snapshot.risky_exposure
            ),
            subtitle="Crypto allocation",
            tone="info",
        )

        st.write("")

        metric_card(
            "Cash Reserve",
            percentage(
                snapshot.cash_weight
            ),
            subtitle=money(
                snapshot.cash
            ),
            tone="neutral",
        )

        st.write("")

        rebalance_text = (
            "Yes"
            if bool(
                latest[
                    "should_rebalance"
                ]
            )
            else "No"
        )

        metric_card(
            "Rebalance Signal",
            rebalance_text,
            subtitle=(
                "Turnover "
                + percentage(
                    float(
                        latest[
                            "turnover"
                        ]
                    )
                )
            ),
            tone=(
                "info"
                if rebalance_text == "Yes"
                else "neutral"
            ),
        )

    st.divider()

    # --------------------------------------------------------------
    # Forecast decision table
    # --------------------------------------------------------------

    st.subheader(
        "Forecast & Allocation Decision"
    )

    st.markdown(
        """
        <div class="section-note">
            Positive weight change means the strategy wants more
            exposure; negative change means risk is being reduced.
        </div>
        """,
        unsafe_allow_html=True,
    )

    decision_rows = []

    for _, row in forecast_data.iterrows():
        weight_change = float(
            row[
                "Weight Change"
            ]
        )

        if weight_change > 0:
            direction = "Increase"
        elif weight_change < 0:
            direction = "Reduce"
        else:
            direction = "Hold"

        decision_rows.append(
            {
                "Asset": row["Asset"],
                "Forecast Log Vol": (
                    f"{float(row['Forecast Log Volatility']):.4f}"
                ),
                "Current": percentage(
                    float(
                        row[
                            "Current Weight"
                        ]
                    )
                ),
                "Target": percentage(
                    float(
                        row[
                            "Target Weight"
                        ]
                    )
                ),
                "Change": signed_percentage(
                    weight_change
                ),
                "Decision": direction,
                "Price": money(
                    float(
                        row[
                            "Price"
                        ]
                    )
                ),
            }
        )

    decision_table = pd.DataFrame(
        decision_rows
    )

    st.dataframe(
        decision_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Asset": st.column_config.TextColumn(
                "Asset",
                width="small",
            ),
            "Forecast Log Vol": (
                st.column_config.TextColumn(
                    "Forecast Log Vol",
                    width="medium",
                )
            ),
            "Current": (
                st.column_config.TextColumn(
                    "Current Weight",
                    width="medium",
                )
            ),
            "Target": (
                st.column_config.TextColumn(
                    "Target Weight",
                    width="medium",
                )
            ),
            "Change": (
                st.column_config.TextColumn(
                    "Weight Change",
                    width="medium",
                )
            ),
            "Decision": (
                st.column_config.TextColumn(
                    "Decision",
                    width="small",
                )
            ),
            "Price": (
                st.column_config.TextColumn(
                    "Price",
                    width="medium",
                )
            ),
        },
    )

    st.caption(
        "Portfolio weights are generated from forecast risk, "
        "correlation structure, portfolio-volatility targeting, "
        "asset caps, and the turnover threshold."
    )


# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

with execution_tab:
    st.subheader("Execution & Trading Activity")

    st.markdown(
        """
        <div class="section-note">
            Trading activity generated by portfolio rebalancing,
            including turnover and transaction costs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_history_row = history.iloc[-1]

    latest_turnover = float(
        latest_history_row["turnover"]
    )

    latest_cost = float(
        latest_history_row["transaction_cost"]
    )

    executed_rate = (
        snapshot.number_of_executed_cycles
        / snapshot.number_of_cycles
        if snapshot.number_of_cycles > 0
        else 0.0
    )

    avg_cost_per_trade = (
        snapshot.cumulative_transaction_cost
        / snapshot.number_of_trades
        if snapshot.number_of_trades > 0
        else 0.0
    )

    summary_columns = st.columns(4)

    with summary_columns[0]:
        metric_card(
            "Trading Cycles",
            str(snapshot.number_of_cycles),
            subtitle=(
                f"{snapshot.number_of_executed_cycles} executed"
            ),
            tone="neutral",
        )

    with summary_columns[1]:
        metric_card(
            "Execution Rate",
            percentage(executed_rate),
            subtitle=(
                f"{snapshot.number_of_rebalances} rebalance signals"
            ),
            tone="info",
        )

    with summary_columns[2]:
        metric_card(
            "Latest Turnover",
            percentage(latest_turnover),
            subtitle="One-way portfolio turnover",
            tone=(
                "info"
                if latest_turnover > 0
                else "neutral"
            ),
        )

    with summary_columns[3]:
        metric_card(
            "Cumulative Costs",
            money(
                snapshot.cumulative_transaction_cost
            ),
            subtitle=(
                f"{money(avg_cost_per_trade)} avg / trade"
            ),
            tone="neutral",
        )

    st.divider()

    # --------------------------------------------------------------
    # Execution history
    # --------------------------------------------------------------

    left_chart, right_chart = st.columns(
        [1.8, 1]
    )

    with left_chart:
        st.subheader("Turnover History")

        st.markdown(
            """
            <div class="section-note">
                Portfolio turnover generated by each daily decision.
            </div>
            """,
            unsafe_allow_html=True,
        )

        turnover_data = history[
            [
                "forecast_origin",
                "turnover",
                "executed",
            ]
        ].copy()

        turnover_data = turnover_data.rename(
            columns={
                "forecast_origin": "Date",
                "turnover": "Turnover",
                "executed": "Executed",
            }
        )

        turnover_chart = (
            alt.Chart(turnover_data)
            .mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                color=COLORS["amber"],
            )
            .encode(
                x=alt.X(
                    "Date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%b %d",
                        labelAngle=0,
                    ),
                ),
                y=alt.Y(
                    "Turnover:Q",
                    title="Turnover",
                    axis=alt.Axis(
                        format=".0%",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Date:T",
                        title="Date",
                        format="%Y-%m-%d",
                    ),
                    alt.Tooltip(
                        "Turnover:Q",
                        title="Turnover",
                        format=".2%",
                    ),
                    alt.Tooltip(
                        "Executed:N",
                        title="Executed",
                    ),
                ],
            )
            .properties(
                height=300,
            )
        )

        threshold_data = pd.DataFrame(
            {
                "Threshold": [0.05]
            }
        )

        threshold_line = (
            alt.Chart(threshold_data)
            .mark_rule(
                strokeDash=[6, 6],
                strokeWidth=2,
                color=COLORS["red"],
            )
            .encode(
                y="Threshold:Q"
            )
        )

        st.altair_chart(
            turnover_chart
            + threshold_line,
            width="stretch",
        )

        st.caption(
            "Dashed line represents the 5% one-way "
            "turnover rebalance threshold."
        )

    with right_chart:
        st.subheader("Cost Summary")

        st.markdown(
            """
            <div class="section-note">
                Trading friction accumulated during paper execution.
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_card(
            "Latest Cycle Cost",
            money(latest_cost),
            subtitle="Most recent recorded cycle",
            tone="neutral",
        )

        st.write("")

        metric_card(
            "Total Transaction Cost",
            money(
                snapshot.cumulative_transaction_cost
            ),
            subtitle=(
                percentage(
                    snapshot.cumulative_transaction_cost
                    / snapshot.initial_equity
                )
                + " of starting capital"
            ),
            tone="neutral",
        )

        st.write("")

        metric_card(
            "Average Cost / Trade",
            money(avg_cost_per_trade),
            subtitle=(
                f"{snapshot.number_of_trades} trades recorded"
            ),
            tone="neutral",
        )

    st.divider()

    # --------------------------------------------------------------
    # Recent trades
    # --------------------------------------------------------------

    st.subheader("Recent Trades")

    st.markdown(
        """
        <div class="section-note">
            Most recent simulated executions from the paper broker.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if trades.empty:
        st.info("No trades recorded.")

    else:
        recent_trades = trades[
            [
                "forecast_origin",
                "asset",
                "side",
                "quantity",
                "price",
                "notional",
                "transaction_cost",
            ]
        ].head(20).copy()

        recent_trades[
            "forecast_origin"
        ] = (
            recent_trades[
                "forecast_origin"
            ]
            .dt.strftime("%Y-%m-%d")
        )

        recent_trades[
            "quantity"
        ] = recent_trades[
            "quantity"
        ].map(
            lambda value: f"{value:,.4f}"
        )

        recent_trades[
            "price"
        ] = recent_trades[
            "price"
        ].map(money)

        recent_trades[
            "notional"
        ] = recent_trades[
            "notional"
        ].map(money)

        recent_trades[
            "transaction_cost"
        ] = recent_trades[
            "transaction_cost"
        ].map(money)

        recent_trades.columns = [
            "Date",
            "Asset",
            "Side",
            "Quantity",
            "Price",
            "Notional",
            "Cost",
        ]

        st.dataframe(
            recent_trades,
            hide_index=True,
            width="stretch",
            column_config={
                "Date": st.column_config.TextColumn(
                    width="medium"
                ),
                "Asset": st.column_config.TextColumn(
                    width="small"
                ),
                "Side": st.column_config.TextColumn(
                    width="small"
                ),
                "Quantity": st.column_config.TextColumn(
                    width="medium"
                ),
                "Price": st.column_config.TextColumn(
                    width="medium"
                ),
                "Notional": st.column_config.TextColumn(
                    width="medium"
                ),
                "Cost": st.column_config.TextColumn(
                    width="medium"
                ),
            },
        )

    st.caption(
        "Paper executions include a 10 bps transaction-cost "
        "assumption and do not represent live capital."
    )


# ------------------------------------------------------------------
# SYSTEM HEALTH
# ------------------------------------------------------------------

with health_tab:
    st.subheader("Production System Health")

    st.markdown(
        """
        <div class="section-note">
            Runtime integrity, scheduler freshness, reconciliation,
            and operational safety checks for the production paper-trading system.
        </div>
        """,
        unsafe_allow_html=True,
    )

    overall_tone = (
        "positive"
        if health.healthy
        else "negative"
    )

    overall_text = (
        "Operational"
        if health.healthy
        else "Attention Required"
    )

    reconciliation_text = (
        "Healthy"
        if health.reconciliation_healthy
        else "Failed"
    )

    scheduler_text = (
        "Healthy"
        if health.scheduler_success_recent
        else "Stale / Missing"
    )

    state_text = (
        "Matched"
        if health.state_matches_latest_history
        else "Mismatch"
    )

    cost_text = (
        "Matched"
        if health.transaction_costs_match
        else "Mismatch"
    )

    # --------------------------------------------------------------
    # Overall status
    # --------------------------------------------------------------

    status_columns = st.columns(
        [1.4, 1, 1, 1]
    )

    with status_columns[0]:
        metric_card(
            "Overall Runtime",
            overall_text,
            subtitle=(
                "All production checks passing"
                if health.healthy
                else "One or more checks require review"
            ),
            tone=overall_tone,
        )

    with status_columns[1]:
        metric_card(
            "Reconciliation",
            reconciliation_text,
            subtitle="State vs persisted history",
            tone=(
                "positive"
                if health.reconciliation_healthy
                else "negative"
            ),
        )

    with status_columns[2]:
        metric_card(
            "Scheduler",
            scheduler_text,
            subtitle="Daily execution freshness",
            tone=(
                "positive"
                if health.scheduler_success_recent
                else "negative"
            ),
        )

    with status_columns[3]:
        metric_card(
            "Duplicate Origins",
            str(
                health.duplicate_forecast_origins
            ),
            subtitle=(
                "No duplicate execution"
                if health.duplicate_forecast_origins == 0
                else "Review required"
            ),
            tone=(
                "positive"
                if health.duplicate_forecast_origins == 0
                else "negative"
            ),
        )

    st.divider()

    # --------------------------------------------------------------
    # Runtime timeline
    # --------------------------------------------------------------

    timeline_col, integrity_col = st.columns(
        [1.5, 1]
    )

    with timeline_col:
        st.subheader("Runtime Timeline")

        st.markdown(
            """
            <div class="section-note">
                Latest production activity recorded by the trading runtime.
            </div>
            """,
            unsafe_allow_html=True,
        )

        timeline_data = pd.DataFrame(
            [
                {
                    "Event": "Latest forecast origin",
                    "Timestamp": format_timestamp(
                        snapshot.latest_forecast_origin
                    ),
                    "Status": "Processed",
                },
                {
                    "Event": "Latest trading run",
                    "Timestamp": format_timestamp(
                        snapshot.latest_run_timestamp_utc
                    ),
                    "Status": "Recorded",
                },
                {
                    "Event": "Last scheduler success",
                    "Timestamp": format_timestamp(
                        health.last_successful_scheduled_run
                    ),
                    "Status": (
                        "Healthy"
                        if health.scheduler_success_recent
                        else "Stale / Missing"
                    ),
                },
            ]
        )

        st.dataframe(
            timeline_data,
            hide_index=True,
            width="stretch",
            column_config={
                "Event": st.column_config.TextColumn(
                    width="medium"
                ),
                "Timestamp": st.column_config.TextColumn(
                    width="large"
                ),
                "Status": st.column_config.TextColumn(
                    width="small"
                ),
            },
        )

    with integrity_col:
        st.subheader("Integrity Checks")

        st.markdown(
            """
            <div class="section-note">
                Internal consistency checks protecting the paper account.
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_card(
            "Portfolio State",
            state_text,
            subtitle="Persisted state vs latest history",
            tone=(
                "positive"
                if health.state_matches_latest_history
                else "negative"
            ),
        )

        st.write("")

        metric_card(
            "Transaction Costs",
            cost_text,
            subtitle="Portfolio vs trade history",
            tone=(
                "positive"
                if health.transaction_costs_match
                else "negative"
            ),
        )

    st.divider()

    # --------------------------------------------------------------
    # Safety controls
    # --------------------------------------------------------------

    st.subheader("Operational Safeguards")

    st.markdown(
        """
        <div class="section-note">
            Controls implemented to reduce execution and runtime risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    safeguard_columns = st.columns(4)

    with safeguard_columns[0]:
        metric_card(
            "Duplicate Protection",
            "Active",
            subtitle="Forecast origin guard",
            tone="positive",
        )

    with safeguard_columns[1]:
        metric_card(
            "Process Lock",
            "Active",
            subtitle="Prevents concurrent execution",
            tone="positive",
        )

    with safeguard_columns[2]:
        metric_card(
            "Startup Validation",
            "Active",
            subtitle="Checks state before execution",
            tone="positive",
        )

    with safeguard_columns[3]:
        metric_card(
            "Reconciliation",
            "Active",
            subtitle="Validates every completed cycle",
            tone="positive",
        )

    st.divider()

    # --------------------------------------------------------------
    # Runtime issues
    # --------------------------------------------------------------

    st.subheader("Detected Issues")

    if health.issues:
        st.markdown(
            """
            <div class="section-note">
                The monitoring layer detected the following runtime issues.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for issue in health.issues:
            st.error(issue)

    else:
        st.success(
            "No runtime integrity issues detected. "
            "Scheduler, persisted state, transaction costs, "
            "and reconciliation checks are consistent."
        )

    st.caption(
        "System Health is read-only. Operational safeguards are "
        "implemented in the trading runtime, not in the dashboard."
    )