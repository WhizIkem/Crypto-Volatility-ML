from __future__ import annotations

from crypto_volatility.monitoring.performance import (
    build_monitoring_snapshot,
)
from crypto_volatility.monitoring.runtime_health import (
    check_runtime_health,
)

STATE_PATH = (
    "logs/paper_portfolio_state.json"
)

PORTFOLIO_HISTORY_PATH = (
    "logs/paper_portfolio_history.csv"
)

TRADE_HISTORY_PATH = (
    "logs/paper_trades.csv"
)

SCHEDULER_LOG_PATH = (
    "logs/paper_scheduler.log"
)


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _percentage(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
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

    print()
    print("Paper-Trading Monitoring")
    print("=" * 48)

    print()
    print("Portfolio")
    print("-" * 48)

    print(
        "Initial equity:          "
        f"{_money(snapshot.initial_equity)}"
    )

    print(
        "Current equity:          "
        f"{_money(snapshot.current_equity)}"
    )

    print(
        "Cumulative P&L:          "
        f"{_money(snapshot.cumulative_pnl)}"
    )

    print(
        "Cumulative return:       "
        f"{_percentage(snapshot.cumulative_return)}"
    )

    print(
        "Latest period P&L:       "
        f"{_money(snapshot.latest_period_pnl)}"
    )

    print(
        "Latest period return:    "
        f"{_percentage(snapshot.latest_period_return)}"
    )

    print()
    print("Risk")
    print("-" * 48)

    if snapshot.risk_metrics_available:
        assert (
            snapshot.annualized_volatility
            is not None
        )

        print(
            "Annualized volatility:   "
            f"{_percentage(snapshot.annualized_volatility)}"
        )

        if snapshot.sharpe_ratio is None:
            print(
                "Sharpe ratio:            "
                "Unavailable"
            )

        else:
            print(
                "Sharpe ratio:            "
                f"{snapshot.sharpe_ratio:.3f}"
            )

    else:
        print(
            "Annualized volatility:   "
            "Insufficient history"
        )

        print(
            "Sharpe ratio:            "
            "Insufficient history"
        )

        print(
            "Risk observations:       "
            f"{snapshot.risk_observations}/"
            f"{snapshot.minimum_risk_observations}"
            " minimum"
        )

    print(
        "Maximum drawdown:        "
        f"{_percentage(snapshot.max_drawdown)}"
    )

    print(
        "Current drawdown:        "
        f"{_percentage(snapshot.current_drawdown)}"
    )

    print()
    print("Allocation")
    print("-" * 48)

    for (
        asset,
        weight,
    ) in snapshot.current_weights.items():
        print(
            f"{asset:<4} weight:              "
            f"{_percentage(weight)}"
        )

    print(
        "Risky exposure:          "
        f"{_percentage(snapshot.risky_exposure)}"
    )

    print(
        "Cash weight:             "
        f"{_percentage(snapshot.cash_weight)}"
    )

    print(
        "Cash balance:            "
        f"{_money(snapshot.cash)}"
    )

    print()
    print("Execution")
    print("-" * 48)

    print(
        "Cycles:                  "
        f"{snapshot.number_of_cycles}"
    )

    print(
        "Rebalance signals:       "
        f"{snapshot.number_of_rebalances}"
    )

    print(
        "Executed cycles:         "
        f"{snapshot.number_of_executed_cycles}"
    )

    print(
        "Trades:                  "
        f"{snapshot.number_of_trades}"
    )

    print(
        "Latest transaction cost: "
        f"{_money(snapshot.latest_transaction_cost)}"
    )

    print(
        "Cumulative costs:        "
        f"{_money(snapshot.cumulative_transaction_cost)}"
    )

    print()
    print("Runtime")
    print("-" * 48)

    print(
        "Latest forecast origin:  "
        f"{snapshot.latest_forecast_origin}"
    )

    print(
        "Latest recorded run:     "
        f"{snapshot.latest_run_timestamp_utc}"
    )

    print(
        "Last scheduled success:  "
        f"{health.last_successful_scheduled_run}"
    )

    print(
        "Scheduler success recent:"
        f"  {health.scheduler_success_recent}"
    )

    print(
        "Reconciliation healthy:  "
        f"{health.reconciliation_healthy}"
    )

    print(
        "Runtime healthy:         "
        f"{health.healthy}"
    )

    if health.issues:
        print()
        print("Issues")
        print("-" * 48)

        for issue in health.issues:
            print(f"- {issue}")

    print()


if __name__ == "__main__":
    main()