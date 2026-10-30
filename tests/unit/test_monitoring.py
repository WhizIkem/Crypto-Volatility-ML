from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from crypto_volatility.monitoring.performance import (
    build_monitoring_snapshot,
)
from crypto_volatility.monitoring.runtime_health import (
    check_runtime_health,
)


def _write_runtime_files(tmp_path):
    portfolio_path = (
        tmp_path
        / "paper_portfolio_history.csv"
    )

    trades_path = (
        tmp_path
        / "paper_trades.csv"
    )

    state_path = (
        tmp_path
        / "paper_portfolio_state.json"
    )

    scheduler_log_path = (
        tmp_path
        / "paper_scheduler.log"
    )

    portfolio = pd.DataFrame(
        [
            {
                "run_timestamp_utc": (
                    "2026-10-27T01:00:00+00:00"
                ),
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "equity_before": 1000.0,
                "equity_after": 990.0,
                "cash": 400.0,
                "cash_weight": (
                    400.0 / 990.0
                ),
                "turnover": 0.60,
                "should_rebalance": True,
                "executed": True,
                "transaction_cost": 10.0,
            },
            {
                "run_timestamp_utc": (
                    "2026-10-28T01:00:00+00:00"
                ),
                "forecast_origin": (
                    "2026-10-27T00:00:00+00:00"
                ),
                "equity_before": 1010.0,
                "equity_after": 1005.0,
                "cash": 405.0,
                "cash_weight": (
                    405.0 / 1005.0
                ),
                "turnover": 0.10,
                "should_rebalance": True,
                "executed": True,
                "transaction_cost": 5.0,
            },
        ]
    )

    portfolio.to_csv(
        portfolio_path,
        index=False,
    )

    trades = pd.DataFrame(
        [
            {
                "run_timestamp_utc": (
                    "2026-10-27T01:00:00+00:00"
                ),
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "asset": "BTC",
                "side": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "notional": 100.0,
                "transaction_cost": 10.0,
            },
            {
                "run_timestamp_utc": (
                    "2026-10-28T01:00:00+00:00"
                ),
                "forecast_origin": (
                    "2026-10-27T00:00:00+00:00"
                ),
                "asset": "BTC",
                "side": "SELL",
                "quantity": 0.5,
                "price": 100.0,
                "notional": 50.0,
                "transaction_cost": 5.0,
            },
        ]
    )

    trades.to_csv(
        trades_path,
        index=False,
    )

    state = {
        "cash": 405.0,
        "quantities": {
            "BTC": 6.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        "last_prices": {
            "BTC": 100.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        "created_at": (
            "2026-10-27T00:00:00+00:00"
        ),
        "updated_at": (
            "2026-10-28T01:00:00+00:00"
        ),
        "last_forecast_timestamp": (
            "2026-10-27T00:00:00+00:00"
        ),
        "last_rebalance_timestamp": (
            "2026-10-27T00:00:00+00:00"
        ),
    }

    with state_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            state,
            handle,
        )

    scheduler_log_path.write_text(
        "[2026-10-28T01:05:00Z] "
        "Scheduled paper-trading cycle "
        "finished successfully\n",
        encoding="utf-8",
    )

    return (
        portfolio_path,
        trades_path,
        state_path,
        scheduler_log_path,
    )


def test_build_monitoring_snapshot(
    tmp_path,
):
    (
        portfolio_path,
        trades_path,
        state_path,
        _,
    ) = _write_runtime_files(
        tmp_path
    )

    snapshot = (
        build_monitoring_snapshot(
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trades_path
            ),
            state_path=state_path,
        )
    )

    assert snapshot.initial_equity == 1000.0
    assert snapshot.current_equity == 1005.0
    assert snapshot.cumulative_pnl == 5.0

    assert np.isclose(
        snapshot.cumulative_return,
        0.005,
    )

    assert snapshot.number_of_cycles == 2
    assert snapshot.number_of_rebalances == 2
    assert (
        snapshot.number_of_executed_cycles
        == 2
    )
    assert snapshot.number_of_trades == 2

    assert (
        snapshot.cumulative_transaction_cost
        == 15.0
    )

    assert (
        snapshot.latest_transaction_cost
        == 5.0
    )

    assert np.isclose(
        snapshot.current_weights["BTC"],
        600.0 / 1005.0,
    )

    assert np.isclose(
        snapshot.cash_weight,
        405.0 / 1005.0,
    )

    assert np.isclose(
        snapshot.risky_exposure,
        600.0 / 1005.0,
    )


def test_risk_metrics_require_minimum_history(
    tmp_path,
):
    (
        portfolio_path,
        trades_path,
        state_path,
        _,
    ) = _write_runtime_files(
        tmp_path
    )

    snapshot = (
        build_monitoring_snapshot(
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trades_path
            ),
            state_path=state_path,
        )
    )

    assert not snapshot.risk_metrics_available
    assert snapshot.risk_observations == 2
    assert snapshot.annualized_volatility is None
    assert snapshot.sharpe_ratio is None


def test_runtime_health_passes_for_consistent_files(
    tmp_path,
):
    (
        portfolio_path,
        trades_path,
        state_path,
        scheduler_log_path,
    ) = _write_runtime_files(
        tmp_path
    )

    health = check_runtime_health(
        portfolio_history_path=(
            portfolio_path
        ),
        trade_history_path=(
            trades_path
        ),
        state_path=state_path,
        scheduler_log_path=(
            scheduler_log_path
        ),
        now_utc=datetime(
            2026,
            10,
            28,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    assert health.files_present
    assert health.state_matches_latest_history
    assert health.transaction_costs_match
    assert health.reconciliation_healthy
    assert health.scheduler_success_found
    assert health.scheduler_success_recent
    assert health.healthy
    assert health.issues == ()


def test_runtime_health_detects_cost_mismatch(
    tmp_path,
):
    (
        portfolio_path,
        trades_path,
        state_path,
        scheduler_log_path,
    ) = _write_runtime_files(
        tmp_path
    )

    trades = pd.read_csv(
        trades_path
    )

    trades.loc[
        0,
        "transaction_cost",
    ] = 99.0

    trades.to_csv(
        trades_path,
        index=False,
    )

    health = check_runtime_health(
        portfolio_history_path=(
            portfolio_path
        ),
        trade_history_path=(
            trades_path
        ),
        state_path=state_path,
        scheduler_log_path=(
            scheduler_log_path
        ),
        now_utc=datetime(
            2026,
            10,
            28,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    assert not health.transaction_costs_match
    assert not health.reconciliation_healthy
    assert not health.healthy


def test_runtime_health_detects_stale_scheduler(
    tmp_path,
):
    (
        portfolio_path,
        trades_path,
        state_path,
        scheduler_log_path,
    ) = _write_runtime_files(
        tmp_path
    )

    health = check_runtime_health(
        portfolio_history_path=(
            portfolio_path
        ),
        trade_history_path=(
            trades_path
        ),
        state_path=state_path,
        scheduler_log_path=(
            scheduler_log_path
        ),
        now_utc=datetime(
            2026,
            10,
            30,
            0,
            0,
            tzinfo=UTC,
        ),
    )

    assert health.reconciliation_healthy
    assert health.scheduler_success_found
    assert not health.scheduler_success_recent
    assert not health.healthy