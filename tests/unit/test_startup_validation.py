import json

import pandas as pd
import pytest

from crypto_volatility.trading.portfolio_state import (
    create_initial_portfolio_state,
    save_portfolio_state,
    update_market_prices,
)
from crypto_volatility.trading.startup_validation import (
    StartupIntegrityError,
    validate_paper_runtime_startup,
)


def build_paths(
    tmp_path,
):
    return (
        tmp_path
        / "paper_portfolio_state.json",
        tmp_path
        / "paper_portfolio_history.csv",
        tmp_path
        / "paper_trades.csv",
    )


def build_valid_runtime(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_paths(
        tmp_path
    )

    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    state.cash = 400_000.0

    state.quantities = {
        "BTC": 2.0,
        "ETH": 40.0,
        "SOL": 1_000.0,
        "BNB": 200.0,
    }

    prices = pd.Series(
        {
            "BTC": 100_000.0,
            "ETH": 2_500.0,
            "SOL": 100.0,
            "BNB": 500.0,
        },
        dtype=float,
    )

    update_market_prices(
        state,
        prices,
    )

    state.last_forecast_timestamp = (
        "2026-10-27T00:00:00+00:00"
    )

    save_portfolio_state(
        state,
        state_path,
    )

    portfolio_history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": (
                    "2026-10-28T01:00:00+00:00"
                ),
                "forecast_origin": (
                    "2026-10-27T00:00:00+00:00"
                ),
                "equity_before": 900_000.0,
                "equity_after": 900_000.0,
                "cash": 400_000.0,
                "cash_weight": (
                    400_000.0
                    / 900_000.0
                ),
                "turnover": 0.0,
                "should_rebalance": False,
                "executed": False,
                "transaction_cost": 0.0,
                "BTC_quantity": 2.0,
                "BTC_price": 100_000.0,
                "ETH_quantity": 40.0,
                "ETH_price": 2_500.0,
                "SOL_quantity": 1_000.0,
                "SOL_price": 100.0,
                "BNB_quantity": 200.0,
                "BNB_price": 500.0,
            }
        ]
    )

    portfolio_history.to_csv(
        portfolio_path,
        index=False,
    )

    return (
        state_path,
        portfolio_path,
        trade_path,
    )


def test_clean_start_is_valid(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_paths(
        tmp_path
    )

    report = (
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )
    )

    assert report.clean_start
    assert not report.state_exists


def test_initial_state_without_history_is_valid(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_paths(
        tmp_path
    )

    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    save_portfolio_state(
        state,
        state_path,
    )

    report = (
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )
    )

    assert not report.clean_start
    assert report.state_exists
    assert not report.portfolio_history_exists


def test_corrupted_state_is_rejected(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_paths(
        tmp_path
    )

    state_path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        StartupIntegrityError,
        match="corrupted or invalid",
    ):
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )


def test_processed_state_without_history_is_rejected(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_paths(
        tmp_path
    )

    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    state.last_forecast_timestamp = (
        "2026-10-27T00:00:00+00:00"
    )

    save_portfolio_state(
        state,
        state_path,
    )

    with pytest.raises(
        StartupIntegrityError,
        match="processed forecast",
    ):
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )


def test_journal_without_state_is_rejected(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_paths(
        tmp_path
    )

    pd.DataFrame(
        [
            {
                "forecast_origin": (
                    "2026-10-27T00:00:00+00:00"
                )
            }
        ]
    ).to_csv(
        portfolio_path,
        index=False,
    )

    with pytest.raises(
        StartupIntegrityError,
        match="state is missing",
    ):
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )


def test_valid_runtime_passes(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_valid_runtime(
        tmp_path
    )

    report = (
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )
    )

    assert not report.clean_start

    assert (
        report.latest_forecast_origin
        == "2026-10-27T00:00:00+00:00"
    )

    assert (
        report.cumulative_transaction_cost
        == pytest.approx(
            0.0
        )
    )


def test_duplicate_portfolio_origins_are_rejected(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_valid_runtime(
        tmp_path
    )

    frame = pd.read_csv(
        portfolio_path
    )

    pd.concat(
        [
            frame,
            frame,
        ],
        ignore_index=True,
    ).to_csv(
        portfolio_path,
        index=False,
    )

    with pytest.raises(
        StartupIntegrityError,
        match="duplicate forecast origins",
    ):
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )


def test_state_timestamp_mismatch_is_rejected(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_valid_runtime(
        tmp_path
    )

    payload = json.loads(
        state_path.read_text(
            encoding="utf-8"
        )
    )

    payload[
        "last_forecast_timestamp"
    ] = (
        "2026-10-26T00:00:00+00:00"
    )

    state_path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        StartupIntegrityError,
        match="last forecast timestamp",
    ):
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )


def test_orphan_trade_origin_is_rejected(
    tmp_path,
):
    (
        state_path,
        portfolio_path,
        trade_path,
    ) = build_valid_runtime(
        tmp_path
    )

    trade_history = pd.DataFrame(
        [
            {
                "forecast_origin": (
                    "2026-10-25T00:00:00+00:00"
                ),
                "asset": "BTC",
                "side": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "notional": 100.0,
                "transaction_cost": 0.1,
            }
        ]
    )

    trade_history.to_csv(
        trade_path,
        index=False,
    )

    with pytest.raises(
        StartupIntegrityError,
        match="missing from portfolio history",
    ):
        validate_paper_runtime_startup(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_path
            ),
            trade_history_path=(
                trade_path
            ),
            initial_capital=(
                1_000_000.0
            ),
        )