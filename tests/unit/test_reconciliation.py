import pandas as pd
import pytest

from crypto_volatility.trading.portfolio_state import (
    create_initial_portfolio_state,
    update_market_prices,
)
from crypto_volatility.trading.reconciliation import (
    assert_reconciliation_passed,
    cumulative_portfolio_cost,
    cumulative_trade_cost,
    reconcile_latest_paper_state,
    reconstruct_equity_from_history_row,
)


def build_state():
    state = create_initial_portfolio_state(
        400.0
    )

    state.quantities = {
        "BTC": 1.0,
        "ETH": 2.0,
        "SOL": 3.0,
        "BNB": 4.0,
    }

    prices = pd.Series(
        {
            "BTC": 100.0,
            "ETH": 50.0,
            "SOL": 20.0,
            "BNB": 10.0,
        },
        dtype=float,
    )

    update_market_prices(
        state,
        prices,
    )

    return state


def build_portfolio_history(
    *,
    transaction_cost: float = 0.0,
):
    return pd.DataFrame(
        [
            {
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "equity_after": 700.0,
                "cash": 400.0,
                "transaction_cost": (
                    transaction_cost
                ),
                "BTC_quantity": 1.0,
                "BTC_price": 100.0,
                "ETH_quantity": 2.0,
                "ETH_price": 50.0,
                "SOL_quantity": 3.0,
                "SOL_price": 20.0,
                "BNB_quantity": 4.0,
                "BNB_price": 10.0,
            }
        ]
    )


def empty_trade_history():
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


def test_reconstruct_equity():
    history = (
        build_portfolio_history()
    )

    reconstructed = (
        reconstruct_equity_from_history_row(
            history.iloc[
                0
            ]
        )
    )

    assert reconstructed == pytest.approx(
        700.0
    )


def test_clean_reconciliation_passes():
    state = build_state()

    portfolio_history = (
        build_portfolio_history()
    )

    trade_history = (
        empty_trade_history()
    )

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                trade_history
            ),
        )
    )

    assert report.passed

    assert report.state_matches_history
    assert report.equity_matches

    assert (
        report.latest_transaction_cost_matches
    )

    assert (
        report.cumulative_transaction_cost_matches
    )

    assert (
        report.latest_portfolio_transaction_cost
        == pytest.approx(
            0.0
        )
    )

    assert (
        report.latest_trade_transaction_cost
        == pytest.approx(
            0.0
        )
    )

    assert (
        report.cumulative_portfolio_transaction_cost
        == pytest.approx(
            0.0
        )
    )

    assert (
        report.cumulative_trade_transaction_cost
        == pytest.approx(
            0.0
        )
    )

    assert_reconciliation_passed(
        report
    )


def test_transaction_cost_reconciliation():
    state = build_state()

    portfolio_history = (
        build_portfolio_history(
            transaction_cost=12.5
        )
    )

    trade_history = pd.DataFrame(
        [
            {
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "asset": "BTC",
                "side": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "notional": 100.0,
                "transaction_cost": 5.0,
            },
            {
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "asset": "ETH",
                "side": "BUY",
                "quantity": 1.0,
                "price": 50.0,
                "notional": 50.0,
                "transaction_cost": 7.5,
            },
        ]
    )

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                trade_history
            ),
        )
    )

    assert (
        report.latest_trade_transaction_cost
        == pytest.approx(
            12.5
        )
    )

    assert (
        report.latest_portfolio_transaction_cost
        == pytest.approx(
            12.5
        )
    )

    assert (
        report.cumulative_trade_transaction_cost
        == pytest.approx(
            12.5
        )
    )

    assert (
        report.cumulative_portfolio_transaction_cost
        == pytest.approx(
            12.5
        )
    )

    assert (
        report.latest_transaction_cost_matches
    )

    assert (
        report.cumulative_transaction_cost_matches
    )

    assert report.passed


def test_state_quantity_mismatch_fails():
    state = build_state()

    portfolio_history = (
        build_portfolio_history()
    )

    portfolio_history.loc[
        0,
        "BTC_quantity",
    ] = 2.0

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                empty_trade_history()
            ),
        )
    )

    assert not report.passed
    assert not report.state_matches_history

    with pytest.raises(
        ValueError,
        match="reconciliation failed",
    ):
        assert_reconciliation_passed(
            report
        )


def test_equity_mismatch_fails():
    state = build_state()

    portfolio_history = (
        build_portfolio_history()
    )

    portfolio_history.loc[
        0,
        "equity_after",
    ] = 999.0

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                empty_trade_history()
            ),
        )
    )

    assert not report.equity_matches
    assert not report.passed


def test_latest_transaction_cost_mismatch_fails():
    state = build_state()

    portfolio_history = (
        build_portfolio_history(
            transaction_cost=10.0
        )
    )

    trade_history = pd.DataFrame(
        [
            {
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "asset": "BTC",
                "side": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "notional": 100.0,
                "transaction_cost": 5.0,
            }
        ]
    )

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                trade_history
            ),
        )
    )

    assert not (
        report.latest_transaction_cost_matches
    )

    assert not (
        report.cumulative_transaction_cost_matches
    )

    assert not report.passed


def test_cumulative_portfolio_cost():
    portfolio_history = pd.DataFrame(
        [
            {
                "transaction_cost": 10.0,
            },
            {
                "transaction_cost": 0.0,
            },
            {
                "transaction_cost": 5.5,
            },
        ]
    )

    total = cumulative_portfolio_cost(
        portfolio_history
    )

    assert total == pytest.approx(
        15.5
    )


def test_cumulative_trade_cost():
    trade_history = pd.DataFrame(
        [
            {
                "transaction_cost": 4.0,
            },
            {
                "transaction_cost": 6.0,
            },
            {
                "transaction_cost": 2.5,
            },
        ]
    )

    total = cumulative_trade_cost(
        trade_history
    )

    assert total == pytest.approx(
        12.5
    )


def test_cumulative_cost_mismatch_detected():
    state = build_state()

    portfolio_history = pd.DataFrame(
        [
            {
                "forecast_origin": (
                    "2026-10-25T00:00:00+00:00"
                ),
                "equity_after": 700.0,
                "cash": 400.0,
                "transaction_cost": 10.0,
                "BTC_quantity": 1.0,
                "BTC_price": 100.0,
                "ETH_quantity": 2.0,
                "ETH_price": 50.0,
                "SOL_quantity": 3.0,
                "SOL_price": 20.0,
                "BNB_quantity": 4.0,
                "BNB_price": 10.0,
            },
            {
                "forecast_origin": (
                    "2026-10-26T00:00:00+00:00"
                ),
                "equity_after": 700.0,
                "cash": 400.0,
                "transaction_cost": 0.0,
                "BTC_quantity": 1.0,
                "BTC_price": 100.0,
                "ETH_quantity": 2.0,
                "ETH_price": 50.0,
                "SOL_quantity": 3.0,
                "SOL_price": 20.0,
                "BNB_quantity": 4.0,
                "BNB_price": 10.0,
            },
        ]
    )

    trade_history = empty_trade_history()

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                trade_history
            ),
        )
    )

    # Latest cycle itself has zero cost in both journals.
    assert (
        report.latest_transaction_cost_matches
    )

    # But cumulative history reveals the missing $10 trade cost.
    assert not (
        report.cumulative_transaction_cost_matches
    )

    assert (
        report.cumulative_portfolio_transaction_cost
        == pytest.approx(
            10.0
        )
    )

    assert (
        report.cumulative_trade_transaction_cost
        == pytest.approx(
            0.0
        )
    )

    assert not report.passed