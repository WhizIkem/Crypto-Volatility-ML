from copy import deepcopy

import pandas as pd
import pytest

from crypto_volatility.trading.paper_broker import (
    simulate_paper_rebalance,
)
from crypto_volatility.trading.portfolio_state import (
    create_initial_portfolio_state,
    target_quantities,
    update_market_prices,
)


@pytest.fixture
def prices():
    return pd.Series(
        {
            "BTC": 80_000.0,
            "ETH": 2_500.0,
            "SOL": 100.0,
            "BNB": 750.0,
        },
        dtype=float,
    )


def test_buy_from_cash(
    prices,
):
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    update_market_prices(
        state,
        prices,
    )

    weights = pd.Series(
        {
            "BTC": 0.20,
            "ETH": 0.15,
            "SOL": 0.10,
            "BNB": 0.15,
        },
        dtype=float,
    )

    targets = target_quantities(
        portfolio_equity=(
            state.equity()
        ),
        target_weights=weights,
        prices=prices,
    )

    result = simulate_paper_rebalance(
        state=state,
        target_quantities=targets,
        prices=prices,
        transaction_cost_bps=10.0,
    )

    assert len(
        result.trades
    ) == 4

    assert all(
        trade.side == "BUY"
        for trade in result.trades
    )

    assert (
        result.total_transaction_cost
        == pytest.approx(
            600.0
        )
    )

    assert state.cash == pytest.approx(
        399_400.0
    )


def test_sell_position(
    prices,
):
    state = (
        create_initial_portfolio_state(
            100_000.0
        )
    )

    state.quantities[
        "BTC"
    ] = 1.0

    update_market_prices(
        state,
        prices,
    )

    targets = pd.Series(
        {
            "BTC": 0.5,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
    )

    result = simulate_paper_rebalance(
        state=state,
        target_quantities=targets,
        prices=prices,
        transaction_cost_bps=10.0,
    )

    assert len(
        result.trades
    ) == 1

    trade = result.trades[
        0
    ]

    assert trade.side == "SELL"

    assert trade.quantity == pytest.approx(
        0.5
    )

    assert trade.notional == pytest.approx(
        40_000.0
    )

    assert (
        trade.transaction_cost
        == pytest.approx(
            40.0
        )
    )

    assert (
        state.quantities[
            "BTC"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert state.cash == pytest.approx(
        139_960.0
    )


def test_sell_before_buy():
    state = (
        create_initial_portfolio_state(
            0.01
        )
    )

    state.quantities[
        "BTC"
    ] = 1.0

    prices = pd.Series(
        {
            "BTC": 80_000.0,
            "ETH": 2_000.0,
            "SOL": 100.0,
            "BNB": 750.0,
        },
        dtype=float,
    )

    update_market_prices(
        state,
        prices,
    )

    targets = pd.Series(
        {
            "BTC": 0.0,
            "ETH": 40.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
    )

    result = simulate_paper_rebalance(
        state=state,
        target_quantities=targets,
        prices=prices,
        transaction_cost_bps=0.0,
    )

    assert result.trades[
        0
    ].side == "SELL"

    assert result.trades[
        1
    ].side == "BUY"

    assert (
        state.quantities[
            "BTC"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        state.quantities[
            "ETH"
        ]
        == pytest.approx(
            40.0
        )
    )


def test_insufficient_cash_raises(
    prices,
):
    state = (
        create_initial_portfolio_state(
            1_000.0
        )
    )

    update_market_prices(
        state,
        prices,
    )

    targets = pd.Series(
        {
            "BTC": 1.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="Insufficient cash",
    ):
        simulate_paper_rebalance(
            state=state,
            target_quantities=targets,
            prices=prices,
        )


def test_failed_execution_leaves_state_unchanged():
    """
    A failed rebalance must not partially modify the portfolio.
    """
    state = (
        create_initial_portfolio_state(
            100.0
        )
    )

    state.quantities[
        "BTC"
    ] = 1.0

    prices = pd.Series(
        {
            "BTC": 80_000.0,
            "ETH": 2_500.0,
            "SOL": 100.0,
            "BNB": 750.0,
        },
        dtype=float,
    )

    update_market_prices(
        state,
        prices,
    )

    before = deepcopy(
        state
    )

    # Sell BTC first, then request far more ETH than sale
    # proceeds can finance.
    targets = pd.Series(
        {
            "BTC": 0.0,
            "ETH": 100.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="Insufficient cash",
    ):
        simulate_paper_rebalance(
            state=state,
            target_quantities=targets,
            prices=prices,
            transaction_cost_bps=10.0,
        )

    assert (
        state.cash
        == pytest.approx(
            before.cash
        )
    )

    assert (
        state.quantities
        == before.quantities
    )

    assert (
        state.last_prices
        == before.last_prices
    )

    assert (
        state.updated_at
        == before.updated_at
    )

    assert (
        state.last_rebalance_timestamp
        == before.last_rebalance_timestamp
    )


def test_no_change_target_generates_no_trades(
    prices,
):
    state = (
        create_initial_portfolio_state(
            100_000.0
        )
    )

    state.quantities = {
        "BTC": 0.25,
        "ETH": 2.0,
        "SOL": 10.0,
        "BNB": 1.0,
    }

    update_market_prices(
        state,
        prices,
    )

    targets = (
        state.quantity_series()
    )

    result = simulate_paper_rebalance(
        state=state,
        target_quantities=targets,
        prices=prices,
        transaction_cost_bps=10.0,
    )

    assert result.trades == []

    assert (
        result.total_transaction_cost
        == pytest.approx(
            0.0
        )
    )


def test_transaction_cost_reduces_equity(
    prices,
):
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    update_market_prices(
        state,
        prices,
    )

    equity_before = (
        state.equity()
    )

    weights = pd.Series(
        {
            "BTC": 0.20,
            "ETH": 0.15,
            "SOL": 0.10,
            "BNB": 0.15,
        },
        dtype=float,
    )

    targets = target_quantities(
        portfolio_equity=(
            equity_before
        ),
        target_weights=weights,
        prices=prices,
    )

    result = simulate_paper_rebalance(
        state=state,
        target_quantities=targets,
        prices=prices,
        transaction_cost_bps=10.0,
    )

    equity_after = (
        state.equity()
    )

    assert (
        equity_before
        - equity_after
        == pytest.approx(
            result.total_transaction_cost
        )
    )


def test_successful_execution_never_creates_negative_cash(
    prices,
):
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    update_market_prices(
        state,
        prices,
    )

    weights = pd.Series(
        {
            "BTC": 0.18,
            "ETH": 0.14,
            "SOL": 0.10,
            "BNB": 0.17,
        },
        dtype=float,
    )

    targets = target_quantities(
        portfolio_equity=(
            state.equity()
        ),
        target_weights=weights,
        prices=prices,
    )

    simulate_paper_rebalance(
        state=state,
        target_quantities=targets,
        prices=prices,
        transaction_cost_bps=10.0,
    )

    assert state.cash >= 0.0


def test_negative_transaction_cost_is_rejected(
    prices,
):
    state = (
        create_initial_portfolio_state(
            100_000.0
        )
    )

    targets = pd.Series(
        {
            "BTC": 0.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="transaction_cost_bps cannot be negative",
    ):
        simulate_paper_rebalance(
            state=state,
            target_quantities=targets,
            prices=prices,
            transaction_cost_bps=-1.0,
        )