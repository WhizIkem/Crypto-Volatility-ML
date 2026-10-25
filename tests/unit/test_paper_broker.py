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


def test_buy_from_cash():
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    prices = pd.Series(
        {
            "BTC": 80_000.0,
            "ETH": 2_500.0,
            "SOL": 100.0,
            "BNB": 750.0,
        }
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
        }
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

    assert result.total_transaction_cost == pytest.approx(
        600.0
    )

    assert state.cash == pytest.approx(
        399_400.0
    )


def test_sell_position():
    state = (
        create_initial_portfolio_state(
            100_000.0
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
        }
    )

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
        }
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

    assert trade.transaction_cost == pytest.approx(
        40.0
    )

    assert state.quantities[
        "BTC"
    ] == pytest.approx(
        0.5
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
        }
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
        }
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

    assert state.quantities[
        "BTC"
    ] == pytest.approx(
        0.0
    )

    assert state.quantities[
        "ETH"
    ] == pytest.approx(
        40.0
    )


def test_insufficient_cash_raises():
    state = (
        create_initial_portfolio_state(
            1_000.0
        )
    )

    prices = pd.Series(
        {
            "BTC": 80_000.0,
            "ETH": 2_500.0,
            "SOL": 100.0,
            "BNB": 750.0,
        }
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
        }
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