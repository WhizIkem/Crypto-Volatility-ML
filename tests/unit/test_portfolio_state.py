import pandas as pd
import pytest

from crypto_volatility.trading.portfolio_state import (
    create_initial_portfolio_state,
    load_portfolio_state,
    save_portfolio_state,
    target_position_values,
    target_quantities,
    trade_quantities,
    update_market_prices,
)


def test_create_initial_state():
    state = (
        create_initial_portfolio_state(
            100_000.0
        )
    )

    assert state.cash == pytest.approx(
        100_000.0
    )

    assert sum(
        state.quantities.values()
    ) == pytest.approx(
        0.0
    )


def test_update_prices_and_equity():
    state = (
        create_initial_portfolio_state(
            100_000.0
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

    assert state.equity() == pytest.approx(
        100_000.0
    )

    assert state.cash_weight() == pytest.approx(
        1.0
    )


def test_asset_values_and_weights():
    state = (
        create_initial_portfolio_state(
            50_000.0
        )
    )

    state.quantities = {
        "BTC": 0.5,
        "ETH": 2.0,
        "SOL": 10.0,
        "BNB": 1.0,
    }

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

    values = (
        state.asset_values()
    )

    assert values[
        "BTC"
    ] == pytest.approx(
        40_000.0
    )

    assert values[
        "ETH"
    ] == pytest.approx(
        5_000.0
    )

    expected_equity = (
        50_000.0
        + 40_000.0
        + 5_000.0
        + 1_000.0
        + 750.0
    )

    assert state.equity() == pytest.approx(
        expected_equity
    )

    weights = (
        state.risky_weights()
    )

    assert weights[
        "BTC"
    ] == pytest.approx(
        40_000.0
        / expected_equity
    )


def test_target_position_values():
    weights = pd.Series(
        {
            "BTC": 0.20,
            "ETH": 0.15,
            "SOL": 0.10,
            "BNB": 0.15,
        }
    )

    values = target_position_values(
        portfolio_equity=100_000.0,
        target_weights=weights,
    )

    assert values[
        "BTC"
    ] == pytest.approx(
        20_000.0
    )

    assert values[
        "ETH"
    ] == pytest.approx(
        15_000.0
    )


def test_target_quantities():
    weights = pd.Series(
        {
            "BTC": 0.20,
            "ETH": 0.15,
            "SOL": 0.10,
            "BNB": 0.15,
        }
    )

    prices = pd.Series(
        {
            "BTC": 80_000.0,
            "ETH": 2_500.0,
            "SOL": 100.0,
            "BNB": 750.0,
        }
    )

    quantities = target_quantities(
        portfolio_equity=100_000.0,
        target_weights=weights,
        prices=prices,
    )

    assert quantities[
        "BTC"
    ] == pytest.approx(
        0.25
    )

    assert quantities[
        "ETH"
    ] == pytest.approx(
        6.0
    )

    assert quantities[
        "SOL"
    ] == pytest.approx(
        100.0
    )

    assert quantities[
        "BNB"
    ] == pytest.approx(
        20.0
    )


def test_trade_quantities_from_cash():
    state = (
        create_initial_portfolio_state(
            100_000.0
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

    trades = trade_quantities(
        state=state,
        target_weights=weights,
        prices=prices,
    )

    assert trades[
        "BTC"
    ] == pytest.approx(
        0.25
    )

    assert trades[
        "ETH"
    ] == pytest.approx(
        6.0
    )


def test_trade_quantities_can_generate_sell():
    state = (
        create_initial_portfolio_state(
            20_000.0
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

    target_weights = pd.Series(
        {
            "BTC": 0.40,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        }
    )

    trades = trade_quantities(
        state=state,
        target_weights=target_weights,
        prices=prices,
    )

    assert trades[
        "BTC"
    ] < 0


def test_state_round_trip(tmp_path):
    path = (
        tmp_path
        / "portfolio_state.json"
    )

    state = (
        create_initial_portfolio_state(
            100_000.0
        )
    )

    state.quantities[
        "BTC"
    ] = 0.25

    state.last_prices[
        "BTC"
    ] = 80_000.0

    save_portfolio_state(
        state,
        path,
    )

    loaded = (
        load_portfolio_state(
            path
        )
    )

    assert loaded.cash == pytest.approx(
        100_000.0
    )

    assert loaded.quantities[
        "BTC"
    ] == pytest.approx(
        0.25
    )

    assert loaded.last_prices[
        "BTC"
    ] == pytest.approx(
        80_000.0
    )