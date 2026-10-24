import pandas as pd
import pytest

from crypto_volatility.portfolio.backtest import (
    build_equity_curve,
    calculate_max_drawdown,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
)


def test_turnover_is_zero_when_weights_do_not_change():
    current = pd.Series(
        {
            "BTC": 0.30,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.20,
        }
    )

    previous = current.copy()

    turnover = calculate_turnover(
        current,
        previous,
    )

    assert turnover == pytest.approx(
        0.0
    )


def test_turnover_calculation():
    previous = pd.Series(
        {
            "BTC": 0.25,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.25,
        }
    )

    current = pd.Series(
        {
            "BTC": 0.35,
            "ETH": 0.20,
            "SOL": 0.25,
            "BNB": 0.20,
        }
    )

    turnover = calculate_turnover(
        current,
        previous,
    )

    expected = (
        0.5
        * (
            0.10
            + 0.05
            + 0.00
            + 0.05
        )
    )

    assert turnover == pytest.approx(
        expected
    )


def test_transaction_cost_from_basis_points():
    cost = calculate_transaction_cost(
        turnover=0.50,
        transaction_cost_bps=10.0,
    )

    assert cost == pytest.approx(
        0.0005
    )


def test_portfolio_return():
    weights = pd.Series(
        {
            "BTC": 0.50,
            "ETH": 0.50,
        }
    )

    returns = pd.Series(
        {
            "BTC": 0.10,
            "ETH": -0.02,
        }
    )

    result = calculate_portfolio_return(
        weights=weights,
        asset_returns=returns,
    )

    assert result == pytest.approx(
        0.04
    )


def test_portfolio_return_with_cash():
    weights = pd.Series(
        {
            "BTC": 0.30,
            "ETH": 0.30,
        }
    )

    returns = pd.Series(
        {
            "BTC": 0.10,
            "ETH": -0.02,
        }
    )

    result = calculate_portfolio_return(
        weights=weights,
        asset_returns=returns,
        cash_weight=0.40,
        cash_return=0.0,
    )

    assert result == pytest.approx(
        0.024
    )


def test_equity_curve():
    returns = pd.Series(
        [
            0.10,
            -0.10,
        ]
    )

    equity = build_equity_curve(
        returns
    )

    assert equity.iloc[
        0
    ] == pytest.approx(
        1.10
    )

    assert equity.iloc[
        1
    ] == pytest.approx(
        0.99
    )


def test_max_drawdown():
    equity = pd.Series(
        [
            1.0,
            1.2,
            0.9,
            1.1,
        ]
    )

    drawdown = (
        calculate_max_drawdown(
            equity
        )
    )

    expected = (
        0.9
        / 1.2
        - 1.0
    )

    assert drawdown == pytest.approx(
        expected
    )