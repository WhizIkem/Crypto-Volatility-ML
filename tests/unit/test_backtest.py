import numpy as np
import pandas as pd
import pytest

from crypto_volatility.portfolio.backtest import (
    build_equity_curve,
    calculate_calmar_ratio,
    calculate_drawdown_series,
    calculate_expected_shortfall,
    calculate_historical_var,
    calculate_hit_rate,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_portfolio_return,
    calculate_sortino_ratio,
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


def test_drawdown_series():
    equity = pd.Series(
        [
            1.0,
            1.2,
            0.9,
            1.1,
            1.3,
        ]
    )

    drawdown = (
        calculate_drawdown_series(
            equity
        )
    )

    assert drawdown.iloc[
        0
    ] == pytest.approx(
        0.0
    )

    assert drawdown.iloc[
        2
    ] == pytest.approx(
        0.9
        / 1.2
        - 1.0
    )

    assert drawdown.iloc[
        4
    ] == pytest.approx(
        0.0
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


def test_max_drawdown_duration():
    equity = pd.Series(
        [
            1.0,
            1.2,
            1.1,
            1.0,
            1.15,
            1.25,
            1.20,
        ]
    )

    duration = (
        calculate_max_drawdown_duration(
            equity
        )
    )

    assert duration == 3


def test_hit_rate():
    returns = pd.Series(
        [
            0.01,
            -0.02,
            0.03,
            0.00,
        ]
    )

    hit_rate = calculate_hit_rate(
        returns
    )

    assert hit_rate == pytest.approx(
        0.50
    )


def test_sortino_ratio():
    returns = pd.Series(
        [
            0.02,
            -0.01,
            0.03,
            -0.02,
        ]
    )

    result = calculate_sortino_ratio(
        returns,
        annualization_factor=1,
    )

    downside = np.array(
        [
            0.0,
            -0.01,
            0.0,
            -0.02,
        ]
    )

    expected_downside_deviation = (
        np.sqrt(
            np.mean(
                downside**2
            )
        )
    )

    expected = (
        returns.mean()
        / expected_downside_deviation
    )

    assert result == pytest.approx(
        expected
    )


def test_sortino_returns_nan_without_downside():
    returns = pd.Series(
        [
            0.01,
            0.02,
            0.03,
        ]
    )

    result = calculate_sortino_ratio(
        returns
    )

    assert np.isnan(
        result
    )


def test_calmar_ratio():
    result = calculate_calmar_ratio(
        annualized_return=0.20,
        max_drawdown=-0.10,
    )

    assert result == pytest.approx(
        2.0
    )


def test_calmar_returns_nan_without_drawdown():
    result = calculate_calmar_ratio(
        annualized_return=0.20,
        max_drawdown=0.0,
    )

    assert np.isnan(
        result
    )


def test_historical_var():
    returns = pd.Series(
        [
            -0.10,
            -0.05,
            -0.02,
            0.01,
            0.02,
            0.03,
        ]
    )

    result = calculate_historical_var(
        returns,
        confidence_level=0.50,
    )

    expected_quantile = np.quantile(
        returns,
        0.50,
    )

    assert result == pytest.approx(
        max(
            -expected_quantile,
            0.0,
        )
    )


def test_expected_shortfall():
    returns = pd.Series(
        [
            -0.10,
            -0.05,
            -0.02,
            0.01,
            0.02,
            0.03,
        ]
    )

    confidence_level = 0.80

    quantile = np.quantile(
        returns,
        1.0
        - confidence_level,
    )

    tail = returns[
        returns
        <= quantile
    ]

    expected = max(
        -float(
            tail.mean()
        ),
        0.0,
    )

    result = calculate_expected_shortfall(
        returns,
        confidence_level=confidence_level,
    )

    assert result == pytest.approx(
        expected
    )


def test_expected_shortfall_is_at_least_var():
    returns = pd.Series(
        [
            -0.10,
            -0.08,
            -0.04,
            -0.01,
            0.02,
            0.03,
            0.04,
        ]
    )

    var = calculate_historical_var(
        returns,
        confidence_level=0.80,
    )

    expected_shortfall = (
        calculate_expected_shortfall(
            returns,
            confidence_level=0.80,
        )
    )

    assert (
        expected_shortfall
        >= var
    )


def test_invalid_var_confidence_level():
    returns = pd.Series(
        [
            0.01,
            -0.01,
        ]
    )

    with pytest.raises(
        ValueError,
        match="confidence_level",
    ):
        calculate_historical_var(
            returns,
            confidence_level=1.0,
        )