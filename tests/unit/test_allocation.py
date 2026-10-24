import numpy as np
import pandas as pd
import pytest

from crypto_volatility.portfolio.allocation import (
    build_portfolio_allocation,
    inverse_volatility_weights,
    log_volatility_to_volatility,
    weights_from_log_volatility_forecasts,
)


def test_log_volatility_to_volatility():
    log_volatility = pd.Series(
        {
            "BTC": np.log(
                0.04
            ),
            "ETH": np.log(
                0.08
            ),
        }
    )

    volatility = (
        log_volatility_to_volatility(
            log_volatility
        )
    )

    assert volatility[
        "BTC"
    ] == pytest.approx(
        0.04
    )

    assert volatility[
        "ETH"
    ] == pytest.approx(
        0.08
    )


def test_inverse_volatility_weights_sum_to_one():
    volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.06,
            "SOL": 0.08,
            "BNB": 0.05,
        }
    )

    weights = (
        inverse_volatility_weights(
            volatility
        )
    )

    assert weights.sum() == pytest.approx(
        1.0
    )


def test_lower_volatility_gets_larger_weight():
    volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.08,
        }
    )

    weights = (
        inverse_volatility_weights(
            volatility
        )
    )

    assert (
        weights["BTC"]
        > weights["ETH"]
    )


def test_equal_volatility_produces_equal_weights():
    volatility = pd.Series(
        {
            "BTC": 0.05,
            "ETH": 0.05,
            "SOL": 0.05,
            "BNB": 0.05,
        }
    )

    weights = (
        inverse_volatility_weights(
            volatility
        )
    )

    for weight in weights:
        assert weight == pytest.approx(
            0.25
        )


def test_weights_from_log_forecasts():
    forecasts = pd.Series(
        {
            "BTC": np.log(
                0.04
            ),
            "ETH": np.log(
                0.08
            ),
        }
    )

    weights = (
        weights_from_log_volatility_forecasts(
            forecasts
        )
    )

    assert weights.sum() == pytest.approx(
        1.0
    )

    assert (
        weights["BTC"]
        > weights["ETH"]
    )


def test_inverse_volatility_rejects_non_positive_values():
    volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.0,
        }
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        inverse_volatility_weights(
            volatility
        )


def test_log_volatility_rejects_non_finite_values():
    forecasts = pd.Series(
        {
            "BTC": -3.0,
            "ETH": np.nan,
        }
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        log_volatility_to_volatility(
            forecasts
        )

def make_historical_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [
                0.01,
                -0.02,
                0.015,
                0.005,
                -0.01,
                0.02,
            ],
            "ETH": [
                0.015,
                -0.018,
                0.012,
                0.003,
                -0.012,
                0.025,
            ],
            "SOL": [
                0.025,
                -0.03,
                0.02,
                0.01,
                -0.02,
                0.035,
            ],
            "BNB": [
                0.008,
                -0.01,
                0.009,
                0.004,
                -0.008,
                0.015,
            ],
        }
    )


def test_build_portfolio_allocation_weights_plus_cash_sum_to_one():
    forecasts = pd.Series(
        {
            "BTC": np.log(
                0.04
            ),
            "ETH": np.log(
                0.05
            ),
            "SOL": np.log(
                0.07
            ),
            "BNB": np.log(
                0.06
            ),
        }
    )

    allocation = build_portfolio_allocation(
        log_volatility_forecasts=forecasts,
        historical_returns=make_historical_returns(),
        max_asset_weight=0.40,
        target_portfolio_volatility=0.03,
        correlation_lookback=6,
    )

    total = (
        allocation.risky_weights.sum()
        + allocation.cash_weight
    )

    assert total == pytest.approx(
        1.0
    )


def test_build_portfolio_allocation_respects_asset_cap():
    forecasts = pd.Series(
        {
            "BTC": np.log(
                0.01
            ),
            "ETH": np.log(
                0.10
            ),
            "SOL": np.log(
                0.12
            ),
            "BNB": np.log(
                0.11
            ),
        }
    )

    allocation = build_portfolio_allocation(
        log_volatility_forecasts=forecasts,
        historical_returns=make_historical_returns(),
        max_asset_weight=0.40,
        target_portfolio_volatility=1.0,
        correlation_lookback=6,
    )

    assert (
        allocation.capped_risky_weights
        <= 0.40
        + 1e-12
    ).all()


def test_build_portfolio_allocation_can_hold_cash():
    forecasts = pd.Series(
        {
            "BTC": np.log(
                0.10
            ),
            "ETH": np.log(
                0.12
            ),
            "SOL": np.log(
                0.15
            ),
            "BNB": np.log(
                0.11
            ),
        }
    )

    allocation = build_portfolio_allocation(
        log_volatility_forecasts=forecasts,
        historical_returns=make_historical_returns(),
        max_asset_weight=0.40,
        target_portfolio_volatility=0.02,
        correlation_lookback=6,
    )

    assert allocation.cash_weight > 0.0

    assert (
        allocation.risky_exposure
        < 1.0
    )


def test_build_portfolio_allocation_has_full_exposure_when_risk_is_low():
    forecasts = pd.Series(
        {
            "BTC": np.log(
                0.01
            ),
            "ETH": np.log(
                0.012
            ),
            "SOL": np.log(
                0.015
            ),
            "BNB": np.log(
                0.011
            ),
        }
    )

    allocation = build_portfolio_allocation(
        log_volatility_forecasts=forecasts,
        historical_returns=make_historical_returns(),
        max_asset_weight=0.40,
        target_portfolio_volatility=0.10,
        correlation_lookback=6,
    )

    assert allocation.risky_exposure == pytest.approx(
        1.0
    )

    assert allocation.cash_weight == pytest.approx(
        0.0
    )


def test_final_risky_weights_do_not_exceed_capped_weights():
    forecasts = pd.Series(
        {
            "BTC": np.log(
                0.05
            ),
            "ETH": np.log(
                0.06
            ),
            "SOL": np.log(
                0.08
            ),
            "BNB": np.log(
                0.07
            ),
        }
    )

    allocation = build_portfolio_allocation(
        log_volatility_forecasts=forecasts,
        historical_returns=make_historical_returns(),
        max_asset_weight=0.40,
        target_portfolio_volatility=0.02,
        correlation_lookback=6,
    )

    assert (
        allocation.risky_weights
        <= allocation.capped_risky_weights
        + 1e-12
    ).all()