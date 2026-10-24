import pandas as pd
import pytest

from crypto_volatility.portfolio.risk import (
    apply_max_weight_cap,
    apply_total_exposure,
    build_forecast_covariance,
    calculate_cash_weight,
    calculate_portfolio_volatility,
    calculate_volatility_target_exposure,
)


def test_weight_cap_preserves_sum_to_one():
    weights = pd.Series(
        {
            "BTC": 0.50,
            "ETH": 0.20,
            "SOL": 0.15,
            "BNB": 0.15,
        }
    )

    capped = apply_max_weight_cap(
        weights,
        max_weight=0.40,
    )

    assert capped.sum() == pytest.approx(
        1.0
    )


def test_weight_cap_limits_largest_asset():
    weights = pd.Series(
        {
            "BTC": 0.60,
            "ETH": 0.15,
            "SOL": 0.15,
            "BNB": 0.10,
        }
    )

    capped = apply_max_weight_cap(
        weights,
        max_weight=0.40,
    )

    assert (
        capped
        <= 0.40
        + 1e-12
    ).all()

    assert capped[
        "BTC"
    ] == pytest.approx(
        0.40
    )


def test_weight_cap_redistributes_excess():
    weights = pd.Series(
        {
            "BTC": 0.60,
            "ETH": 0.20,
            "SOL": 0.10,
            "BNB": 0.10,
        }
    )

    capped = apply_max_weight_cap(
        weights,
        max_weight=0.40,
    )

    assert capped[
        "BTC"
    ] == pytest.approx(
        0.40
    )

    assert capped[
        "ETH"
    ] > 0.20

    assert capped[
        "SOL"
    ] > 0.10

    assert capped[
        "BNB"
    ] > 0.10


def test_weights_below_cap_remain_valid():
    weights = pd.Series(
        {
            "BTC": 0.30,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.20,
        }
    )

    capped = apply_max_weight_cap(
        weights,
        max_weight=0.40,
    )

    assert capped.sum() == pytest.approx(
        1.0
    )

    assert capped[
        "BTC"
    ] == pytest.approx(
        0.30
    )


def test_equal_weights_with_valid_cap():
    weights = pd.Series(
        {
            "BTC": 0.25,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.25,
        }
    )

    capped = apply_max_weight_cap(
        weights,
        max_weight=0.40,
    )

    for value in capped:
        assert value == pytest.approx(
            0.25
        )


def test_rejects_impossible_weight_cap():
    weights = pd.Series(
        {
            "BTC": 0.25,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.25,
        }
    )

    with pytest.raises(
        ValueError,
        match="too small",
    ):
        apply_max_weight_cap(
            weights,
            max_weight=0.20,
        )


def test_rejects_negative_weights():
    weights = pd.Series(
        {
            "BTC": 0.50,
            "ETH": -0.10,
            "SOL": 0.30,
            "BNB": 0.30,
        }
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        apply_max_weight_cap(
            weights,
            max_weight=0.40,
        )


def test_rejects_invalid_max_weight():
    weights = pd.Series(
        {
            "BTC": 0.25,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.25,
        }
    )

    with pytest.raises(
        ValueError,
        match="max_weight",
    ):
        apply_max_weight_cap(
            weights,
            max_weight=0.0,
        )

def test_volatility_target_reduces_exposure():
    exposure = (
        calculate_volatility_target_exposure(
            forecast_portfolio_volatility=0.06,
            target_volatility=0.03,
        )
    )

    assert exposure == pytest.approx(
        0.50
    )


def test_volatility_target_does_not_create_leverage():
    exposure = (
        calculate_volatility_target_exposure(
            forecast_portfolio_volatility=0.02,
            target_volatility=0.03,
        )
    )

    assert exposure == pytest.approx(
        1.0
    )


def test_volatility_target_respects_max_exposure():
    exposure = (
        calculate_volatility_target_exposure(
            forecast_portfolio_volatility=0.02,
            target_volatility=0.03,
            max_exposure=0.80,
        )
    )

    assert exposure == pytest.approx(
        0.80
    )


def test_apply_total_exposure():
    weights = pd.Series(
        {
            "BTC": 0.30,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.20,
        }
    )

    scaled = apply_total_exposure(
        weights,
        exposure=0.60,
    )

    assert scaled.sum() == pytest.approx(
        0.60
    )

    assert scaled[
        "BTC"
    ] == pytest.approx(
        0.18
    )


def test_cash_weight():
    weights = pd.Series(
        {
            "BTC": 0.18,
            "ETH": 0.15,
            "SOL": 0.15,
            "BNB": 0.12,
        }
    )

    cash = calculate_cash_weight(
        weights
    )

    assert cash == pytest.approx(
        0.40
    )


def test_full_exposure_has_zero_cash():
    weights = pd.Series(
        {
            "BTC": 0.25,
            "ETH": 0.25,
            "SOL": 0.25,
            "BNB": 0.25,
        }
    )

    cash = calculate_cash_weight(
        weights
    )

    assert cash == pytest.approx(
        0.0
    )


def test_rejects_non_positive_forecast_volatility():
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_volatility_target_exposure(
            forecast_portfolio_volatility=0.0,
            target_volatility=0.03,
        )


def test_rejects_weights_not_summing_to_one():
    weights = pd.Series(
        {
            "BTC": 0.20,
            "ETH": 0.20,
            "SOL": 0.20,
            "BNB": 0.20,
        }
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        apply_total_exposure(
            weights,
            exposure=0.50,
        )

def test_forecast_covariance_has_expected_shape():
    forecast_volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.05,
            "SOL": 0.07,
            "BNB": 0.06,
        }
    )

    historical_returns = pd.DataFrame(
        {
            "BTC": [
                0.01,
                -0.02,
                0.03,
                0.01,
            ],
            "ETH": [
                0.02,
                -0.01,
                0.02,
                0.00,
            ],
            "SOL": [
                0.03,
                -0.03,
                0.04,
                0.02,
            ],
            "BNB": [
                0.01,
                -0.01,
                0.01,
                0.02,
            ],
        }
    )

    covariance = build_forecast_covariance(
        forecast_volatility=forecast_volatility,
        historical_returns=historical_returns,
        lookback=4,
    )

    assert covariance.shape == (
        4,
        4,
    )

    assert list(
        covariance.index
    ) == [
        "BTC",
        "ETH",
        "SOL",
        "BNB",
    ]


def test_covariance_diagonal_matches_forecast_variance():
    forecast_volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.08,
        }
    )

    historical_returns = pd.DataFrame(
        {
            "BTC": [
                0.01,
                -0.02,
                0.03,
                -0.01,
            ],
            "ETH": [
                0.02,
                -0.03,
                0.01,
                0.04,
            ],
        }
    )

    covariance = build_forecast_covariance(
        forecast_volatility=forecast_volatility,
        historical_returns=historical_returns,
        lookback=4,
    )

    assert covariance.loc[
        "BTC",
        "BTC",
    ] == pytest.approx(
        0.04**2
    )

    assert covariance.loc[
        "ETH",
        "ETH",
    ] == pytest.approx(
        0.08**2
    )


def test_portfolio_volatility_for_uncorrelated_assets():
    weights = pd.Series(
        {
            "BTC": 0.50,
            "ETH": 0.50,
        }
    )

    covariance = pd.DataFrame(
        [
            [
                0.04**2,
                0.0,
            ],
            [
                0.0,
                0.06**2,
            ],
        ],
        index=[
            "BTC",
            "ETH",
        ],
        columns=[
            "BTC",
            "ETH",
        ],
    )

    volatility = calculate_portfolio_volatility(
        weights=weights,
        covariance=covariance,
    )

    expected = (
        (
            0.50**2
            * 0.04**2
        )
        + (
            0.50**2
            * 0.06**2
        )
    ) ** 0.5

    assert volatility == pytest.approx(
        expected
    )


def test_portfolio_volatility_increases_with_correlation():
    weights = pd.Series(
        {
            "BTC": 0.50,
            "ETH": 0.50,
        }
    )

    uncorrelated = pd.DataFrame(
        [
            [
                0.04**2,
                0.0,
            ],
            [
                0.0,
                0.04**2,
            ],
        ],
        index=[
            "BTC",
            "ETH",
        ],
        columns=[
            "BTC",
            "ETH",
        ],
    )

    correlated = pd.DataFrame(
        [
            [
                0.04**2,
                0.0012,
            ],
            [
                0.0012,
                0.04**2,
            ],
        ],
        index=[
            "BTC",
            "ETH",
        ],
        columns=[
            "BTC",
            "ETH",
        ],
    )

    uncorrelated_vol = (
        calculate_portfolio_volatility(
            weights,
            uncorrelated,
        )
    )

    correlated_vol = (
        calculate_portfolio_volatility(
            weights,
            correlated,
        )
    )

    assert (
        correlated_vol
        > uncorrelated_vol
    )


def test_forecast_covariance_uses_only_requested_lookback():
    forecast_volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.05,
        }
    )

    historical_returns = pd.DataFrame(
        {
            "BTC": [
                0.50,
                -0.50,
                0.01,
                0.02,
                0.03,
            ],
            "ETH": [
                -0.50,
                0.50,
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    covariance = build_forecast_covariance(
        forecast_volatility=forecast_volatility,
        historical_returns=historical_returns,
        lookback=3,
    )

    assert covariance.loc[
        "BTC",
        "ETH",
    ] > 0


def test_forecast_covariance_rejects_missing_asset():
    forecast_volatility = pd.Series(
        {
            "BTC": 0.04,
            "ETH": 0.05,
        }
    )

    historical_returns = pd.DataFrame(
        {
            "BTC": [
                0.01,
                0.02,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing historical return columns",
    ):
        build_forecast_covariance(
            forecast_volatility=forecast_volatility,
            historical_returns=historical_returns,
        )