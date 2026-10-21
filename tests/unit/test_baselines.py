import pandas as pd
import pytest

from crypto_volatility.models.baselines import (
    historical_mean_forecast,
    persistence_forecast,
    rolling_mean_forecast,
)


def test_persistence_forecast_matches_current_volatility():
    volatility = pd.Series(
        [
            -3.0,
            -2.5,
            -2.0,
        ]
    )

    forecast = persistence_forecast(
        volatility
    )

    pd.testing.assert_series_equal(
        forecast,
        volatility,
    )


def test_persistence_forecast_does_not_modify_input():
    volatility = pd.Series(
        [
            -3.0,
            -2.5,
        ]
    )

    forecast = persistence_forecast(
        volatility
    )

    forecast.iloc[0] = 100.0

    assert volatility.iloc[0] == -3.0


def test_historical_mean_forecast_uses_training_mean():
    train_target = pd.Series(
        [
            -3.0,
            -2.0,
            -1.0,
        ]
    )

    forecast_index = pd.RangeIndex(
        start=0,
        stop=2,
    )

    forecast = historical_mean_forecast(
        train_target,
        forecast_index,
    )

    expected = pd.Series(
        [
            -2.0,
            -2.0,
        ],
        index=forecast_index,
    )

    pd.testing.assert_series_equal(
        forecast,
        expected,
    )


def test_historical_mean_forecast_rejects_empty_training_target():
    train_target = pd.Series(
        [],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        historical_mean_forecast(
            train_target,
            pd.RangeIndex(2),
        )


def test_rolling_mean_forecast():
    volatility = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    forecast = rolling_mean_forecast(
        volatility,
        window=2,
    )

    expected = pd.Series(
        [
            float("nan"),
            1.5,
            2.5,
            3.5,
        ]
    )

    pd.testing.assert_series_equal(
        forecast,
        expected,
    )


def test_rolling_mean_forecast_rejects_invalid_window():
    volatility = pd.Series(
        [
            1.0,
            2.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        rolling_mean_forecast(
            volatility,
            window=0,
        )