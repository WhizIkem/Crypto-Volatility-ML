import numpy as np
import pandas as pd
import pytest

from crypto_volatility.analysis.volatility import (
    add_garman_klass_volatility,
    add_log_garman_klass_volatility,
    add_log_returns,
    add_rolling_volatility,
)


def test_add_log_returns():
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                110.0,
                121.0,
            ]
        }
    )

    result = add_log_returns(df)

    expected = np.log(1.1)

    assert pd.isna(
        result.loc[0, "log_return"]
    )

    assert result.loc[
        1,
        "log_return",
    ] == pytest.approx(expected)

    assert result.loc[
        2,
        "log_return",
    ] == pytest.approx(expected)


def test_add_log_returns_does_not_modify_input():
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                110.0,
            ]
        }
    )

    add_log_returns(df)

    assert "log_return" not in df.columns


def test_add_log_returns_rejects_non_positive_prices():
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                0.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="positive values",
    ):
        add_log_returns(df)


def test_add_rolling_volatility():
    df = pd.DataFrame(
        {
            "log_return": [
                0.01,
                0.02,
                -0.01,
                0.03,
            ]
        }
    )

    result = add_rolling_volatility(
        df,
        window=3,
        annualization_factor=365,
    )

    expected_daily = (
        df["log_return"]
        .rolling(3)
        .std()
    )

    expected_annualized = (
        expected_daily
        * np.sqrt(365)
    )

    pd.testing.assert_series_equal(
        result["rolling_vol_3d"],
        expected_daily,
        check_names=False,
    )

    pd.testing.assert_series_equal(
        result["rolling_vol_3d_annualized"],
        expected_annualized,
        check_names=False,
    )


def test_add_garman_klass_volatility():
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        }
    )

    result = add_garman_klass_volatility(
        df,
        annualization_factor=365,
    )

    log_hl = np.log(110.0 / 90.0)
    log_co = np.log(105.0 / 100.0)

    expected_variance = (
        0.5 * log_hl**2
        - (2 * np.log(2) - 1)
        * log_co**2
    )

    expected_volatility = np.sqrt(
        expected_variance
    )

    expected_annualized = (
        expected_volatility
        * np.sqrt(365)
    )

    assert result.loc[
        0,
        "gk_variance",
    ] == pytest.approx(
        expected_variance
    )

    assert result.loc[
        0,
        "gk_volatility",
    ] == pytest.approx(
        expected_volatility
    )

    assert result.loc[
        0,
        "gk_volatility_annualized",
    ] == pytest.approx(
        expected_annualized
    )


def test_garman_klass_rejects_missing_ohlc_columns():
    df = pd.DataFrame(
        {
            "open": [100.0],
            "close": [105.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required OHLC columns",
    ):
        add_garman_klass_volatility(df)


def test_add_log_garman_klass_volatility():
    df = pd.DataFrame(
        {
            "gk_volatility": [
                0.10,
                0.20,
            ]
        }
    )

    result = (
        add_log_garman_klass_volatility(df)
    )

    assert result.loc[
        0,
        "log_gk_volatility",
    ] == pytest.approx(
        np.log(0.10)
    )

    assert result.loc[
        1,
        "log_gk_volatility",
    ] == pytest.approx(
        np.log(0.20)
    )


def test_log_garman_klass_handles_zero():
    df = pd.DataFrame(
        {
            "gk_volatility": [
                0.0,
            ]
        }
    )

    result = (
        add_log_garman_klass_volatility(
            df,
            epsilon=1e-8,
        )
    )

    assert result.loc[
        0,
        "log_gk_volatility",
    ] == pytest.approx(
        np.log(1e-8)
    )