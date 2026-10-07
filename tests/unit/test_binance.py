import pandas as pd
import pytest

from crypto_volatility.data.binance import (
    get_historical_klines,
    get_klines,
    normalize_klines,
)


# Test cases for the Binance data module
def test_get_klines_rejects_invalid_limit():
    with pytest.raises(ValueError):
        get_klines(
            symbol="BTCUSDT",
            interval="1d",
            limit=1001,
        )


# Test cases for the normalization of raw Binance kline data
def test_normalize_klines():
    raw_data = pd.DataFrame(
        [
            [
                1609459200000,
                "29000.00",
                "29600.00",
                "28700.00",
                "29300.00",
                "100.5",
                1609545599999,
                "2940000.00",
                5000,
                "50.25",
                "1470000.00",
                "0",
            ]
        ],
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    clean = normalize_klines(raw_data, "BTCUSDT")

    assert clean.loc[0, "symbol"] == "BTCUSDT"
    assert clean.loc[0, "open"] == 29000.00
    assert clean.loc[0, "close"] == 29300.00
    assert clean.loc[0, "timestamp"].tzinfo is not None
    assert "ignore" not in clean.columns
    assert "close_time" not in clean.columns


#
def test_historical_klines_rejects_invalid_date_range():
    with pytest.raises(ValueError):
        get_historical_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_date="2024-02-01",
            end_date="2024-01-01",
        )