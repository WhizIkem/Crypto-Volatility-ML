import pandas as pd
import pytest

from crypto_volatility.data.validation import (
    find_missing_timestamps,
    validate_candles,
)


# Test cases for the validation functions in crypto_volatility.data.validation
def test_validate_candles_rejects_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                ],
                utc=True
            ),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": [100.0, 101.0],
            "high": [110.0, 111.0],
            "low": [90.0, 91.0],
            "close": [105.0, 106.0],
            "volume": [10.0, 11.0],
            "quote_volume": [1000.0, 1100.0],
            "number_of_trades": [100, 110],
            "taker_buy_base_volume": [5.0, 5.5],
            "taker_buy_quote_volume": [500.0, 550.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="timestamps contain duplicate values"
    ):
        validate_candles(df)


# Test for finding missing daily timestamps
def test_find_missing_daily_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-03",
                ],
                utc=True
            )
        }
    )

    missing = find_missing_timestamps(
        df,
        interval="1d"
    )

    assert len(missing) == 1
    assert missing[0] == pd.Timestamp(
        "2024-01-02",
        tz="UTC",
    )