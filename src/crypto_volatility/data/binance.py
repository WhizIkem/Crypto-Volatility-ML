from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision"
KLINES_ENDPOINT = "/api/v3/klines"


# Columns for the Binance kline (candlestick) data
KLINE_COLUMNS = [
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
]

# Canonical columns for the Binance kline (candlestick) data
CANONICAL_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


# Convert datetime or ISO date string to Unix milliseconds
def datetime_to_milliseconds(value: str | datetime) -> int:
    """
    Convert a UTC date/datetime into Unix milliseconds.
    
    Parameters
    ----------
    value :
        ISO date string such as "2021-01-01" or a datetime object.
    Returns
    -------
    int
        Unix timestamp in milliseconds.
    """
    if isinstance(value, str):
        dt = pd.Timestamp(value, tz="UTC")
    else:
        dt = pd.Timestamp(value)

        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")


    return int(dt.timestamp() * 1000)


# Fetch raw Binance kline (candlestick) data
def get_klines(
    symbol: str,
    interval: str = "1d",
    limit: int = 5,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """
    Fetch candlestick data from  the Binance public market-data API.

    Parameters
    ----------
    symbol :
        Binance trading pair, for example "BTCUSDT".
    interval :
        Candlestick interval, for example 1d, 4h, or 1h.
    limit :
        Number of candlesticks to request.

    Returns
    -------
    pd.DataFrame
        Raw Binance kline data as a DataFrame.
    """

    if not 1 <= limit <= 1000:
        raise ValueError("Limit must be between 1 and 1000.")

    url = f"{BASE_URL}{KLINES_ENDPOINT}"

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }

    if start_time is not None:
        params["startTime"] = start_time

    if end_time is not None:
        params["endTime"] = end_time

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(
        data,
        columns=KLINE_COLUMNS,
    )


# Normalize raw Binance kline (candlestick) data
def normalize_klines(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Normalize raw Binance kline data to a canonical format.

    Parameters
    ----------
    df :
        Raw Binance kline DataFrame.
    symbol :
        Trading pair symbol, for example "BTCUSDT".

    Returns
    -------
    pd.DataFrame
        Normalized kline data as a DataFrame with canonical columns.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["open_time"], 
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    df["number_of_trades"] = pd.to_numeric(
        df["number_of_trades"],
        downcast="integer",
    )

    df["symbol"] = symbol.upper()

    return df[CANONICAL_COLUMNS]


def get_historical_klines(
    symbol: str,
    interval: str,
    start_date: str | datetime,
    end_date: str | datetime | None = None,
) -> pd.DataFrame:
    """
    Download historical candlestick data from the Binance public market-data API.

    Parameters
    ----------
    symbol :
        Binance trading pair, for example "BTCUSDT".
    interval :
        Candlestick interval, for example 1d, 4h, or 1h.
    start_date :
        First request date.
    end_date :
        Optional final request date. if omitted, use current UTC time.
    
    Returns
    -------
    pd.DataFrame
        Raw Binance kline data across the requested period.
    """

    start_time = datetime_to_milliseconds(start_date)

    if end_date is None:
        end_time = int(
            datetime.now(UTC).timestamp() * 1000
        )
    else:
        end_time = datetime_to_milliseconds(end_date)

    if start_time >= end_time:
        raise ValueError("start_date must be earlier than end_date.")

    batches = []

    current_start = start_time

    while current_start < end_time:
        batch = get_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time - 1,
            limit=1000,
        )

        if batch.empty:
            break

        batches.append(batch)

        last_open_time = int(batch.iloc[-1]["open_time"])

        next_start = last_open_time + 1

        if next_start <= current_start:
            raise RuntimeError(
                "Historical pagination did not advance."
            )
        
        current_start = next_start

        if len(batch) < 1000:
            break

    if not batches:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    result = pd.concat(
        batches,
        ignore_index=True,
    )

    result = (
        result
        .drop_duplicates(subset=["open_time"])
        .sort_values(by=["open_time"])
        .reset_index(drop=True)
    )

    return result