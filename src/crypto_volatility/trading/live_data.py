from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

BINANCE_BASE_URL = "https://data-api.binance.vision"

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

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def fetch_recent_klines(
    symbol: str,
    *,
    interval: str = "1d",
    limit: int = 120,
    base_url: str = BINANCE_BASE_URL,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """
    Fetch recent Binance Spot klines.

    The returned frame still includes the current open candle
    if Binance returns it. Use ``filter_closed_candles`` before
    feeding data into the feature or inference pipeline.
    """
    if not symbol:
        raise ValueError(
            "symbol cannot be empty"
        )

    if limit < 1 or limit > 1000:
        raise ValueError(
            "limit must be between 1 and 1000"
        )

    response = requests.get(
        f"{base_url}/api/v3/klines",
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    payload: list[
        list[Any]
    ] = response.json()

    if not payload:
        raise ValueError(
            f"No Binance klines returned for {symbol}"
        )

    frame = pd.DataFrame(
        payload,
        columns=KLINE_COLUMNS,
    )

    frame[
        "open_time"
    ] = pd.to_datetime(
        frame[
            "open_time"
        ],
        unit="ms",
        utc=True,
    )

    frame[
        "close_time"
    ] = pd.to_datetime(
        frame[
            "close_time"
        ],
        unit="ms",
        utc=True,
    )

    for column in NUMERIC_COLUMNS:
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="raise",
        )

    frame[
        "number_of_trades"
    ] = pd.to_numeric(
        frame[
            "number_of_trades"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    frame = (
        frame
        .drop(
            columns=[
                "ignore",
            ]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=[
                "open_time",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return frame


def filter_closed_candles(
    candles: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """
    Remove any candle whose close time has not yet passed.

    Only fully closed candles are allowed into the production
    feature and inference pipeline.
    """
    required = {
        "open_time",
        "close_time",
    }

    missing = (
        required
        - set(
            candles.columns
        )
    )

    if missing:
        raise ValueError(
            "Candles missing columns: "
            f"{sorted(missing)}"
        )

    current_time = (
        now
        if now is not None
        else datetime.now(
            UTC
        )
    )

    current_timestamp = pd.Timestamp(
        current_time
    )

    if current_timestamp.tzinfo is None:
        current_timestamp = (
            current_timestamp.tz_localize(
                "UTC"
            )
        )
    else:
        current_timestamp = (
            current_timestamp.tz_convert(
                "UTC"
            )
        )

    closed = candles.loc[
        candles[
            "close_time"
        ]
        < current_timestamp
    ].copy()

    return (
        closed
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )


def fetch_recent_closed_klines(
    symbol: str,
    *,
    interval: str = "1d",
    limit: int = 120,
    base_url: str = BINANCE_BASE_URL,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """
    Fetch recent Binance klines and retain only closed candles.
    """
    candles = fetch_recent_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
        base_url=base_url,
        timeout=timeout,
    )

    closed = filter_closed_candles(
        candles
    )

    if closed.empty:
        raise ValueError(
            f"No closed candles available for {symbol}"
        )

    return closed