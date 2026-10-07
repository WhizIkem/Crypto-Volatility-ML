import pandas as pd

REQUIRED_COLUMNS = {
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
}


# Validate canonical OHLCV candlestick data for completeness, consistency, and correctness.
def validate_candles(df: pd.DataFrame) -> None:
    """
    Validate canonical OHLCV candlestick data.

    Raises
    ------
    ValueError
        If one or more validation checks fail.
    """
    errors: list[str] = []

    if df.empty:
        errors.append("DataFrame is empty")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if errors:
        raise ValueError("; ".join(errors))

    if df["timestamp"].isna().any():
        errors.append("timestamps contain missing values")

    if df["timestamp"].duplicated().any():
        errors.append("timestamps contain duplicate values")

    if not df["timestamp"].is_monotonic_increasing:
        errors.append("timestamps are not sorted chronologically")

    numeric_columns = [
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

    if df[numeric_columns].isna().any().any():
        errors.append("numeric columns contain missing values")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("OHLC prices must be positive")

    if (df["volume"] < 0).any():
        errors.append("volume contains negative values")

    if (df["quote_volume"] < 0).any():
        errors.append("quote_volume contains negative values")

    if (df["number_of_trades"] < 0).any():
        errors.append("number_of_trades contains negative values")

    if (df["taker_buy_base_volume"] < 0).any():
        errors.append("taker_buy_base_volume contains negative values")

    if (df["taker_buy_quote_volume"] < 0).any():
        errors.append("taker_buy_quote_volume contains negative values")

    if (df["high"] < df["low"]).any():
        errors.append("high price is less than low price")

    if (df["high"] < df["open"]).any():
        errors.append("high price is less than open price")

    if (df["high"] < df["close"]).any():
        errors.append("high price is less than close price")

    if (df["low"] > df["open"]).any():
        errors.append("low price is greater than open price")

    if (df["low"] > df["close"]).any():
        errors.append("low price is greater than close price")

    if errors:
        raise ValueError("; ".join(errors))


#
def find_missing_timestamps(
    df: pd.DataFrame,
    interval: str
) -> pd.DatetimeIndex:
    """
    Find missing timestamps in a candle time series.

    Parameters
    ----------
    df :
        Canonical candle DataFrame.

    interval :
        Candle interval, currently supported "1d".
    
    Returns
    -------
    pd.DatetimeIndex
        Missing timestamps between the first and last observation.
    """ 

    if df.empty:
        return pd.DatetimeIndex([])

    if interval != "1d":
        raise ValueError(
            "Missing timestamp checks currently support only 1d candles"
        )

    expected = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="1D",
        tz="UTC"
    )

    observed = pd.DatetimeIndex(df["timestamp"])

    return expected.difference(observed)

    
            