from pathlib import Path

import pandas as pd

from crypto_volatility.data.binance import (
    get_historical_klines,
    normalize_klines,
)
from crypto_volatility.data.validation import (
    find_missing_timestamps,
    validate_candles,
)
from crypto_volatility.utils.config import load_config


def main() -> None:
    config = load_config()

    data_config = config["data"]

    symbols = data_config["symbols"]
    interval = data_config["interval"]
    start_date = data_config["start_date"]
    end_date = data_config.get("end_date")
    raw_dir = Path(data_config["raw_dir"])
    interim_dir = Path(data_config["interim_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    for asset, symbol in symbols.items():
        print(f"Downloading data for {asset} ({symbol})...")

        raw_data = get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )

        raw_path = (
            raw_dir
            / f"{symbol.lower()}_{interval}_raw.parquet"
        )
        raw_data.to_parquet(
            raw_path,
            index=False,
        )

        cleaned_data = normalize_klines(
            raw_data,
            symbol=symbol,
        )
        if interval == "1d":
            today_utc = pd.Timestamp.now(tz="UTC").normalize()

            cleaned_data = cleaned_data[
                cleaned_data["timestamp"] < today_utc
            ].reset_index(drop=True)

        validate_candles(cleaned_data)

        missing_timestamps = find_missing_timestamps(
            cleaned_data,
            interval=interval,
        )

        if len(missing_timestamps) > 0:
            print(
                f"WARNING: {asset} has "
                f"{len(missing_timestamps)} missing candles"
            )

        interim_path = (
            interim_dir
            / f"{symbol.lower()}_{interval}_clean.parquet"
        )

        cleaned_data.to_parquet(
            interim_path,
            index=False,
        )

        print(
            f"{asset}:"
            f"{len(cleaned_data)} rows | "
            f"{cleaned_data['timestamp'].min()} → "
            f"{cleaned_data['timestamp'].max()}"
        )


if __name__ == "__main__":
    main()