import pandas as pd

from crypto_volatility.data.binance import (
    get_historical_klines,
    normalize_klines,
)
from crypto_volatility.utils.config import load_config


def main() -> None:
    config = load_config()

    data_config = config["data"]

    symbols = data_config["symbols"]
    interval = data_config["interval"]
    start_date = data_config["start_date"]
    end_date = data_config.get("end_date")

    for asset, symbol in symbols.items():
        print(f"Downloading data for {asset} ({symbol}...")

        raw_data = get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
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

        print(
            f"{asset}:"
            f"{len(cleaned_data)} rows | "
            f"{cleaned_data['timestamp'].min()} → "
            f"{cleaned_data['timestamp'].max()}"
        )


if __name__ == "__main__":
    main()