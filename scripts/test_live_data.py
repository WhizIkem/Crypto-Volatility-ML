from __future__ import annotations

from crypto_volatility.trading.live_data import (
    fetch_recent_closed_klines,
)
from crypto_volatility.utils.config import (
    load_config,
)


def main() -> None:
    print(
        "\nLive Binance closed-candle smoke test"
    )

    config = load_config()

    data_config = config[
        "data"
    ]

    symbols = data_config[
        "symbols"
    ]

    interval = data_config[
        "interval"
    ]

    results = {}

    for asset, symbol in (
        symbols.items()
    ):
        print(
            f"\nFetching {asset} ({symbol})..."
        )

        candles = (
            fetch_recent_closed_klines(
                symbol=symbol,
                interval=interval,
                limit=120,
            )
        )

        results[
            asset
        ] = candles

        latest = candles.iloc[
            -1
        ]

        print(
            "Rows:",
            len(
                candles
            ),
        )

        print(
            "Latest open time:",
            latest[
                "open_time"
            ],
        )

        print(
            "Latest close time:",
            latest[
                "close_time"
            ],
        )

        print(
            "Latest close price:",
            f"{latest['close']:.6f}",
        )

    latest_dates = {
        asset: candles[
            "open_time"
        ].iloc[
            -1
        ]
        for asset, candles in (
            results.items()
        )
    }

    if len(
        set(
            latest_dates.values()
        )
    ) != 1:
        raise ValueError(
            "Assets do not share the same latest closed candle"
        )

    print(
        "\nAll assets aligned on latest closed candle:"
    )

    print(
        next(
            iter(
                latest_dates.values()
            )
        )
    )

    print(
        "\nLive Binance data ingestion succeeded."
    )


if __name__ == "__main__":
    main()