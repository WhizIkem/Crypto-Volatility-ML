from __future__ import annotations

from crypto_volatility.models.production import (
    ASSETS,
    FEATURE_COLUMNS,
)
from crypto_volatility.trading.live_data import (
    fetch_recent_closed_klines,
)
from crypto_volatility.trading.live_features import (
    build_live_feature_datasets,
    build_live_return_history,
    build_live_var_history,
    get_latest_valid_feature_history,
    get_latest_valid_feature_row,
)
from crypto_volatility.utils.config import (
    load_config,
)


def main() -> None:
    print(
        "\nLive feature-generation smoke test"
    )

    config = load_config()

    data_config = config[
        "data"
    ]

    interval = data_config[
        "interval"
    ]

    live_candles = {}

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
        print(
            f"Fetching {asset}..."
        )

        live_candles[
            asset
        ] = (
            fetch_recent_closed_klines(
                symbol=symbol,
                interval=interval,
                limit=120,
            )
        )

    feature_datasets = (
        build_live_feature_datasets(
            live_candles
        )
    )

    print(
        "\nLatest production feature rows:"
    )

    latest_timestamps = []

    for asset in ASSETS:
        latest = (
            get_latest_valid_feature_row(
                feature_datasets[
                    asset
                ]
            )
        )

        latest_timestamps.append(
            latest[
                "timestamp"
            ]
        )

        print(
            f"\n{asset}"
        )

        print(
            "Timestamp:",
            latest[
                "timestamp"
            ],
        )

        print(
            latest[
                FEATURE_COLUMNS
            ].to_string()
        )

    if len(
        set(
            latest_timestamps
        )
    ) != 1:
        raise ValueError(
            "Latest valid feature rows are not aligned"
        )

    eth_history = (
        get_latest_valid_feature_history(
            feature_datasets[
                "ETH"
            ],
            rows=7,
        )
    )

    var_history = (
        build_live_var_history(
            feature_datasets,
            rows=30,
        )
    )

    return_history = (
        build_live_return_history(
            feature_datasets,
            rows=30,
        )
    )

    print(
        "\nETH LSTM history rows:",
        len(
            eth_history
        ),
    )

    print(
        "VAR history rows:",
        len(
            var_history
        ),
    )

    print(
        "Portfolio return-history rows:",
        len(
            return_history
        ),
    )

    print(
        "\nLatest feature timestamp:",
        latest_timestamps[
            0
        ],
    )

    print(
        "\nLive feature generation succeeded."
    )


if __name__ == "__main__":
    main()