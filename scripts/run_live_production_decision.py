from __future__ import annotations

import pandas as pd

from crypto_volatility.models.production import (
    ASSETS,
    FEATURE_COLUMNS,
    load_production_models,
    predict_all_assets,
)
from crypto_volatility.trading.decision import (
    build_trading_decision,
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


def fetch_live_candles() -> dict[
    str,
    pd.DataFrame,
]:
    """
    Fetch recent closed Binance candles for all configured assets.
    """
    config = load_config()

    data_config = config[
        "data"
    ]

    interval = data_config[
        "interval"
    ]

    datasets = {}

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
        print(
            f"Fetching {asset} ({symbol})..."
        )

        datasets[
            asset
        ] = (
            fetch_recent_closed_klines(
                symbol=symbol,
                interval=interval,
                limit=120,
            )
        )

    return datasets


def build_current_weights() -> pd.Series:
    """
    Temporary portfolio state.

    For this first live production smoke test we assume the
    portfolio starts fully in cash.

    This will be replaced by persisted paper-portfolio state
    in the next phase.
    """
    return pd.Series(
        {
            "BTC": 0.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
    )


def main() -> None:
    print(
        "\nLive production portfolio decision"
    )

    print(
        "\nLoading saved production models..."
    )

    bundle = (
        load_production_models()
    )

    print(
        "Production models loaded."
    )

    print(
        "\nFetching live Binance data..."
    )

    live_candles = (
        fetch_live_candles()
    )

    print(
        "\nBuilding live features..."
    )

    feature_datasets = (
        build_live_feature_datasets(
            live_candles
        )
    )

    latest_rows = {
        asset: (
            get_latest_valid_feature_row(
                feature_datasets[
                    asset
                ]
            )
        )
        for asset in ASSETS
    }

    latest_timestamps = {
        asset: row[
            "timestamp"
        ]
        for asset, row in (
            latest_rows.items()
        )
    }

    if len(
        set(
            latest_timestamps.values()
        )
    ) != 1:
        raise ValueError(
            "Latest live feature timestamps are not aligned"
        )

    forecast_origin = next(
        iter(
            latest_timestamps.values()
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

    eth_feature_history = (
        get_latest_valid_feature_history(
            feature_datasets[
                "ETH"
            ],
            rows=7,
        )
    )

    bnb_latest_features = (
        latest_rows[
            "BNB"
        ][
            FEATURE_COLUMNS
        ]
        .copy()
    )

    print(
        "\nGenerating live volatility forecasts..."
    )

    forecasts = (
        predict_all_assets(
            bundle,
            volatility_history=(
                var_history
            ),
            eth_feature_history=(
                eth_feature_history
            ),
            bnb_latest_features=(
                bnb_latest_features
            ),
        )
    )

    current_weights = (
        build_current_weights()
    )

    decision = (
        build_trading_decision(
            forecasts=forecasts,
            historical_returns=(
                return_history
            ),
            current_weights=(
                current_weights
            ),
        )
    )

    print(
        "\nForecast origin:"
    )

    print(
        forecast_origin
    )

    print(
        "\nNext-day log-volatility forecasts:"
    )

    print(
        decision
        .forecast_log_volatility
        .to_string()
    )

    print(
        "\nCurrent risky weights:"
    )

    print(
        decision
        .current_weights
        .to_string()
    )

    print(
        "\nTarget risky weights:"
    )

    print(
        decision
        .target_weights
        .to_string()
    )

    print(
        "\nProposed trade weights:"
    )

    print(
        decision
        .trade_weights
        .to_string()
    )

    print(
        "\nPortfolio risk:"
    )

    print(
        "Forecast portfolio volatility:",
        f"{decision.forecast_portfolio_volatility:.6f}",
    )

    print(
        "Risky exposure:",
        f"{decision.risky_exposure:.6f}",
    )

    print(
        "Cash weight:",
        f"{decision.cash_weight:.6f}",
    )

    print(
        "\nExecution:"
    )

    print(
        "One-way turnover:",
        f"{decision.turnover:.6f}",
    )

    print(
        "Turnover threshold:",
        "0.050000",
    )

    print(
        "Should rebalance:",
        decision.should_rebalance,
    )

    print(
        "\nLive production decision succeeded."
    )


if __name__ == "__main__":
    main()