from __future__ import annotations

from pathlib import Path

import numpy as np
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
from crypto_volatility.utils.config import load_config


def load_datasets() -> dict[
    str,
    pd.DataFrame,
]:
    config = load_config()

    data_config = config[
        "data"
    ]

    processed_dir = Path(
        data_config[
            "processed_dir"
        ]
    )

    interval = data_config[
        "interval"
    ]

    datasets = {}

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
        path = (
            processed_dir
            / f"{symbol.lower()}_{interval}_model.parquet"
        )

        datasets[
            asset
        ] = (
            pd.read_parquet(
                path
            )
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

    return datasets


def build_var_history(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    frames = []

    for asset in ASSETS:
        frame = (
            datasets[
                asset
            ][
                [
                    "timestamp",
                    "log_gk_volatility",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_gk_volatility": asset,
                }
            )
        )

        frames.append(
            frame
        )

    history = frames[
        0
    ]

    for frame in frames[
        1:
    ]:
        history = history.merge(
            frame,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

    return (
        history[
            ASSETS
        ]
        .tail(
            30
        )
        .reset_index(
            drop=True
        )
    )


def build_return_history(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    frames = []

    for asset in ASSETS:
        frame = (
            datasets[
                asset
            ][
                [
                    "timestamp",
                    "log_return",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_return": asset,
                }
            )
        )

        frame[
            asset
        ] = np.expm1(
            frame[
                asset
            ].to_numpy(
                dtype=float
            )
        )

        frames.append(
            frame
        )

    history = frames[
        0
    ]

    for frame in frames[
        1:
    ]:
        history = history.merge(
            frame,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

    return (
        history[
            ASSETS
        ]
        .tail(
            30
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    print(
        "\nProduction portfolio decision smoke test"
    )

    bundle = (
        load_production_models()
    )

    datasets = (
        load_datasets()
    )

    volatility_history = (
        build_var_history(
            datasets
        )
    )

    return_history = (
        build_return_history(
            datasets
        )
    )

    eth_features = (
        datasets[
            "ETH"
        ][
            FEATURE_COLUMNS
        ]
        .tail(
            7
        )
    )

    bnb_features = (
        datasets[
            "BNB"
        ][
            FEATURE_COLUMNS
        ]
        .iloc[
            -1
        ]
    )

    forecasts = (
        predict_all_assets(
            bundle,
            volatility_history=(
                volatility_history
            ),
            eth_feature_history=(
                eth_features
            ),
            bnb_latest_features=(
                bnb_features
            ),
        )
    )

    current_weights = pd.Series(
        {
            "BTC": 0.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "BNB": 0.0,
        },
        dtype=float,
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
        "\nForecast log volatility:"
    )

    print(
        decision
        .forecast_log_volatility
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
        "\nTrade weights:"
    )

    print(
        decision
        .trade_weights
        .to_string()
    )

    print(
        "\nForecast portfolio volatility:",
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
        "Turnover:",
        f"{decision.turnover:.6f}",
    )

    print(
        "Should rebalance:",
        decision.should_rebalance,
    )

    print(
        "\nProduction portfolio decision succeeded."
    )


if __name__ == "__main__":
    main()