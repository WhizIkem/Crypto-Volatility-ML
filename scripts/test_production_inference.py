from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.models.production import (
    ASSETS,
    FEATURE_COLUMNS,
    load_production_models,
    predict_all_assets,
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

        if not path.exists():
            raise FileNotFoundError(
                f"{asset} dataset not found: {path}"
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
    """
    Build aligned current volatility history for production VAR.
    """
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


def main() -> None:
    print(
        "\nProduction inference smoke test"
    )

    print(
        "\nLoading production artifacts..."
    )

    bundle = (
        load_production_models()
    )

    print(
        "Artifacts loaded successfully."
    )

    datasets = (
        load_datasets()
    )

    latest_timestamp = (
        datasets[
            "BTC"
        ][
            "timestamp"
        ]
        .iloc[
            -1
        ]
    )

    volatility_history = (
        build_var_history(
            datasets
        )
    )

    eth_feature_history = (
        datasets[
            "ETH"
        ][
            FEATURE_COLUMNS
        ]
        .tail(
            7
        )
        .copy()
    )

    bnb_latest_features = (
        datasets[
            "BNB"
        ][
            FEATURE_COLUMNS
        ]
        .iloc[
            -1
        ]
        .copy()
    )

    forecasts = (
        predict_all_assets(
            bundle,
            volatility_history=(
                volatility_history
            ),
            eth_feature_history=(
                eth_feature_history
            ),
            bnb_latest_features=(
                bnb_latest_features
            ),
        )
    )

    print(
        "\nForecast origin:",
        latest_timestamp,
    )

    print(
        "\nNext-day log-volatility forecasts:"
    )

    print(
        forecasts
        .to_series()
        .to_string()
    )

    print(
        "\nProduction inference succeeded."
    )


if __name__ == "__main__":
    main()