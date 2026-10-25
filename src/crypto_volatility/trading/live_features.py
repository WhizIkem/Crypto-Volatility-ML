from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_volatility.features.build_features import (
    build_feature_datasets,
)
from crypto_volatility.models.production import (
    ASSETS,
    FEATURE_COLUMNS,
)

LIVE_BASE_COLUMNS = [
    "timestamp",
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


def prepare_live_base_frame(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert Binance live candle output into the same base schema
    used by the research feature-engineering pipeline.
    """
    required_columns = {
        "open_time",
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

    missing = (
        required_columns
        - set(
            candles.columns
        )
    )

    if missing:
        raise ValueError(
            "Live candles missing columns: "
            f"{sorted(missing)}"
        )

    frame = (
        candles
        .rename(
            columns={
                "open_time": "timestamp",
            }
        )
        [
            LIVE_BASE_COLUMNS
        ]
        .copy()
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=[
                "timestamp",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if frame.empty:
        raise ValueError(
            "Live candle frame cannot be empty"
        )

    if frame[
        "timestamp"
    ].isna().any():
        raise ValueError(
            "Live candle timestamps contain missing values"
        )

    return frame


def build_live_feature_datasets(
    live_candles: dict[
        str,
        pd.DataFrame,
    ],
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Run live candles through the exact research feature pipeline.

    This intentionally calls build_feature_datasets rather than
    reimplementing feature formulas inside the trading layer.
    """
    missing_assets = (
        set(
            ASSETS
        )
        - set(
            live_candles
        )
    )

    if missing_assets:
        raise ValueError(
            "Live candle datasets missing assets: "
            f"{sorted(missing_assets)}"
        )

    base_datasets = {
        asset: prepare_live_base_frame(
            live_candles[
                asset
            ]
        )
        for asset in ASSETS
    }

    latest_timestamps = {
        asset: frame[
            "timestamp"
        ].iloc[
            -1
        ]
        for asset, frame in (
            base_datasets.items()
        )
    }

    if len(
        set(
            latest_timestamps.values()
        )
    ) != 1:
        raise ValueError(
            "Live assets are not aligned on the same latest candle"
        )

    features = build_feature_datasets(
        base_datasets
    )

    for asset in ASSETS:
        frame = features[
            asset
        ]

        if frame.empty:
            raise ValueError(
                f"{asset} live feature dataset is empty"
            )

        missing_features = (
            set(
                FEATURE_COLUMNS
            )
            - set(
                frame.columns
            )
        )

        if missing_features:
            raise ValueError(
                f"{asset} missing production features: "
                f"{sorted(missing_features)}"
            )

    return features


def get_latest_valid_feature_row(
    feature_df: pd.DataFrame,
) -> pd.Series:
    """
    Return the newest row with all production feature values present.

    The supervised t+1 target is deliberately ignored because it
    is unavailable for the current live forecast origin.
    """
    required = [
        "timestamp",
        *FEATURE_COLUMNS,
    ]

    missing = (
        set(
            required
        )
        - set(
            feature_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Feature frame missing columns: "
            f"{sorted(missing)}"
        )

    valid = (
        feature_df[
            required
        ]
        .dropna(
            subset=FEATURE_COLUMNS
        )
        .sort_values(
            "timestamp"
        )
    )

    if valid.empty:
        raise ValueError(
            "No valid live production feature rows available"
        )

    latest = valid.iloc[
        -1
    ].copy()

    values = (
        latest[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Latest production feature row contains non-finite values"
        )

    return latest


def get_latest_valid_feature_history(
    feature_df: pd.DataFrame,
    *,
    rows: int,
) -> pd.DataFrame:
    """
    Return the latest consecutive usable production feature rows.

    Used by sequence models such as the ETH LSTM.
    """
    if rows < 1:
        raise ValueError(
            "rows must be at least 1"
        )

    required = [
        "timestamp",
        *FEATURE_COLUMNS,
    ]

    missing = (
        set(
            required
        )
        - set(
            feature_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Feature frame missing columns: "
            f"{sorted(missing)}"
        )

    valid = (
        feature_df[
            required
        ]
        .dropna(
            subset=FEATURE_COLUMNS
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        valid
    ) < rows:
        raise ValueError(
            f"Need at least {rows} valid feature rows, "
            f"found {len(valid)}"
        )

    result = (
        valid
        .tail(
            rows
        )
        .reset_index(
            drop=True
        )
    )

    values = (
        result[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Feature history contains non-finite values"
        )

    return result


def build_live_var_history(
    feature_datasets: dict[
        str,
        pd.DataFrame,
    ],
    *,
    rows: int = 30,
) -> pd.DataFrame:
    """
    Build aligned log-volatility history for production VAR.
    """
    frames = []

    for asset in ASSETS:
        frame = (
            feature_datasets[
                asset
            ][
                [
                    "timestamp",
                    "log_gk_volatility",
                ]
            ]
            .dropna()
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

    history = (
        history
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        history
    ) < rows:
        raise ValueError(
            f"Need at least {rows} aligned volatility rows, "
            f"found {len(history)}"
        )

    result = (
        history[
            ASSETS
        ]
        .tail(
            rows
        )
        .reset_index(
            drop=True
        )
    )

    if not np.isfinite(
        result.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "VAR history contains non-finite values"
        )

    return result


def build_live_return_history(
    feature_datasets: dict[
        str,
        pd.DataFrame,
    ],
    *,
    rows: int = 30,
) -> pd.DataFrame:
    """
    Build aligned simple-return history for portfolio covariance.

    Research features store log returns, so they are converted
    back to simple returns before portfolio calculations.
    """
    frames = []

    for asset in ASSETS:
        frame = (
            feature_datasets[
                asset
            ][
                [
                    "timestamp",
                    "log_return",
                ]
            ]
            .dropna()
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

    history = (
        history
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        history
    ) < rows:
        raise ValueError(
            f"Need at least {rows} aligned return rows, "
            f"found {len(history)}"
        )

    result = (
        history[
            ASSETS
        ]
        .tail(
            rows
        )
        .reset_index(
            drop=True
        )
    )

    if not np.isfinite(
        result.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Return history contains non-finite values"
        )

    return result