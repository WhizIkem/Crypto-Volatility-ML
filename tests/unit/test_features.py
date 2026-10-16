import numpy as np
import pandas as pd

from crypto_volatility.features.build_features import (
    TARGET_COLUMN,
    add_cross_asset_features,
    add_har_features,
    add_target,
    build_single_asset_features,
)


def make_market_data(
    rows: int = 40,
) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2024-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    close = np.linspace(
        100.0,
        140.0,
        rows,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.linspace(
                1000.0,
                2000.0,
                rows,
            ),
            "quote_volume": np.linspace(
                100000.0,
                200000.0,
                rows,
            ),
            "number_of_trades": np.arange(
                1000,
                1000 + rows,
            ),
            "taker_buy_base_volume": np.linspace(
                500.0,
                1000.0,
                rows,
            ),
            "taker_buy_quote_volume": np.linspace(
                50000.0,
                100000.0,
                rows,
            ),
        }
    )


def test_add_target_uses_next_day_volatility():
    df = pd.DataFrame(
        {
            "log_gk_volatility": [
                -3.0,
                -2.0,
                -1.0,
            ]
        }
    )

    result = add_target(
        df
    )

    assert result.loc[
        0,
        TARGET_COLUMN,
    ] == -2.0

    assert result.loc[
        1,
        TARGET_COLUMN,
    ] == -1.0

    assert pd.isna(
        result.loc[
            2,
            TARGET_COLUMN,
        ]
    )


def test_har_features_use_only_current_and_past_values():
    df = pd.DataFrame(
        {
            "log_gk_volatility": np.arange(
                1.0,
                31.0,
            )
        }
    )

    result = add_har_features(
        df
    )

    assert result.loc[
        6,
        "har_vol_weekly",
    ] == np.mean(
        np.arange(
            1.0,
            8.0,
        )
    )

    assert result.loc[
        29,
        "har_vol_monthly",
    ] == np.mean(
        np.arange(
            1.0,
            31.0,
        )
    )


def test_single_asset_feature_builder_preserves_rows():
    df = make_market_data()

    result = build_single_asset_features(
        df
    )

    assert len(result) == len(df)

    assert (
        result["timestamp"]
        .is_monotonic_increasing
    )


def test_single_asset_target_alignment():
    df = make_market_data()

    result = build_single_asset_features(
        df
    )

    expected = (
        result["log_gk_volatility"]
        .shift(-1)
    )

    assert result[
        TARGET_COLUMN
    ].equals(
        expected
    )


def test_cross_asset_features_exclude_target_asset():
    btc = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=3,
                freq="D",
                tz="UTC",
            ),
            "log_return": [
                100.0,
                100.0,
                100.0,
            ],
            "log_gk_volatility": [
                100.0,
                100.0,
                100.0,
            ],
        }
    )

    eth = btc.copy()
    sol = btc.copy()

    eth["log_return"] = 1.0
    eth["log_gk_volatility"] = 2.0

    sol["log_return"] = 3.0
    sol["log_gk_volatility"] = 4.0

    result = add_cross_asset_features(
        {
            "BTC": btc,
            "ETH": eth,
            "SOL": sol,
        }
    )

    btc_result = result["BTC"]

    assert np.allclose(
        btc_result[
            "cross_asset_mean_return"
        ],
        2.0,
    )

    assert np.allclose(
        btc_result[
            "cross_asset_mean_log_volatility"
        ],
        3.0,
    )


def test_negative_return_indicator_preserves_missing_return():
    df = make_market_data()

    result = build_single_asset_features(
        df
    )

    assert pd.isna(
        result.loc[
            0,
            "negative_return_indicator",
        ]
    )