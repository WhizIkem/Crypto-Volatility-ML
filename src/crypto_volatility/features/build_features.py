from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_volatility.analysis.volatility import (
    add_garman_klass_volatility,
    add_log_garman_klass_volatility,
    add_log_returns,
)

TRADING_DAYS = 365
HAR_WEEKLY_WINDOW = 7
HAR_MONTHLY_WINDOW = 30
RSI_WINDOW = 14

TARGET_COLUMN = "target_log_gk_vol_t_plus_1"


CANDIDATE_FEATURES = [
    "log_return",
    "log_return_lag_1",
    "log_return_lag_2",
    "log_return_lag_3",
    "log_gk_volatility",
    "log_gk_volatility_lag_1",
    "log_gk_volatility_lag_2",
    "log_gk_volatility_lag_3",
    "har_vol_weekly",
    "har_vol_monthly",
    "squared_return",
    "absolute_return",
    "negative_return_indicator",
    "downside_return",
    "log_volume",
    "log_quote_volume",
    "log_number_of_trades",
    "volume_change",
    "quote_volume_change",
    "trade_count_change",
    "taker_buy_volume_ratio",
    "momentum_7d",
    "momentum_30d",
    "rsi_14",
    "cross_asset_mean_return",
    "cross_asset_mean_log_volatility",
    "cross_asset_return_dispersion",
    "cross_asset_volatility_dispersion",
]


def _validate_required_columns(
    df: pd.DataFrame,
) -> None:
    """
    Validate columns required by the feature pipeline.
    """
    required_columns = {
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
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            f"{sorted(missing_columns)}"
        )


def add_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the next-day log Garman-Klass volatility target.
    """
    if "log_gk_volatility" not in df.columns:
        raise ValueError(
            "log_gk_volatility must exist before creating the target"
        )

    result = df.copy()

    result[TARGET_COLUMN] = (
        result["log_gk_volatility"]
        .shift(-1)
    )

    return result


def add_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add lagged return and volatility features.
    """
    required_columns = {
        "log_return",
        "log_gk_volatility",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing lag source columns: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    for lag in (1, 2, 3):
        result[
            f"log_return_lag_{lag}"
        ] = (
            result["log_return"]
            .shift(lag)
        )

        result[
            f"log_gk_volatility_lag_{lag}"
        ] = (
            result["log_gk_volatility"]
            .shift(lag)
        )

    return result


def add_har_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add weekly and monthly HAR-style volatility features.
    """
    if "log_gk_volatility" not in df.columns:
        raise ValueError(
            "log_gk_volatility is required for HAR features"
        )

    result = df.copy()

    result["har_vol_weekly"] = (
        result["log_gk_volatility"]
        .rolling(
            window=HAR_WEEKLY_WINDOW,
            min_periods=HAR_WEEKLY_WINDOW,
        )
        .mean()
    )

    result["har_vol_monthly"] = (
        result["log_gk_volatility"]
        .rolling(
            window=HAR_MONTHLY_WINDOW,
            min_periods=HAR_MONTHLY_WINDOW,
        )
        .mean()
    )

    return result


def add_return_magnitude_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add return magnitude and downside-shock features.
    """
    if "log_return" not in df.columns:
        raise ValueError(
            "log_return is required for return magnitude features"
        )

    result = df.copy()

    result["squared_return"] = (
        result["log_return"]
        .pow(2)
    )

    result["absolute_return"] = (
        result["log_return"]
        .abs()
    )

    result["negative_return_indicator"] = (
        result["log_return"]
        .lt(0)
        .where(
            result["log_return"].notna()
        )
    .astype("Int64")
    )

    result["downside_return"] = (
        result["log_return"]
        .clip(upper=0)
        .abs()
    )

    return result


def add_activity_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add volume, trade-activity, and taker-buy features.
    """
    result = df.copy()

    result["log_volume"] = np.log1p(
        result["volume"]
    )

    result["log_quote_volume"] = np.log1p(
        result["quote_volume"]
    )

    result["log_number_of_trades"] = np.log1p(
        result["number_of_trades"]
    )

    result["volume_change"] = (
        result["volume"]
        .pct_change()
    )

    result["quote_volume_change"] = (
        result["quote_volume"]
        .pct_change()
    )

    result["trade_count_change"] = (
        result["number_of_trades"]
        .pct_change()
    )

    safe_volume = (
        result["volume"]
        .replace(0, np.nan)
    )

    result["taker_buy_volume_ratio"] = (
        result["taker_buy_base_volume"]
        / safe_volume
    )

    return result


def add_momentum_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add momentum and RSI features.
    """
    result = df.copy()

    result["momentum_7d"] = np.log(
        result["close"]
        / result["close"].shift(7)
    )

    result["momentum_30d"] = np.log(
        result["close"]
        / result["close"].shift(30)
    )

    price_change = (
        result["close"]
        .diff()
    )

    gains = (
        price_change
        .clip(lower=0)
    )

    losses = (
        -price_change
        .clip(upper=0)
    )

    average_gain = (
        gains
        .rolling(
            window=RSI_WINDOW,
            min_periods=RSI_WINDOW,
        )
        .mean()
    )

    average_loss = (
        losses
        .rolling(
            window=RSI_WINDOW,
            min_periods=RSI_WINDOW,
        )
        .mean()
    )

    relative_strength = (
        average_gain
        / average_loss
    )

    result["rsi_14"] = (
        100
        - (
            100
            / (
                1
                + relative_strength
            )
        )
    )

    return result


def build_single_asset_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build validated single-asset features.

    Cross-asset features are added separately because they
    require simultaneous access to all asset datasets.
    """
    _validate_required_columns(
        df
    )

    result = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    result = add_log_returns(
        result
    )

    result = add_garman_klass_volatility(
        result,
        annualization_factor=TRADING_DAYS,
    )

    result = add_log_garman_klass_volatility(
        result
    )

    result = add_target(
        result
    )

    result = add_lag_features(
        result
    )

    result = add_har_features(
        result
    )

    result = add_return_magnitude_features(
        result
    )

    result = add_activity_features(
        result
    )

    result = add_momentum_features(
        result
    )

    return result


def add_cross_asset_features(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Add aggregate cross-asset return and volatility features.

    Each asset uses only the other assets when constructing
    its cross-asset aggregates.
    """
    if len(datasets) < 2:
        raise ValueError(
            "Cross-asset features require at least two assets"
        )

    return_panel = pd.DataFrame(
        {
            asset: (
                df
                .set_index("timestamp")
                ["log_return"]
            )
            for asset, df in datasets.items()
        }
    )

    volatility_panel = pd.DataFrame(
        {
            asset: (
                df
                .set_index("timestamp")
                ["log_gk_volatility"]
            )
            for asset, df in datasets.items()
        }
    )

    output = {}

    for asset, df in datasets.items():
        other_assets = [
            other_asset
            for other_asset in datasets
            if other_asset != asset
        ]

        cross_return_mean = (
            return_panel[
                other_assets
            ]
            .mean(axis=1)
        )

        cross_volatility_mean = (
            volatility_panel[
                other_assets
            ]
            .mean(axis=1)
        )

        cross_return_dispersion = (
            return_panel[
                other_assets
            ]
            .std(axis=1)
        )

        cross_volatility_dispersion = (
            volatility_panel[
                other_assets
            ]
            .std(axis=1)
        )

        result = df.copy()

        result[
            "cross_asset_mean_return"
        ] = (
            result["timestamp"]
            .map(cross_return_mean)
        )

        result[
            "cross_asset_mean_log_volatility"
        ] = (
            result["timestamp"]
            .map(cross_volatility_mean)
        )

        result[
            "cross_asset_return_dispersion"
        ] = (
            result["timestamp"]
            .map(cross_return_dispersion)
        )

        result[
            "cross_asset_volatility_dispersion"
        ] = (
            result["timestamp"]
            .map(cross_volatility_dispersion)
        )

        output[asset] = result

    return output


def build_feature_datasets(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Build the complete candidate feature dataset for all assets.
    """
    single_asset_features = {
        asset: build_single_asset_features(
            df
        )
        for asset, df in datasets.items()
    }

    return add_cross_asset_features(
        single_asset_features
    )