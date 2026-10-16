import numpy as np
import pandas as pd
import pytest

from crypto_volatility.features.build_features import (
    TARGET_COLUMN,
)
from crypto_volatility.features.model_dataset import (
    build_model_dataset,
)


def make_feature_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=5,
                freq="D",
                tz="UTC",
            ),
            "symbol": [
                "BTCUSDT",
            ] * 5,
            "feature_a": [
                np.nan,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "feature_b": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
            ],
            TARGET_COLUMN: [
                100.0,
                101.0,
                102.0,
                103.0,
                np.nan,
            ],
        }
    )


def test_build_model_dataset_drops_missing_boundaries():
    df = make_feature_data()

    result = build_model_dataset(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
    )

    assert len(result) == 3

    assert result[
        "timestamp"
    ].iloc[0] == pd.Timestamp(
        "2024-01-02",
        tz="UTC",
    )

    assert result[
        "timestamp"
    ].iloc[-1] == pd.Timestamp(
        "2024-01-04",
        tz="UTC",
    )


def test_build_model_dataset_contains_no_missing_values():
    df = make_feature_data()

    result = build_model_dataset(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
    )

    assert not result[
        [
            "feature_a",
            "feature_b",
            TARGET_COLUMN,
        ]
    ].isna().any().any()


def test_build_model_dataset_rejects_missing_columns():
    df = make_feature_data()

    with pytest.raises(
        ValueError,
        match="Missing model-dataset columns",
    ):
        build_model_dataset(
            df,
            feature_columns=[
                "does_not_exist",
            ],
        )


def test_build_model_dataset_rejects_infinite_values():
    df = make_feature_data()

    df.loc[
        2,
        "feature_a",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="infinite values",
    ):
        build_model_dataset(
            df,
            feature_columns=[
                "feature_a",
                "feature_b",
            ],
        )