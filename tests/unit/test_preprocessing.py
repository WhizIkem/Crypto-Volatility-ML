import numpy as np
import pandas as pd
import pytest

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
    standardize_feature_splits,
)


def make_dataset(
    rows: int = 100,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="D",
                tz="UTC",
            ),
            "feature": np.arange(
                rows,
                dtype=float,
            ),
            "feature_2": np.arange(
                rows,
                dtype=float,
            ) * 10,
            "target": np.arange(
                rows,
                dtype=float,
            ),
        }
    )


def test_chronological_split_purges_boundary_rows():
    df = make_dataset(
        rows=100
    )

    split = chronological_train_validation_test_split(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    assert len(split.train) == 69
    assert len(split.validation) == 14
    assert len(split.test) == 15


def test_chronological_split_preserves_order():
    df = make_dataset(
        rows=100
    )

    split = chronological_train_validation_test_split(
        df
    )

    assert (
        split.train["timestamp"].max()
        < split.validation["timestamp"].min()
    )

    assert (
        split.validation["timestamp"].max()
        < split.test["timestamp"].min()
    )


def test_chronological_split_uses_oldest_data_for_training():
    df = make_dataset(
        rows=100
    )

    split = chronological_train_validation_test_split(
        df
    )

    assert (
        split.train["timestamp"].iloc[0]
        == df["timestamp"].iloc[0]
    )

    assert (
        split.test["timestamp"].iloc[-1]
        == df["timestamp"].iloc[-1]
    )


def test_chronological_split_does_not_drop_rows():
    df = make_dataset(
        rows=100
    )

    split = chronological_train_validation_test_split(
        df
    )

    total_rows = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    assert total_rows == len(df) - 2  # accounting for the purged rows at the boundaries


def test_chronological_split_rejects_unsorted_data():
    df = (
        make_dataset(
            rows=100
        )
        .sort_values(
            "timestamp",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    with pytest.raises(
        ValueError,
        match="sorted chronologically",
    ):
        chronological_train_validation_test_split(
            df
        )


def test_chronological_split_rejects_invalid_fractions():
    df = make_dataset(
        rows=100
    )

    with pytest.raises(
        ValueError,
        match="must be less than 1",
    ):
        chronological_train_validation_test_split(
            df,
            train_fraction=0.90,
            validation_fraction=0.20,
        )


def test_standardization_centres_training_features():
    df = make_dataset(
        rows=100
    )

    split = (
        chronological_train_validation_test_split(
            df
        )
    )

    scaled = standardize_feature_splits(
        split,
        feature_columns=[
            "feature",
            "feature_2",
        ],
    )

    assert np.allclose(
        scaled.train[
            [
                "feature",
                "feature_2",
            ]
        ].mean(),
        0.0,
        atol=1e-12,
    )


def test_standardization_uses_training_statistics_only():
    df = make_dataset(
        rows=100
    )

    split = (
        chronological_train_validation_test_split(
            df
        )
    )

    scaled = standardize_feature_splits(
        split,
        feature_columns=[
            "feature",
        ],
    )

    train_mean = (
        split.train["feature"]
        .mean()
    )

    train_std = (
        split.train["feature"]
        .std(ddof=0)
    )

    expected_first_validation = (
        (
            split.validation[
                "feature"
            ].iloc[0]
            - train_mean
        )
        / train_std
    )

    assert scaled.validation[
        "feature"
    ].iloc[0] == pytest.approx(
        expected_first_validation
    )


def test_standardization_does_not_scale_target():
    df = make_dataset(
        rows=100
    )

    split = (
        chronological_train_validation_test_split(
            df
        )
    )

    scaled = standardize_feature_splits(
        split,
        feature_columns=[
            "feature",
            "feature_2",
        ],
    )

    pd.testing.assert_series_equal(
        scaled.train["target"],
        split.train["target"],
    )

    pd.testing.assert_series_equal(
        scaled.validation["target"],
        split.validation["target"],
    )

    pd.testing.assert_series_equal(
        scaled.test["target"],
        split.test["target"],
    )


def test_standardization_preserves_timestamps():
    df = make_dataset(
        rows=100
    )

    split = (
        chronological_train_validation_test_split(
            df
        )
    )

    scaled = standardize_feature_splits(
        split,
        feature_columns=[
            "feature",
        ],
    )

    pd.testing.assert_series_equal(
        scaled.train["timestamp"],
        split.train["timestamp"],
    )


def test_standardization_rejects_missing_features():
    df = make_dataset(
        rows=100
    )

    split = (
        chronological_train_validation_test_split(
            df
        )
    )

    with pytest.raises(
        ValueError,
        match="missing feature columns",
    ):
        standardize_feature_splits(
            split,
            feature_columns=[
                "does_not_exist",
            ],
        )


def test_standardization_rejects_missing_feature_values():
    df = make_dataset(
        rows=100
    )

    df.loc[
        10,
        "feature",
    ] = np.nan

    split = chronological_train_validation_test_split(
        df
    )

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        standardize_feature_splits(
            split,
            feature_columns=[
                "feature",
            ],
        )

def test_split_purges_one_row_at_each_boundary():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=100,
                freq="D",
            ),
            "feature": range(100),
            "target": range(1, 101),
        }
    )

    split = chronological_train_validation_test_split(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    assert len(split.train) == 69
    assert len(split.validation) == 14
    assert len(split.test) == 15


def test_purged_rows_are_not_present_in_any_partition():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=100,
                freq="D",
            ),
            "feature": range(100),
            "target": range(1, 101),
        }
    )

    split = chronological_train_validation_test_split(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    all_split_timestamps = set(
        pd.concat(
            [
                split.train["timestamp"],
                split.validation["timestamp"],
                split.test["timestamp"],
            ]
        )
    )

    first_purged_timestamp = df.loc[
        69,
        "timestamp",
    ]

    second_purged_timestamp = df.loc[
        84,
        "timestamp",
    ]

    assert (
        first_purged_timestamp
        not in all_split_timestamps
    )

    assert (
        second_purged_timestamp
        not in all_split_timestamps
    )


def test_split_rejects_negative_purge():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=100,
                freq="D",
            ),
            "feature": range(100),
        }
    )

    with pytest.raises(
        ValueError,
        match="purge_rows cannot be negative",
    ):
        chronological_train_validation_test_split(
            df,
            purge_rows=-1,
        )