import numpy as np
import pandas as pd
import pytest

from crypto_volatility.data.sequences import (
    build_leakage_safe_sequence_splits,
    build_sequence_train_validation_split,
    build_sequences,
    inverse_transform_sequence_targets,
    scale_sequence_targets,
)


def make_sequence_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "feature_b": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
            "target": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
            ],
        }
    )


def test_build_sequences_returns_expected_shapes():
    df = make_sequence_data()

    x, y = build_sequences(
        df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        sequence_length=3,
    )

    assert x.shape == (
        3,
        3,
        2,
    )

    assert y.shape == (
        3,
    )


def test_build_sequences_preserves_time_order():
    df = make_sequence_data()

    x, y = build_sequences(
        df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        sequence_length=3,
    )

    expected_first_sequence = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    assert np.array_equal(
        x[0],
        expected_first_sequence,
    )

    assert y[0] == 300.0


def test_build_sequences_uses_target_at_sequence_end():
    df = make_sequence_data()

    _, y = build_sequences(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=2,
    )

    assert np.array_equal(
        y,
        np.array(
            [
                200.0,
                300.0,
                400.0,
                500.0,
            ]
        ),
    )


def test_build_sequences_rejects_invalid_length():
    df = make_sequence_data()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        build_sequences(
            df=df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            sequence_length=0,
        )


def test_build_sequences_rejects_missing_columns():
    df = make_sequence_data()

    with pytest.raises(
        ValueError,
        match="Missing sequence columns",
    ):
        build_sequences(
            df=df,
            feature_columns=[
                "does_not_exist",
            ],
            target_column="target",
            sequence_length=2,
        )


def test_build_sequences_rejects_short_dataset():
    df = make_sequence_data()

    with pytest.raises(
        ValueError,
        match="at least sequence_length",
    ):
        build_sequences(
            df=df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            sequence_length=10,
        )

def make_split_data(
    rows: int = 100,
) -> pd.DataFrame:
    values = np.arange(
        rows,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "feature_a": values,
            "feature_b": values * 10.0,
            "target": values + 1000.0,
        }
    )


def test_sequence_split_uses_expected_partition_sizes():
    df = make_split_data()

    split = build_leakage_safe_sequence_splits(
        df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        sequence_length=10,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    # Training target rows are 0..68.
    # The first 9 rows cannot form complete 10-row sequences.
    assert split.x_train.shape == (
        60,
        10,
        2,
    )

    # Validation target rows are 70..83.
    assert split.x_validation.shape == (
        14,
        10,
        2,
    )

    # Test target rows are 85..99.
    assert split.x_test.shape == (
        15,
        10,
        2,
    )


def test_sequence_split_preserves_purged_target_boundaries():
    df = make_split_data()

    split = build_leakage_safe_sequence_splits(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    assert split.y_train[-1] == 1068.0

    assert split.y_validation[0] == 1070.0
    assert split.y_validation[-1] == 1083.0

    assert split.y_test[0] == 1085.0

    # Purged rows 69 and 84 must never appear as targets.
    all_targets = np.concatenate(
        [
            split.y_train,
            split.y_validation,
            split.y_test,
        ]
    )

    assert 1069.0 not in all_targets
    assert 1084.0 not in all_targets


def test_validation_sequence_can_use_purge_row_as_history():
    df = make_split_data()

    split = build_leakage_safe_sequence_splits(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    scaler = split.feature_scaler

    expected_history = scaler.transform(
        np.array(
            [
                [66.0],
                [67.0],
                [68.0],
                [69.0],
                [70.0],
            ]
        )
    )

    assert np.allclose(
        split.x_validation[0],
        expected_history,
    )


def test_feature_scaler_is_fitted_on_training_rows_only():
    df = make_split_data()

    split = build_leakage_safe_sequence_splits(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    # With 100 rows and a one-row purge,
    # training feature rows are 0..68.
    expected_training_mean = (
        np.arange(
            69,
            dtype=float,
        ).mean()
    )

    assert np.isclose(
        split.feature_scaler.mean_[0],
        expected_training_mean,
    )


def test_test_sequence_can_use_validation_history():
    df = make_split_data()

    split = build_leakage_safe_sequence_splits(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    scaler = split.feature_scaler

    expected_history = scaler.transform(
        np.array(
            [
                [81.0],
                [82.0],
                [83.0],
                [84.0],
                [85.0],
            ]
        )
    )

    assert np.allclose(
        split.x_test[0],
        expected_history,
    )

def test_inner_sequence_split_preserves_purge_boundary():
    df = make_split_data()

    split = build_sequence_train_validation_split(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.80,
        purge_rows=1,
    )

    # Inner train targets: rows 0..78
    assert split.y_train[-1] == 1078.0

    # Row 79 is purged as a supervised target.
    assert 1079.0 not in split.y_train
    assert 1079.0 not in split.y_validation

    # Inner validation starts at row 80.
    assert split.y_validation[0] == 1080.0


def test_inner_validation_sequence_uses_prior_context():
    df = make_split_data()

    split = build_sequence_train_validation_split(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.80,
        purge_rows=1,
    )

    scaler = split.feature_scaler

    expected_history = scaler.transform(
        np.array(
            [
                [76.0],
                [77.0],
                [78.0],
                [79.0],
                [80.0],
            ]
        )
    )

    assert np.allclose(
        split.x_validation[0],
        expected_history,
    )


def test_inner_sequence_scaler_uses_train_only():
    df = make_split_data()

    split = build_sequence_train_validation_split(
        df=df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        sequence_length=5,
        train_fraction=0.80,
        purge_rows=1,
    )

    expected_mean = (
        np.arange(
            79,
            dtype=float,
        ).mean()
    )

    assert np.isclose(
        split.feature_scaler.mean_[0],
        expected_mean,
    )

def test_scale_sequence_targets_uses_training_statistics_only():
    y_train = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    y_validation = np.array(
        [
            100.0,
            200.0,
        ]
    )

    scaled = scale_sequence_targets(
        y_train=y_train,
        y_validation=y_validation,
    )

    assert np.isclose(
        scaled.target_scaler.mean_[0],
        2.0,
    )


def test_scaled_training_targets_have_zero_mean():
    y_train = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    y_validation = np.array(
        [
            5.0,
            6.0,
        ]
    )

    scaled = scale_sequence_targets(
        y_train=y_train,
        y_validation=y_validation,
    )

    assert np.isclose(
        scaled.y_train.mean(),
        0.0,
    )


def test_inverse_transform_sequence_targets_recovers_original_values():
    y_train = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    y_validation = np.array(
        [
            4.0,
            5.0,
        ]
    )

    scaled = scale_sequence_targets(
        y_train=y_train,
        y_validation=y_validation,
    )

    recovered = (
        inverse_transform_sequence_targets(
            scaled.y_validation,
            scaled.target_scaler,
        )
    )

    assert np.allclose(
        recovered,
        y_validation,
    )