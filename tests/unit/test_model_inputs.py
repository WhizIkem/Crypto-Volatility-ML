import numpy as np
import pandas as pd

from crypto_volatility.data.model_inputs import (
    prepare_model_inputs,
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
            "feature_a": np.arange(
                rows,
                dtype=float,
            ),
            "feature_b": np.arange(
                rows,
                dtype=float,
            ) * 10,
            "target": np.arange(
                rows,
                dtype=float,
            ),
        }
    )


def test_prepare_model_inputs_shapes():
    df = make_dataset()

    inputs = prepare_model_inputs(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
    )

    assert inputs.x_train.shape == (
        69,
        2,
    )

    assert inputs.x_validation.shape == (
        14,
        2,
    )

    assert inputs.x_test.shape == (
        15,
        2,
    )

    assert len(inputs.y_train) == 69
    assert len(inputs.y_validation) == 14
    assert len(inputs.y_test) == 15


def test_prepare_model_inputs_preserves_time_order():
    df = make_dataset()

    inputs = prepare_model_inputs(
        df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
    )

    assert (
        inputs.train_timestamps.max()
        < inputs.validation_timestamps.min()
    )

    assert (
        inputs.validation_timestamps.max()
        < inputs.test_timestamps.min()
    )


def test_prepare_model_inputs_can_scale_features():
    df = make_dataset()

    inputs = prepare_model_inputs(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        scale_features=True,
    )

    assert np.allclose(
        inputs.x_train.mean(),
        0.0,
        atol=1e-12,
    )


def test_prepare_model_inputs_does_not_scale_target():
    df = make_dataset()

    inputs = prepare_model_inputs(
        df,
        feature_columns=[
            "feature_a",
        ],
        target_column="target",
        scale_features=True,
    )

    expected_target = (
        df["target"]
        .iloc[:69]
        .reset_index(drop=True)
    )

    pd.testing.assert_series_equal(
        inputs.y_train.reset_index(
            drop=True
        ),
        expected_target,
    )