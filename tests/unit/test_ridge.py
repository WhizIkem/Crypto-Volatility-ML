import numpy as np
import pandas as pd
import pytest

from crypto_volatility.models.ridge import (
    fit_ridge_model,
    predict_ridge,
)


def make_ridge_data(
    rows: int = 200,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        42
    )

    feature_a = rng.normal(
        size=rows
    )

    feature_b = rng.normal(
        size=rows
    )

    target = (
        1.5
        + 0.8 * feature_a
        - 0.4 * feature_b
        + rng.normal(
            scale=0.05,
            size=rows,
        )
    )

    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "target": target,
        }
    )


def test_fit_ridge_model():
    df = make_ridge_data()

    model = fit_ridge_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
    )

    assert model is not None

    assert "scaler" in model.named_steps

    assert "ridge" in model.named_steps


def test_ridge_scaler_uses_training_statistics():
    df = make_ridge_data()

    model = fit_ridge_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
    )

    scaler = model.named_steps[
        "scaler"
    ]

    expected_means = (
        df[
            [
                "feature_a",
                "feature_b",
            ]
        ]
        .mean()
        .to_numpy()
    )

    assert np.allclose(
        scaler.mean_,
        expected_means,
    )


def test_predict_ridge_returns_expected_length_and_index():
    df = make_ridge_data()

    model = fit_ridge_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
    )

    predictions = predict_ridge(
        model,
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
    )

    assert len(predictions) == len(df)

    assert predictions.index.equals(
        df.index
    )


def test_ridge_predictions_are_finite():
    df = make_ridge_data()

    model = fit_ridge_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
    )

    predictions = predict_ridge(
        model,
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
    )

    assert np.isfinite(
        predictions.to_numpy()
    ).all()


def test_fit_ridge_rejects_missing_columns():
    df = make_ridge_data()

    with pytest.raises(
        ValueError,
        match="Missing Ridge columns",
    ):
        fit_ridge_model(
            df,
            feature_columns=[
                "does_not_exist",
            ],
            target_column="target",
        )


def test_fit_ridge_rejects_invalid_alpha():
    df = make_ridge_data()

    with pytest.raises(
        ValueError,
        match="alpha must be greater than zero",
    ):
        fit_ridge_model(
            df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            alpha=0,
        )


def test_predict_ridge_rejects_missing_values():
    df = make_ridge_data()

    model = fit_ridge_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
    )

    prediction_df = df.copy()

    prediction_df.loc[
        0,
        "feature_a",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="cannot contain missing values",
    ):
        predict_ridge(
            model,
            prediction_df,
            feature_columns=[
                "feature_a",
                "feature_b",
            ],
        )