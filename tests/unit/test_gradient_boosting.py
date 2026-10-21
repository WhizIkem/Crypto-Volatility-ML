import numpy as np
import pandas as pd
import pytest

from crypto_volatility.models.gradient_boosting import (
    fit_gradient_boosting_model,
    predict_gradient_boosting,
)


def make_gradient_boosting_data(
    rows: int = 250,
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
        0.5 * feature_a**2
        + 0.3 * np.abs(feature_b)
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


def test_fit_gradient_boosting_model():
    df = make_gradient_boosting_data()

    model = fit_gradient_boosting_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
    )

    assert model is not None
    assert model.n_estimators == 50


def test_predict_gradient_boosting_returns_expected_length_and_index():
    df = make_gradient_boosting_data()

    model = fit_gradient_boosting_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
    )

    predictions = predict_gradient_boosting(
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


def test_gradient_boosting_predictions_are_finite():
    df = make_gradient_boosting_data()

    model = fit_gradient_boosting_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
    )

    predictions = predict_gradient_boosting(
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


def test_gradient_boosting_rejects_missing_columns():
    df = make_gradient_boosting_data()

    with pytest.raises(
        ValueError,
        match="Missing Gradient Boosting columns",
    ):
        fit_gradient_boosting_model(
            df,
            feature_columns=[
                "does_not_exist",
            ],
            target_column="target",
        )


def test_gradient_boosting_rejects_invalid_learning_rate():
    df = make_gradient_boosting_data()

    with pytest.raises(
        ValueError,
        match="learning_rate must be greater than zero",
    ):
        fit_gradient_boosting_model(
            df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            learning_rate=0.0,
        )


def test_predict_gradient_boosting_rejects_missing_values():
    df = make_gradient_boosting_data()

    model = fit_gradient_boosting_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
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
        predict_gradient_boosting(
            model,
            prediction_df,
            feature_columns=[
                "feature_a",
                "feature_b",
            ],
        )