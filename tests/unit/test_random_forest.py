import numpy as np
import pandas as pd
import pytest

from crypto_volatility.models.random_forest import (
    fit_random_forest_model,
    predict_random_forest,
)


def make_random_forest_data(
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


def test_fit_random_forest_model():
    df = make_random_forest_data()

    model = fit_random_forest_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
        random_state=42,
    )

    assert model is not None

    assert model.n_estimators == 50


def test_predict_random_forest_returns_expected_length_and_index():
    df = make_random_forest_data()

    model = fit_random_forest_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
    )

    predictions = predict_random_forest(
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


def test_random_forest_predictions_are_finite():
    df = make_random_forest_data()

    model = fit_random_forest_model(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        n_estimators=50,
    )

    predictions = predict_random_forest(
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


def test_random_forest_rejects_missing_columns():
    df = make_random_forest_data()

    with pytest.raises(
        ValueError,
        match="Missing Random Forest columns",
    ):
        fit_random_forest_model(
            df,
            feature_columns=[
                "does_not_exist",
            ],
            target_column="target",
        )


def test_random_forest_rejects_invalid_n_estimators():
    df = make_random_forest_data()

    with pytest.raises(
        ValueError,
        match="n_estimators must be at least 1",
    ):
        fit_random_forest_model(
            df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            n_estimators=0,
        )


def test_predict_random_forest_rejects_missing_values():
    df = make_random_forest_data()

    model = fit_random_forest_model(
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
        predict_random_forest(
            model,
            prediction_df,
            feature_columns=[
                "feature_a",
                "feature_b",
            ],
        )