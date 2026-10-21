import numpy as np
import pandas as pd
import pytest

from crypto_volatility.models.har import (
    HAR_FEATURES,
    fit_har_model,
    predict_har,
)


def make_har_data(
    rows: int = 200,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        42
    )

    daily = rng.normal(
        loc=-2.5,
        scale=0.4,
        size=rows,
    )

    weekly = rng.normal(
        loc=-2.5,
        scale=0.3,
        size=rows,
    )

    monthly = rng.normal(
        loc=-2.5,
        scale=0.2,
        size=rows,
    )

    target = (
        -0.5
        + 0.20 * daily
        + 0.50 * weekly
        + 0.10 * monthly
    )

    return pd.DataFrame(
        {
            "log_gk_volatility": daily,
            "har_vol_weekly": weekly,
            "har_vol_monthly": monthly,
            "target": target,
        }
    )


def test_fit_har_model_estimates_expected_coefficients():
    df = make_har_data()

    model = fit_har_model(
        df,
        target_column="target",
    )

    assert model.params[
        "const"
    ] == pytest.approx(
        -0.5,
        abs=1e-10,
    )

    assert model.params[
        "log_gk_volatility"
    ] == pytest.approx(
        0.20,
        abs=1e-10,
    )

    assert model.params[
        "har_vol_weekly"
    ] == pytest.approx(
        0.50,
        abs=1e-10,
    )

    assert model.params[
        "har_vol_monthly"
    ] == pytest.approx(
        0.10,
        abs=1e-10,
    )


def test_predict_har_returns_expected_length():
    df = make_har_data()

    model = fit_har_model(
        df,
        target_column="target",
    )

    predictions = predict_har(
        model,
        df,
    )

    assert len(predictions) == len(df)


def test_predict_har_matches_known_relationship():
    df = make_har_data()

    model = fit_har_model(
        df,
        target_column="target",
    )

    predictions = predict_har(
        model,
        df,
    )

    expected = df[
        "target"
    ]

    assert np.allclose(
        predictions,
        expected,
        atol=1e-10,
    )


def test_fit_har_model_rejects_missing_columns():
    df = make_har_data().drop(
        columns=[
            "har_vol_weekly",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Missing HAR columns",
    ):
        fit_har_model(
            df,
            target_column="target",
        )


def test_predict_har_rejects_missing_values():
    df = make_har_data()

    model = fit_har_model(
        df,
        target_column="target",
    )

    prediction_df = df[
        HAR_FEATURES
    ].copy()

    prediction_df.loc[
        0,
        "har_vol_weekly",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="cannot contain missing values",
    ):
        predict_har(
            model,
            prediction_df,
        )