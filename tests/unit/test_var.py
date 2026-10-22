import numpy as np
import pandas as pd
import pytest

from crypto_volatility.models.var import (
    extract_var_coefficients,
    fit_var_model,
    forecast_var_one_step,
)


def make_var_data(
    rows: int = 250,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        42
    )

    asset_a = np.zeros(
        rows
    )

    asset_b = np.zeros(
        rows
    )

    for index in range(
        1,
        rows,
    ):
        asset_a[index] = (
            0.6 * asset_a[
                index - 1
            ]
            + rng.normal(
                scale=0.2
            )
        )

        asset_b[index] = (
            0.3 * asset_b[
                index - 1
            ]
            + 0.4 * asset_a[
                index - 1
            ]
            + rng.normal(
                scale=0.2
            )
        )

    return pd.DataFrame(
        {
            "asset_a": asset_a,
            "asset_b": asset_b,
        }
    )


def test_fit_var_model():
    df = make_var_data()

    model = fit_var_model(
        train_df=df,
        columns=[
            "asset_a",
            "asset_b",
        ],
        maxlags=3,
        ic="aic",
    )

    assert model is not None

    assert model.k_ar >= 1


def test_forecast_var_one_step_returns_expected_columns():
    df = make_var_data()

    columns = [
        "asset_a",
        "asset_b",
    ]

    model = fit_var_model(
        train_df=df,
        columns=columns,
        maxlags=3,
        ic="aic",
    )

    forecast = forecast_var_one_step(
        model=model,
        history_df=df,
        columns=columns,
    )

    assert list(
        forecast.index
    ) == columns


def test_var_forecast_is_finite():
    df = make_var_data()

    columns = [
        "asset_a",
        "asset_b",
    ]

    model = fit_var_model(
        train_df=df,
        columns=columns,
        maxlags=3,
        ic="aic",
    )

    forecast = forecast_var_one_step(
        model=model,
        history_df=df,
        columns=columns,
    )

    assert np.isfinite(
        forecast.to_numpy()
    ).all()


def test_extract_var_coefficients_returns_expected_columns():
    df = make_var_data()

    model = fit_var_model(
        train_df=df,
        columns=[
            "asset_a",
            "asset_b",
        ],
        maxlags=3,
        ic="aic",
    )

    coefficients = (
        extract_var_coefficients(
            model
        )
    )

    assert {
        "term",
        "equation",
        "coefficient",
    }.issubset(
        coefficients.columns
    )


def test_var_rejects_missing_columns():
    df = make_var_data()

    with pytest.raises(
        ValueError,
        match="Missing VAR columns",
    ):
        fit_var_model(
            train_df=df,
            columns=[
                "does_not_exist",
            ],
        )


def test_var_rejects_missing_values():
    df = make_var_data()

    df.loc[
        0,
        "asset_a",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="cannot contain missing values",
    ):
        fit_var_model(
            train_df=df,
            columns=[
                "asset_a",
                "asset_b",
            ],
        )


def test_var_rejects_invalid_maxlags():
    df = make_var_data()

    with pytest.raises(
        ValueError,
        match="maxlags must be at least 1",
    ):
        fit_var_model(
            train_df=df,
            columns=[
                "asset_a",
                "asset_b",
            ],
            maxlags=0,
        )