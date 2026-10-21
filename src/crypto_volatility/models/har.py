from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from crypto_volatility.features.build_features import (
    TARGET_COLUMN,
)

HAR_FEATURES = [
    "log_gk_volatility",
    "har_vol_weekly",
    "har_vol_monthly",
]


def fit_har_model(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> RegressionResultsWrapper:
    """
    Fit a HAR-style volatility regression using OLS.

    The model uses daily, weekly, and monthly log
    Garman-Klass volatility components.

    Parameters
    ----------
    train_df:
        Training dataset containing HAR predictors and target.

    target_column:
        Forecasting target column.

    Returns
    -------
    RegressionResultsWrapper
        Fitted statsmodels OLS regression result.

    Raises
    ------
    ValueError
        If required columns are missing or training data
        contain missing values.
    """
    required_columns = {
        *HAR_FEATURES,
        target_column,
    }

    missing_columns = (
        required_columns
        - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing HAR columns: "
            f"{sorted(missing_columns)}"
        )

    model_data = train_df[
        HAR_FEATURES
        + [target_column]
    ].copy()

    if model_data.isna().any().any():
        raise ValueError(
            "HAR training data cannot contain missing values"
        )

    x_train = sm.add_constant(
        model_data[HAR_FEATURES],
        has_constant="add",
    )

    y_train = model_data[
        target_column
    ]

    model = sm.OLS(
        y_train,
        x_train,
    )

    return model.fit()


def predict_har(
    model: RegressionResultsWrapper,
    df: pd.DataFrame,
) -> pd.Series:
    """
    Generate HAR volatility forecasts.

    Parameters
    ----------
    model:
        Fitted HAR OLS model.

    df:
        Dataset containing HAR predictor columns.

    Returns
    -------
    pd.Series
        Predicted next-day log volatility.

    Raises
    ------
    ValueError
        If required predictor columns are missing or
        contain missing values.
    """
    missing_columns = (
        set(HAR_FEATURES)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing HAR prediction columns: "
            f"{sorted(missing_columns)}"
        )

    features = df[
        HAR_FEATURES
    ].copy()

    if features.isna().any().any():
        raise ValueError(
            "HAR prediction data cannot contain missing values"
        )

    x = sm.add_constant(
        features,
        has_constant="add",
    )

    predictions = model.predict(
        x
    )

    return pd.Series(
        predictions,
        index=df.index,
        name="har_prediction",
        dtype=float,
    )