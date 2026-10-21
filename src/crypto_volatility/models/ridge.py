from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_ridge_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha: float = 1.0,
) -> Pipeline:
    """
    Fit a Ridge regression model using training data only.

    Feature standardization is performed inside the sklearn
    pipeline. The StandardScaler is therefore fitted exclusively
    on the training partition supplied to this function.

    Parameters
    ----------
    train_df:
        Leakage-safe training partition.

    feature_columns:
        Predictor columns used by the model.

    target_column:
        Forecast target column.

    alpha:
        Ridge L2 regularization strength. Must be positive.

    Returns
    -------
    Pipeline
        Fitted StandardScaler + Ridge pipeline.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be greater than zero"
        )

    required_columns = (
        set(feature_columns)
        | {target_column}
    )

    missing_columns = (
        required_columns
        - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Ridge columns: "
            f"{sorted(missing_columns)}"
        )

    x_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        target_column
    ].copy()

    if (
        x_train.isna().any().any()
        or y_train.isna().any()
    ):
        raise ValueError(
            "Ridge training data cannot contain missing values"
        )

    if not np.isfinite(
        x_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Ridge features must contain only finite values"
        )

    if not np.isfinite(
        y_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Ridge target must contain only finite values"
        )

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    return pipeline


def predict_ridge(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """
    Generate Ridge forecasts from a fitted pipeline.

    Parameters
    ----------
    model:
        Fitted Ridge pipeline.

    df:
        Dataset containing model predictors.

    feature_columns:
        Predictor columns in the same order used during fitting.

    Returns
    -------
    pd.Series
        Model forecasts aligned with the input DataFrame index.
    """
    missing_columns = (
        set(feature_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Ridge prediction columns: "
            f"{sorted(missing_columns)}"
        )

    x = df[
        feature_columns
    ].copy()

    if x.isna().any().any():
        raise ValueError(
            "Ridge prediction data cannot contain missing values"
        )

    if not np.isfinite(
        x.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Ridge prediction features must contain only "
            "finite values"
        )

    predictions = model.predict(
        x
    )

    return pd.Series(
        predictions,
        index=df.index,
        name="ridge_prediction",
        dtype=float,
    )