from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_lasso_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha: float = 0.001,
    max_iter: int = 10_000,
) -> Pipeline:
    """
    Fit a Lasso regression model using training data only.

    Standardization is performed inside the sklearn pipeline so
    scaling statistics are learned exclusively from the supplied
    training partition.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be greater than zero"
        )

    if max_iter < 1:
        raise ValueError(
            "max_iter must be at least 1"
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
            "Missing Lasso columns: "
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
            "Lasso training data cannot contain missing values"
        )

    if not np.isfinite(
        x_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Lasso features must contain only finite values"
        )

    if not np.isfinite(
        y_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Lasso target must contain only finite values"
        )

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "lasso",
                Lasso(
                    alpha=alpha,
                    max_iter=max_iter,
                ),
            ),
        ]
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    return pipeline


def predict_lasso(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """
    Generate forecasts from a fitted Lasso pipeline.
    """
    missing_columns = (
        set(feature_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Lasso prediction columns: "
            f"{sorted(missing_columns)}"
        )

    x = df[
        feature_columns
    ].copy()

    if x.isna().any().any():
        raise ValueError(
            "Lasso prediction data cannot contain missing values"
        )

    if not np.isfinite(
        x.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Lasso prediction features must contain only "
            "finite values"
        )

    predictions = model.predict(
        x
    )

    return pd.Series(
        predictions,
        index=df.index,
        name="lasso_prediction",
        dtype=float,
    )