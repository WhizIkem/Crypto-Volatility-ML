from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def fit_gradient_boosting_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """
    Fit a Gradient Boosting regression model using training data only.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if n_estimators < 1:
        raise ValueError(
            "n_estimators must be at least 1"
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero"
        )

    if max_depth < 1:
        raise ValueError(
            "max_depth must be at least 1"
        )

    if min_samples_leaf < 1:
        raise ValueError(
            "min_samples_leaf must be at least 1"
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
            "Missing Gradient Boosting columns: "
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
            "Gradient Boosting training data cannot contain "
            "missing values"
        )

    if not np.isfinite(
        x_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Gradient Boosting features must contain only "
            "finite values"
        )

    if not np.isfinite(
        y_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Gradient Boosting target must contain only "
            "finite values"
        )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        loss="squared_error",
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def predict_gradient_boosting(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """
    Generate Gradient Boosting forecasts.
    """
    missing_columns = (
        set(feature_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Gradient Boosting prediction columns: "
            f"{sorted(missing_columns)}"
        )

    x = df[
        feature_columns
    ].copy()

    if x.isna().any().any():
        raise ValueError(
            "Gradient Boosting prediction data cannot contain "
            "missing values"
        )

    if not np.isfinite(
        x.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Gradient Boosting prediction features must contain "
            "only finite values"
        )

    predictions = model.predict(
        x
    )

    return pd.Series(
        predictions,
        index=df.index,
        name="gradient_boosting_prediction",
        dtype=float,
    )