from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def fit_random_forest_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: float | str | None = 1.0,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Fit a Random Forest regression model using training data only.

    Random Forest does not require feature standardization.

    Parameters
    ----------
    train_df:
        Leakage-safe training partition.

    feature_columns:
        Predictor columns.

    target_column:
        Forecast target.

    n_estimators:
        Number of trees.

    max_depth:
        Maximum depth of each tree. None allows unrestricted depth.

    min_samples_leaf:
        Minimum number of observations required in each leaf.

    max_features:
        Number or fraction of features considered at each split.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Fitted Random Forest model.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if n_estimators < 1:
        raise ValueError(
            "n_estimators must be at least 1"
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
            "Missing Random Forest columns: "
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
            "Random Forest training data cannot contain "
            "missing values"
        )

    if not np.isfinite(
        x_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Random Forest features must contain only "
            "finite values"
        )

    if not np.isfinite(
        y_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Random Forest target must contain only "
            "finite values"
        )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def predict_random_forest(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """
    Generate Random Forest forecasts.
    """
    missing_columns = (
        set(feature_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Random Forest prediction columns: "
            f"{sorted(missing_columns)}"
        )

    x = df[
        feature_columns
    ].copy()

    if x.isna().any().any():
        raise ValueError(
            "Random Forest prediction data cannot contain "
            "missing values"
        )

    if not np.isfinite(
        x.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Random Forest prediction features must contain "
            "only finite values"
        )

    predictions = model.predict(
        x
    )

    return pd.Series(
        predictions,
        index=df.index,
        name="random_forest_prediction",
        dtype=float,
    )