from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_volatility.data.preprocessing import (
    TimeSeriesSplit,
    chronological_train_validation_test_split,
    standardize_feature_splits,
)


@dataclass(frozen=True)
class ModelInputs:
    """
    Container for model-ready train, validation, and test arrays.
    """

    x_train: pd.DataFrame
    y_train: pd.Series

    x_validation: pd.DataFrame
    y_validation: pd.Series

    x_test: pd.DataFrame
    y_test: pd.Series

    train_timestamps: pd.Series
    validation_timestamps: pd.Series
    test_timestamps: pd.Series


def prepare_model_inputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    scale_features: bool = False,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ModelInputs:
    """
    Prepare chronological model inputs from a model-ready dataset.

    Parameters
    ----------
    df:
        Model-ready dataset.

    feature_columns:
        Predictor columns used by the model.

    target_column:
        Forecasting target column.

    scale_features:
        Whether to standardize predictors using training data only.

    train_fraction:
        Fraction assigned to training.

    validation_fraction:
        Fraction assigned to validation.

    Returns
    -------
    ModelInputs
        Predictor matrices, target vectors, and timestamps.
    """
    required_columns = {
        "timestamp",
        target_column,
        *feature_columns,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing model input columns: "
            f"{sorted(missing_columns)}"
        )

    split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    )

    if scale_features:
        scaled_split = standardize_feature_splits(
            split,
            feature_columns=feature_columns,
        )

        split = TimeSeriesSplit(
            train=scaled_split.train,
            validation=scaled_split.validation,
            test=scaled_split.test,
        )

    return ModelInputs(
        x_train=split.train[
            feature_columns
        ].copy(),
        y_train=split.train[
            target_column
        ].copy(),
        x_validation=split.validation[
            feature_columns
        ].copy(),
        y_validation=split.validation[
            target_column
        ].copy(),
        x_test=split.test[
            feature_columns
        ].copy(),
        y_test=split.test[
            target_column
        ].copy(),
        train_timestamps=split.train[
            "timestamp"
        ].copy(),
        validation_timestamps=split.validation[
            "timestamp"
        ].copy(),
        test_timestamps=split.test[
            "timestamp"
        ].copy(),
    )