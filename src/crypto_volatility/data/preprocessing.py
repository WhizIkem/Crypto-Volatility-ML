from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TimeSeriesSplit:
    """
    Container for chronological train, validation, and test datasets.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ScaledTimeSeriesSplit:
    """
    Container for scaled train, validation, and test datasets.

    The fitted scaler is retained so it can later be reused
    for inference on unseen data.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TimeSeriesSplit:
    """
    Split a time-series DataFrame chronologically.

    The remaining fraction after train and validation is assigned
    to the test set.

    Parameters
    ----------
    df:
        Chronologically ordered model-ready dataset.

    train_fraction:
        Fraction of observations assigned to training.

    validation_fraction:
        Fraction of observations assigned to validation.

    Returns
    -------
    TimeSeriesSplit
        Chronological train, validation, and test partitions.

    Raises
    ------
    ValueError
        If the dataset is empty, timestamps are missing or unsorted,
        split fractions are invalid, or any partition would be empty.
    """
    if df.empty:
        raise ValueError(
            "Cannot split an empty dataset"
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            "timestamp column is required"
        )

    if df["timestamp"].isna().any():
        raise ValueError(
            "timestamp column cannot contain missing values"
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Dataset must be sorted chronologically"
        )

    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be between 0 and 1"
        )

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    if (
        train_fraction
        + validation_fraction
        >= 1
    ):
        raise ValueError(
            "train_fraction + validation_fraction "
            "must be less than 1"
        )

    n_rows = len(df)

    train_end = int(
        n_rows * train_fraction
    )

    validation_end = int(
        n_rows
        * (
            train_fraction
            + validation_fraction
        )
    )

    if (
        train_end == 0
        or validation_end <= train_end
        or validation_end >= n_rows
    ):
        raise ValueError(
            "Split fractions produce an empty partition"
        )

    train = (
        df.iloc[:train_end]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        df.iloc[
            train_end:validation_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df.iloc[
            validation_end:
        ]
        .copy()
        .reset_index(drop=True)
    )

    return TimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
    )


def standardize_feature_splits(
    split: TimeSeriesSplit,
    feature_columns: list[str],
) -> ScaledTimeSeriesSplit:
    """
    Standardize predictor columns using training data only.

    The StandardScaler is fitted exclusively on the training set.
    The fitted training statistics are then applied unchanged to
    validation and test data.

    Non-feature columns, including timestamp, symbol, and target,
    are preserved without transformation.

    Parameters
    ----------
    split:
        Chronological train, validation, and test partitions.

    feature_columns:
        Predictor columns to standardize.

    Returns
    -------
    ScaledTimeSeriesSplit
        Scaled partitions together with the fitted StandardScaler.

    Raises
    ------
    ValueError
        If no features are supplied, required columns are missing,
        or missing values exist in the selected features.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    for partition_name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        missing_columns = (
            set(feature_columns)
            - set(partition.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{partition_name} is missing feature columns: "
                f"{sorted(missing_columns)}"
            )

        if (
            partition[feature_columns]
            .isna()
            .any()
            .any()
        ):
            raise ValueError(
                f"{partition_name} contains missing feature values"
            )

    scaler = StandardScaler()

    train = split.train.copy()
    validation = split.validation.copy()
    test = split.test.copy()

    train[feature_columns] = (
        scaler.fit_transform(
            train[feature_columns]
        )
    )

    validation[feature_columns] = (
        scaler.transform(
            validation[feature_columns]
        )
    )

    test[feature_columns] = (
        scaler.transform(
            test[feature_columns]
        )
    )

    return ScaledTimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
        scaler=scaler,
    )