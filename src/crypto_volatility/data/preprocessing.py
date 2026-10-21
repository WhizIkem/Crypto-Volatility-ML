from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ScaledTimeSeriesSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_rows: int = 1,
) -> TimeSeriesSplit:
    """
    Split a supervised time-series dataset chronologically.

    Rows are never shuffled.

    A purge is applied at the train/validation and
    validation/test boundaries to prevent forward targets from
    crossing into the next evaluation period.

    For the project's one-day-ahead target:

        features at t -> target at t + 1

    purge_rows=1 ensures that the final training target does not
    belong to the validation period, and the final validation
    target does not belong to the test period.

    Parameters
    ----------
    df:
        Model-ready supervised dataset. Must contain ``timestamp``
        and already be sorted chronologically.

    train_fraction:
        Fraction of rows assigned to the initial training region
        before purging.

    validation_fraction:
        Fraction of rows assigned to the validation region before
        purging.

    purge_rows:
        Number of supervised rows removed immediately before each
        split boundary. For a one-step-ahead forecast horizon this
        should be 1.

    Returns
    -------
    TimeSeriesSplit
        Purged train, validation, and test DataFrames.
    """
    if df.empty:
        raise ValueError(
            "Cannot split an empty DataFrame"
        )

    if "timestamp" not in df.columns:
        raise ValueError(
            "DataFrame must contain a timestamp column"
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

    if purge_rows < 0:
        raise ValueError(
            "purge_rows cannot be negative"
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically"
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

    train_stop = (
        train_end
        - purge_rows
    )

    validation_stop = (
        validation_end
        - purge_rows
    )

    if train_stop <= 0:
        raise ValueError(
            "purge_rows leaves no training observations"
        )

    if validation_stop <= train_end:
        raise ValueError(
            "purge_rows leaves no validation observations"
        )

    train = (
        df.iloc[
            :train_stop
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        df.iloc[
            train_end:validation_stop
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

    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        raise ValueError(
            "Split produced an empty partition"
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
    Standardize features using statistics learned from training
    data only.

    The scaler is fit exclusively on the training partition and
    subsequently applied to validation and test data.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    for split_name, frame in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        missing_columns = (
            set(feature_columns)
            - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing feature columns: "
                f"{sorted(missing_columns)}"
            )

        feature_data = frame[
            feature_columns
        ]

        if feature_data.isna().any().any():
            raise ValueError(
                f"{split_name} feature data contains "
                "missing values"
            )

        if not np.isfinite(
            feature_data.to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                f"{split_name} feature data contains "
                "non-finite values"
            )

    scaler = StandardScaler()

    train = split.train.copy()
    validation = split.validation.copy()
    test = split.test.copy()

    train[
        feature_columns
    ] = scaler.fit_transform(
        train[
            feature_columns
        ]
    )

    validation[
        feature_columns
    ] = scaler.transform(
        validation[
            feature_columns
        ]
    )

    test[
        feature_columns
    ] = scaler.transform(
        test[
            feature_columns
        ]
    )

    return ScaledTimeSeriesSplit(
        train=train,
        validation=validation,
        test=test,
        scaler=scaler,
    )