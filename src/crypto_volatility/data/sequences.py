from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SequenceDatasetSplit:
    """
    Leakage-safe sequence datasets.

    The feature scaler is fitted using training rows only.
    Validation and test sequences may use earlier observed feature
    history across split boundaries, but their labels remain inside
    their own partitions.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler

@dataclass(frozen=True)
class SequenceTrainValidationSplit:
    """
    Leakage-safe sequence split for model tuning inside training data.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    feature_scaler: StandardScaler

@dataclass(frozen=True)
class ScaledSequenceTargets:
    y_train: np.ndarray
    y_validation: np.ndarray
    target_scaler: StandardScaler


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ordered tabular data into fixed-length sequences.

    Each sequence ends at row t and predicts the target stored
    on row t, which represents volatility at t + 1.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be at least 1"
        )

    required_columns = (
        set(feature_columns)
        | {target_column}
    )

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing sequence columns: "
            f"{sorted(missing_columns)}"
        )

    features = df[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    target = df[
        target_column
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        features
    ).all():
        raise ValueError(
            "Sequence features must contain only finite values"
        )

    if not np.isfinite(
        target
    ).all():
        raise ValueError(
            "Sequence target must contain only finite values"
        )

    if len(df) < sequence_length:
        raise ValueError(
            "Dataset must contain at least sequence_length rows"
        )

    x_sequences = []
    y_sequences = []

    for end_index in range(
        sequence_length - 1,
        len(df),
    ):
        start_index = (
            end_index
            - sequence_length
            + 1
        )

        x_sequences.append(
            features[
                start_index:
                end_index + 1
            ]
        )

        y_sequences.append(
            target[
                end_index
            ]
        )

    return (
        np.asarray(
            x_sequences,
            dtype=float,
        ),
        np.asarray(
            y_sequences,
            dtype=float,
        ),
    )


def build_leakage_safe_sequence_splits(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_rows: int = 1,
) -> SequenceDatasetSplit:
    """
    Build leakage-safe train, validation, and test sequences.

    The scaler is fitted only on rows belonging to the training
    partition.

    Validation and test sequences are allowed to use earlier feature
    rows as historical context. This includes purge rows because their
    market information would already be observable by forecast time.

    Only sequence targets are restricted to their respective
    train/validation/test partitions.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be at least 1"
        )

    if purge_rows < 0:
        raise ValueError(
            "purge_rows cannot be negative"
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

    required_columns = (
        set(feature_columns)
        | {target_column}
    )

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing sequence columns: "
            f"{sorted(missing_columns)}"
        )

    n_rows = len(df)

    train_boundary = int(
        n_rows
        * train_fraction
    )

    validation_boundary = int(
        n_rows
        * (
            train_fraction
            + validation_fraction
        )
    )

    train_target_end = (
        train_boundary
        - purge_rows
    )

    validation_target_start = (
        train_boundary
    )

    validation_target_end = (
        validation_boundary
        - purge_rows
    )

    test_target_start = (
        validation_boundary
    )

    if train_target_end <= 0:
        raise ValueError(
            "Training partition is empty after purging"
        )

    if (
        validation_target_end
        <= validation_target_start
    ):
        raise ValueError(
            "Validation partition is empty after purging"
        )

    if test_target_start >= n_rows:
        raise ValueError(
            "Test partition is empty"
        )

    if train_target_end < sequence_length:
        raise ValueError(
            "Training partition is shorter than sequence_length"
        )

    feature_values = df[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    target_values = df[
        target_column
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Sequence features must contain only finite values"
        )

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "Sequence target must contain only finite values"
        )

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        feature_values[
            :train_target_end
        ]
    )

    scaled_features = (
        feature_scaler.transform(
            feature_values
        )
    )

    def make_partition(
        target_start: int,
        target_end: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        x_values = []
        y_values = []

        first_target_index = max(
            target_start,
            sequence_length - 1,
        )

        for end_index in range(
            first_target_index,
            target_end,
        ):
            start_index = (
                end_index
                - sequence_length
                + 1
            )

            x_values.append(
                scaled_features[
                    start_index:
                    end_index + 1
                ]
            )

            y_values.append(
                target_values[
                    end_index
                ]
            )

        return (
            np.asarray(
                x_values,
                dtype=float,
            ),
            np.asarray(
                y_values,
                dtype=float,
            ),
        )

    x_train, y_train = make_partition(
        target_start=0,
        target_end=train_target_end,
    )

    x_validation, y_validation = (
        make_partition(
            target_start=validation_target_start,
            target_end=validation_target_end,
        )
    )

    x_test, y_test = make_partition(
        target_start=test_target_start,
        target_end=n_rows,
    )

    if len(x_train) == 0:
        raise ValueError(
            "No training sequences could be created"
        )

    if len(x_validation) == 0:
        raise ValueError(
            "No validation sequences could be created"
        )

    if len(x_test) == 0:
        raise ValueError(
            "No test sequences could be created"
        )

    return SequenceDatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
    )

def build_sequence_train_validation_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    train_fraction: float = 0.85,
    purge_rows: int = 1,
) -> SequenceTrainValidationSplit:
    """
    Build leakage-safe train and validation sequences.

    This helper is intended for model tuning inside an already
    isolated outer training partition.

    The scaler is fitted only on inner-training feature rows.

    Inner-validation sequences may use earlier observed feature
    history from the training period and purge row, but validation
    labels remain strictly inside the validation partition.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be at least 1"
        )

    if purge_rows < 0:
        raise ValueError(
            "purge_rows cannot be negative"
        )

    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be between 0 and 1"
        )

    required_columns = (
        set(feature_columns)
        | {target_column}
    )

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing sequence columns: "
            f"{sorted(missing_columns)}"
        )

    n_rows = len(df)

    train_boundary = int(
        n_rows
        * train_fraction
    )

    train_target_end = (
        train_boundary
        - purge_rows
    )

    validation_target_start = (
        train_boundary
    )

    if train_target_end <= 0:
        raise ValueError(
            "Training partition is empty after purging"
        )

    if validation_target_start >= n_rows:
        raise ValueError(
            "Validation partition is empty"
        )

    if train_target_end < sequence_length:
        raise ValueError(
            "Training partition is shorter than sequence_length"
        )

    feature_values = df[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    target_values = df[
        target_column
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Sequence features must contain only finite values"
        )

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "Sequence target must contain only finite values"
        )

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        feature_values[
            :train_target_end
        ]
    )

    scaled_features = (
        feature_scaler.transform(
            feature_values
        )
    )

    def make_partition(
        target_start: int,
        target_end: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        x_values = []
        y_values = []

        first_target_index = max(
            target_start,
            sequence_length - 1,
        )

        for end_index in range(
            first_target_index,
            target_end,
        ):
            start_index = (
                end_index
                - sequence_length
                + 1
            )

            x_values.append(
                scaled_features[
                    start_index:
                    end_index + 1
                ]
            )

            y_values.append(
                target_values[
                    end_index
                ]
            )

        return (
            np.asarray(
                x_values,
                dtype=float,
            ),
            np.asarray(
                y_values,
                dtype=float,
            ),
        )

    x_train, y_train = make_partition(
        target_start=0,
        target_end=train_target_end,
    )

    x_validation, y_validation = (
        make_partition(
            target_start=validation_target_start,
            target_end=n_rows,
        )
    )

    if len(x_train) == 0:
        raise ValueError(
            "No training sequences could be created"
        )

    if len(x_validation) == 0:
        raise ValueError(
            "No validation sequences could be created"
        )

    return SequenceTrainValidationSplit(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        feature_scaler=feature_scaler,
    )

def scale_sequence_targets(
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> ScaledSequenceTargets:
    """
    Standardize sequence targets using training targets only.
    """
    if y_train.ndim != 1:
        raise ValueError(
            "y_train must be one-dimensional"
        )

    if y_validation.ndim != 1:
        raise ValueError(
            "y_validation must be one-dimensional"
        )

    if len(y_train) == 0:
        raise ValueError(
            "y_train cannot be empty"
        )

    if len(y_validation) == 0:
        raise ValueError(
            "y_validation cannot be empty"
        )

    if not np.isfinite(
        y_train
    ).all():
        raise ValueError(
            "y_train must contain only finite values"
        )

    if not np.isfinite(
        y_validation
    ).all():
        raise ValueError(
            "y_validation must contain only finite values"
        )

    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(
        y_train.reshape(
            -1,
            1,
        )
    ).ravel()

    scaled_validation = scaler.transform(
        y_validation.reshape(
            -1,
            1,
        )
    ).ravel()

    return ScaledSequenceTargets(
        y_train=scaled_train,
        y_validation=scaled_validation,
        target_scaler=scaler,
    )


def inverse_transform_sequence_targets(
    values: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    """
    Convert standardized LSTM targets back to original units.
    """
    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional"
        )

    return (
        target_scaler
        .inverse_transform(
            values.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )