from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_volatility.features.build_features import (
    CANDIDATE_FEATURES,
    TARGET_COLUMN,
)


def build_model_dataset(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a model-ready supervised dataset.

    Rows with unavailable predictors or target values are removed.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame.

    feature_columns:
        Predictor columns to retain. If None, use the canonical
        candidate feature list.

    Returns
    -------
    pd.DataFrame
        Chronologically ordered dataset containing timestamp,
        symbol when available, predictors, and target.

    Raises
    ------
    ValueError
        If required columns are missing or invalid values remain.
    """
    if feature_columns is None:
        feature_columns = CANDIDATE_FEATURES

    required_columns = {
        "timestamp",
        TARGET_COLUMN,
        *feature_columns,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing model-dataset columns: "
            f"{sorted(missing_columns)}"
        )

    metadata_columns = [
        "timestamp",
    ]

    if "symbol" in df.columns:
        metadata_columns.append(
            "symbol"
        )

    selected_columns = (
        metadata_columns
        + feature_columns
        + [TARGET_COLUMN]
    )

    result = (
        df[selected_columns]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    result = result.dropna(
        subset=(
            feature_columns
            + [TARGET_COLUMN]
        )
    )

    numeric_columns = (
        result[
            feature_columns
            + [TARGET_COLUMN]
        ]
        .select_dtypes(
            include="number"
        )
    )

    infinite_count = int(
        np.isinf(
            numeric_columns
        )
        .sum()
        .sum()
    )

    if infinite_count > 0:
        raise ValueError(
            "Model dataset contains "
            f"{infinite_count} infinite values"
        )

    if (
        result[
            feature_columns
            + [TARGET_COLUMN]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Model dataset still contains missing values"
        )

    if not result[
        "timestamp"
    ].is_monotonic_increasing:
        raise ValueError(
            "Model dataset timestamps are not sorted"
        )

    return result