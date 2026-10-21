from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_volatility.features.build_features import (
    TARGET_COLUMN,
)


def calculate_target_correlations(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Calculate training-only Pearson correlations between candidate
    features and the forecasting target.

    This function must only be called on training data during
    feature-selection research.

    Parameters
    ----------
    train_df:
        Leakage-safe training partition.

    feature_columns:
        Candidate predictor columns.

    target_column:
        Forecast target.

    Returns
    -------
    pd.DataFrame
        Feature correlations sorted by absolute correlation.
    """
    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
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
            "Missing feature-selection columns: "
            f"{sorted(missing_columns)}"
        )

    data = train_df[
        feature_columns
        + [target_column]
    ].copy()

    if data.isna().any().any():
        raise ValueError(
            "Feature-selection data cannot contain missing values"
        )

    if not np.isfinite(
        data.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Feature-selection data must contain only finite values"
        )

    correlations = (
        data[feature_columns]
        .corrwith(
            data[target_column]
        )
    )

    result = pd.DataFrame(
        {
            "feature": correlations.index,
            "target_correlation": correlations.values,
        }
    )

    result[
        "absolute_target_correlation"
    ] = (
        result["target_correlation"]
        .abs()
    )

    return (
        result
        .sort_values(
            "absolute_target_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def find_high_correlation_pairs(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Find highly correlated feature pairs using training data only.

    This is intended to identify potentially redundant predictors,
    not to automatically remove features.

    Parameters
    ----------
    train_df:
        Leakage-safe training partition.

    feature_columns:
        Candidate predictor columns.

    threshold:
        Minimum absolute pairwise correlation to report.

    Returns
    -------
    pd.DataFrame
        Highly correlated feature pairs sorted by absolute
        correlation.
    """
    if not 0 < threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty"
        )

    missing_columns = (
        set(feature_columns)
        - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing feature-selection columns: "
            f"{sorted(missing_columns)}"
        )

    feature_data = train_df[
        feature_columns
    ].copy()

    if feature_data.isna().any().any():
        raise ValueError(
            "Feature-selection data cannot contain missing values"
        )

    if not np.isfinite(
        feature_data.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Feature-selection data must contain only finite values"
        )

    correlation_matrix = (
        feature_data.corr()
    )

    rows = []

    for left_index, left_feature in enumerate(
        feature_columns
    ):
        for right_feature in feature_columns[
            left_index + 1:
        ]:
            correlation = float(
                correlation_matrix.loc[
                    left_feature,
                    right_feature,
                ]
            )

            if abs(correlation) >= threshold:
                rows.append(
                    {
                        "feature_1": left_feature,
                        "feature_2": right_feature,
                        "correlation": correlation,
                        "absolute_correlation": abs(
                            correlation
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )