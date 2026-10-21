import numpy as np
import pandas as pd
import pytest

from crypto_volatility.features.selection import (
    calculate_target_correlations,
    find_high_correlation_pairs,
)


def make_selection_data() -> pd.DataFrame:
    x = np.arange(
        1,
        101,
        dtype=float,
    )

    rng = np.random.default_rng(
        42
    )

    return pd.DataFrame(
        {
            "strong_feature": x,
            "duplicate_feature": x * 2,
            "weak_feature": rng.normal(
                size=len(x)
            ),
            "target": x + rng.normal(
                scale=0.1,
                size=len(x),
            ),
        }
    )


def test_target_correlations_rank_strong_feature_highly():
    df = make_selection_data()

    result = calculate_target_correlations(
        df,
        feature_columns=[
            "strong_feature",
            "weak_feature",
        ],
        target_column="target",
    )

    assert (
        result.iloc[0]["feature"]
        == "strong_feature"
    )

    assert (
        result.iloc[0][
            "absolute_target_correlation"
        ]
        > 0.99
    )


def test_high_correlation_pairs_detects_duplicate():
    df = make_selection_data()

    result = find_high_correlation_pairs(
        df,
        feature_columns=[
            "strong_feature",
            "duplicate_feature",
            "weak_feature",
        ],
        threshold=0.90,
    )

    matching = result[
        (
            result["feature_1"]
            == "strong_feature"
        )
        & (
            result["feature_2"]
            == "duplicate_feature"
        )
    ]

    assert len(matching) == 1

    assert matching.iloc[0][
        "absolute_correlation"
    ] == pytest.approx(
        1.0
    )


def test_high_correlation_pairs_can_return_empty_result():
    rng = np.random.default_rng(
        42
    )

    df = pd.DataFrame(
        {
            "feature_a": rng.normal(
                size=200
            ),
            "feature_b": rng.normal(
                size=200
            ),
        }
    )

    result = find_high_correlation_pairs(
        df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        threshold=0.99,
    )

    assert result.empty


def test_target_correlations_rejects_missing_columns():
    df = make_selection_data()

    with pytest.raises(
        ValueError,
        match="Missing feature-selection columns",
    ):
        calculate_target_correlations(
            df,
            feature_columns=[
                "does_not_exist",
            ],
            target_column="target",
        )


def test_high_correlation_pairs_rejects_invalid_threshold():
    df = make_selection_data()

    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1",
    ):
        find_high_correlation_pairs(
            df,
            feature_columns=[
                "strong_feature",
            ],
            threshold=1.5,
        )