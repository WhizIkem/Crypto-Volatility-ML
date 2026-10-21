import numpy as np
import pandas as pd
import pytest

from crypto_volatility.evaluation.walk_forward import (
    purged_time_series_splits,
    tune_gradient_boosting,
    tune_lasso_alpha,
    tune_random_forest,
    tune_ridge_alpha,
)


def make_dataset(
    rows: int = 300,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        42
    )

    feature_a = rng.normal(
        size=rows
    )

    feature_b = rng.normal(
        size=rows
    )

    target = (
        0.8 * feature_a
        - 0.3 * feature_b
        + rng.normal(
            scale=0.1,
            size=rows,
        )
    )

    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "target": target,
        }
    )


def test_purged_time_series_splits_are_chronological():
    splits = purged_time_series_splits(
        n_rows=300,
        n_splits=5,
        gap=1,
    )

    for train_index, validation_index in splits:
        assert (
            train_index.max()
            < validation_index.min()
        )


def test_purged_time_series_splits_include_gap():
    splits = purged_time_series_splits(
        n_rows=300,
        n_splits=5,
        gap=1,
    )

    for train_index, validation_index in splits:
        assert (
            validation_index.min()
            - train_index.max()
            == 2
        )


def test_training_window_expands():
    splits = purged_time_series_splits(
        n_rows=300,
        n_splits=5,
        gap=1,
    )

    train_lengths = [
        len(train_index)
        for train_index, _ in splits
    ]

    assert train_lengths == sorted(
        train_lengths
    )

    assert len(
        set(train_lengths)
    ) == len(train_lengths)


def test_tune_ridge_alpha_returns_candidate_alpha():
    df = make_dataset()

    result = tune_ridge_alpha(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        alphas=[
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        n_splits=4,
        gap=1,
    )

    assert result.best_alpha in {
        0.01,
        0.1,
        1.0,
        10.0,
    }


def test_tune_ridge_alpha_records_every_fold():
    df = make_dataset()

    result = tune_ridge_alpha(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        alphas=[
            0.1,
            1.0,
            10.0,
        ],
        n_splits=4,
        gap=1,
    )

    assert len(
        result.fold_metrics
    ) == 12

    assert len(
        result.summary
    ) == 3


def test_tune_ridge_alpha_rejects_invalid_alpha():
    df = make_dataset()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        tune_ridge_alpha(
            train_df=df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            alphas=[
                0.0,
                1.0,
            ],
        )

def test_tune_lasso_alpha_returns_candidate_alpha():
    df = make_dataset()

    result = tune_lasso_alpha(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        alphas=[
            0.0001,
            0.001,
            0.01,
            0.1,
        ],
        n_splits=4,
        gap=1,
    )

    assert result.best_alpha in {
        0.0001,
        0.001,
        0.01,
        0.1,
    }


def test_tune_lasso_alpha_records_every_fold():
    df = make_dataset()

    result = tune_lasso_alpha(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        alphas=[
            0.001,
            0.01,
            0.1,
        ],
        n_splits=4,
        gap=1,
    )

    assert len(
        result.fold_metrics
    ) == 12

    assert len(
        result.summary
    ) == 3


def test_tune_lasso_alpha_rejects_invalid_alpha():
    df = make_dataset()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        tune_lasso_alpha(
            train_df=df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            alphas=[
                0.0,
                0.01,
            ],
        )

def test_tune_random_forest_returns_candidate_parameters():
    df = make_dataset()

    result = tune_random_forest(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        max_depth_values=[
            3,
            5,
        ],
        min_samples_leaf_values=[
            1,
            5,
        ],
        n_estimators=30,
        n_splits=3,
        gap=1,
    )

    assert result.best_max_depth in {
        3,
        5,
    }

    assert result.best_min_samples_leaf in {
        1,
        5,
    }


def test_tune_random_forest_records_every_fold():
    df = make_dataset()

    result = tune_random_forest(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        max_depth_values=[
            3,
            5,
        ],
        min_samples_leaf_values=[
            1,
            5,
        ],
        n_estimators=20,
        n_splits=3,
        gap=1,
    )

    assert len(
        result.fold_metrics
    ) == 12

    assert len(
        result.summary
    ) == 4


def test_tune_random_forest_rejects_invalid_leaf_size():
    df = make_dataset()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        tune_random_forest(
            train_df=df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            max_depth_values=[
                3,
            ],
            min_samples_leaf_values=[
                0,
            ],
        )

def test_tune_gradient_boosting_returns_candidate_parameters():
    df = make_dataset()

    result = tune_gradient_boosting(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        learning_rates=[
            0.03,
            0.05,
        ],
        max_depth_values=[
            1,
            2,
        ],
        min_samples_leaf_values=[
            5,
            10,
        ],
        n_estimators=30,
        n_splits=3,
        gap=1,
    )

    assert result.best_learning_rate in {
        0.03,
        0.05,
    }

    assert result.best_max_depth in {
        1,
        2,
    }

    assert result.best_min_samples_leaf in {
        5,
        10,
    }


def test_tune_gradient_boosting_records_every_fold():
    df = make_dataset()

    result = tune_gradient_boosting(
        train_df=df,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        target_column="target",
        learning_rates=[
            0.03,
            0.05,
        ],
        max_depth_values=[
            1,
            2,
        ],
        min_samples_leaf_values=[
            5,
            10,
        ],
        n_estimators=20,
        n_splits=3,
        gap=1,
    )

    assert len(
        result.fold_metrics
    ) == 24

    assert len(
        result.summary
    ) == 8


def test_tune_gradient_boosting_rejects_invalid_learning_rate():
    df = make_dataset()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        tune_gradient_boosting(
            train_df=df,
            feature_columns=[
                "feature_a",
            ],
            target_column="target",
            learning_rates=[
                0.0,
            ],
            max_depth_values=[
                2,
            ],
            min_samples_leaf_values=[
                5,
            ],
        )