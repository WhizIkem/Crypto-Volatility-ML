import pandas as pd
import pytest

from crypto_volatility.evaluation.robust_validation import (
    expanding_validation_splits,
)


def make_data(
    rows: int = 100,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(
                rows
            ),
            "value": range(
                rows
            ),
        }
    )


def test_expanding_validation_split_boundaries():
    df = make_data()

    splits = expanding_validation_splits(
        df,
        initial_train_size=60,
        validation_size=10,
        step_size=10,
        purge_rows=1,
    )

    first = splits[0]

    assert first.fold == 1

    assert first.train[
        "row_id"
    ].tolist() == list(
        range(
            0,
            60,
        )
    )

    assert first.context[
        "row_id"
    ].tolist() == [
        60
    ]

    assert first.validation[
        "row_id"
    ].tolist() == list(
        range(
            61,
            71,
        )
    )


def test_training_window_expands():
    df = make_data()

    splits = expanding_validation_splits(
        df,
        initial_train_size=60,
        validation_size=10,
        step_size=10,
        purge_rows=1,
    )

    first = splits[0]
    second = splits[1]

    assert len(
        first.train
    ) == 60

    assert len(
        second.train
    ) == 70

    assert second.train[
        "row_id"
    ].iloc[-1] == 69


def test_validation_window_moves_forward():
    df = make_data()

    splits = expanding_validation_splits(
        df,
        initial_train_size=60,
        validation_size=10,
        step_size=10,
        purge_rows=1,
    )

    first = splits[0]
    second = splits[1]

    assert first.validation[
        "row_id"
    ].iloc[0] == 61

    assert second.validation[
        "row_id"
    ].iloc[0] == 71


def test_purged_rows_are_not_scored():
    df = make_data()

    splits = expanding_validation_splits(
        df,
        initial_train_size=60,
        validation_size=10,
        step_size=10,
        purge_rows=1,
    )

    first = splits[0]

    train_ids = set(
        first.train[
            "row_id"
        ]
    )

    context_ids = set(
        first.context[
            "row_id"
        ]
    )

    validation_ids = set(
        first.validation[
            "row_id"
        ]
    )

    assert train_ids.isdisjoint(
        context_ids
    )

    assert context_ids.isdisjoint(
        validation_ids
    )

    assert train_ids.isdisjoint(
        validation_ids
    )


def test_default_step_matches_validation_size():
    df = make_data()

    splits = expanding_validation_splits(
        df,
        initial_train_size=60,
        validation_size=10,
        purge_rows=1,
    )

    assert splits[0].validation[
        "row_id"
    ].iloc[0] == 61

    assert splits[1].validation[
        "row_id"
    ].iloc[0] == 71


def test_rejects_insufficient_rows():
    df = make_data(
        rows=20
    )

    with pytest.raises(
        ValueError,
        match="Not enough rows",
    ):
        expanding_validation_splits(
            df,
            initial_train_size=15,
            validation_size=10,
            purge_rows=1,
        )


def test_rejects_invalid_initial_train_size():
    df = make_data()

    with pytest.raises(
        ValueError,
        match="initial_train_size",
    ):
        expanding_validation_splits(
            df,
            initial_train_size=0,
            validation_size=10,
        )


def test_rejects_invalid_validation_size():
    df = make_data()

    with pytest.raises(
        ValueError,
        match="validation_size",
    ):
        expanding_validation_splits(
            df,
            initial_train_size=60,
            validation_size=0,
        )


def test_rejects_negative_purge():
    df = make_data()

    with pytest.raises(
        ValueError,
        match="purge_rows",
    ):
        expanding_validation_splits(
            df,
            initial_train_size=60,
            validation_size=10,
            purge_rows=-1,
        )