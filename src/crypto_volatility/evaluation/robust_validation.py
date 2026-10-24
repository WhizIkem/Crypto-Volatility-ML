from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExpandingValidationSplit:
    """
    One leakage-safe expanding-window validation split.

    train:
        Supervised training rows.

    context:
        Purged rows that are not training or validation targets but may
        be used as observed historical context when appropriate.

    validation:
        Rows scored during this validation window.
    """

    fold: int
    train: pd.DataFrame
    context: pd.DataFrame
    validation: pd.DataFrame


def expanding_validation_splits(
    df: pd.DataFrame,
    *,
    initial_train_size: int,
    validation_size: int,
    step_size: int | None = None,
    purge_rows: int = 1,
) -> list[ExpandingValidationSplit]:
    """
    Create expanding-window validation folds.

    Example with:

        initial_train_size = 60
        validation_size = 10
        purge_rows = 1
        step_size = 10

    First fold:

        train       rows 0..59
        context     row 60
        validation  rows 61..70

    Second fold:

        train       rows 0..69
        context     row 70
        validation  rows 71..80

    Training history expands while validation moves forward.

    The context rows are excluded from supervised fitting and scoring,
    but may be used as observed historical information by models such
    as VAR or LSTM.
    """
    if df.empty:
        raise ValueError(
            "df cannot be empty"
        )

    if initial_train_size < 1:
        raise ValueError(
            "initial_train_size must be at least 1"
        )

    if validation_size < 1:
        raise ValueError(
            "validation_size must be at least 1"
        )

    if purge_rows < 0:
        raise ValueError(
            "purge_rows cannot be negative"
        )

    if step_size is None:
        step_size = validation_size

    if step_size < 1:
        raise ValueError(
            "step_size must be at least 1"
        )

    minimum_rows = (
        initial_train_size
        + purge_rows
        + validation_size
    )

    if len(df) < minimum_rows:
        raise ValueError(
            "Not enough rows for the requested "
            "train, purge, and validation sizes"
        )

    splits: list[
        ExpandingValidationSplit
    ] = []

    fold = 1
    train_end = initial_train_size

    while True:
        context_start = train_end
        context_end = (
            context_start
            + purge_rows
        )

        validation_start = context_end
        validation_end = (
            validation_start
            + validation_size
        )

        if validation_end > len(df):
            break

        train = (
            df
            .iloc[
                :train_end
            ]
            .copy()
            .reset_index(drop=True)
        )

        context = (
            df
            .iloc[
                context_start:
                context_end
            ]
            .copy()
            .reset_index(drop=True)
        )

        validation = (
            df
            .iloc[
                validation_start:
                validation_end
            ]
            .copy()
            .reset_index(drop=True)
        )

        splits.append(
            ExpandingValidationSplit(
                fold=fold,
                train=train,
                context=context,
                validation=validation,
            )
        )

        fold += 1
        train_end += step_size

    if not splits:
        raise ValueError(
            "No validation folds could be created"
        )

    return splits