from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.evaluation.robust_validation import (
    expanding_validation_splits,
)
from crypto_volatility.utils.config import load_config

INITIAL_TRAIN_SIZE = 1200
VALIDATION_SIZE = 120
STEP_SIZE = 120
PURGE_ROWS = 1


def load_btc_model_dataset(
    config: dict,
) -> pd.DataFrame:
    """
    Load the BTC processed model dataset.

    BTC is used only to inspect the common Phase 7 date boundaries.
    The four model datasets are expected to share the same aligned
    modelling period.
    """
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config["interval"]

    symbol = data_config[
        "symbols"
    ][
        "BTC"
    ]

    path = processed_dir / (
        f"{symbol.lower()}_{interval}_model.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    return pd.read_parquet(
        path
    )


def build_development_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
]:
    """
    Restrict Phase 7 validation to the development period.

    The original outer 70/15/15 split is preserved.

    Phase 7 may use the original training and validation history, but
    must not use the final test partition.

    The purge row immediately before the final test set is also excluded
    from Phase 7 scoring because its t+1 target crosses into the sealed
    test period.
    """
    outer_split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=1,
        )
    )

    if outer_split.test.empty:
        raise ValueError(
            "Outer test split is empty"
        )

    final_test_start = (
        outer_split.test[
            "timestamp"
        ].iloc[0]
    )

    matching_positions = df.index[
        df[
            "timestamp"
        ].eq(
            final_test_start
        )
    ].tolist()

    if len(
        matching_positions
    ) != 1:
        raise ValueError(
            "Could not uniquely identify the final test start row"
        )

    test_start_position = (
        matching_positions[
            0
        ]
    )

    development_end = (
        test_start_position
        - PURGE_ROWS
    )

    if development_end <= 0:
        raise ValueError(
            "Development period is empty"
        )

    development_df = (
        df
        .iloc[
            :development_end
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        development_df,
        final_test_start,
    )


def build_split_summary(
    splits,
) -> pd.DataFrame:
    """
    Convert robust-validation folds into a readable table.
    """
    rows = []

    for split in splits:
        context_date = None

        if not split.context.empty:
            context_date = (
                split.context[
                    "timestamp"
                ].iloc[-1]
            )

        rows.append(
            {
                "fold": split.fold,
                "train_rows": len(
                    split.train
                ),
                "train_start": split.train[
                    "timestamp"
                ].iloc[0],
                "train_end": split.train[
                    "timestamp"
                ].iloc[-1],
                "context_rows": len(
                    split.context
                ),
                "context_date": context_date,
                "validation_rows": len(
                    split.validation
                ),
                "validation_start": (
                    split.validation[
                        "timestamp"
                    ].iloc[0]
                ),
                "validation_end": (
                    split.validation[
                        "timestamp"
                    ].iloc[-1]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    config = load_config()

    df = load_btc_model_dataset(
        config
    )

    if "timestamp" not in df.columns:
        raise ValueError(
            "Expected 'timestamp' column in processed dataset"
        )

    df = (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    (
        development_df,
        final_test_start,
    ) = build_development_dataset(
        df
    )

    splits = expanding_validation_splits(
        development_df,
        initial_train_size=INITIAL_TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        step_size=STEP_SIZE,
        purge_rows=PURGE_ROWS,
    )

    summary = build_split_summary(
        splits
    )

    print(
        "\nFull dataset rows:",
        len(df),
    )

    print(
        "Full dataset start:",
        df[
            "timestamp"
        ].iloc[0],
    )

    print(
        "Full dataset end:",
        df[
            "timestamp"
        ].iloc[-1],
    )

    print(
        "\nPhase 7 development rows:",
        len(
            development_df
        ),
    )

    print(
        "Development start:",
        development_df[
            "timestamp"
        ].iloc[0],
    )

    print(
        "Development end:",
        development_df[
            "timestamp"
        ].iloc[-1],
    )

    print(
        "\nFinal test begins:",
        final_test_start,
    )

    print(
        "\nPhase 7 robust validation folds:\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()