from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.utils.config import load_config


def load_processed_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Load model-ready processed datasets.
    """
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config["interval"]
    symbols = data_config["symbols"]

    datasets = {}

    for asset, symbol in symbols.items():
        path = processed_dir / (
            f"{symbol.lower()}_{interval}_model.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        datasets[asset] = pd.read_parquet(
            path
        )

    return datasets


def main() -> None:
    """
    Inspect chronological train, validation, and test splits.
    """
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    for asset, df in datasets.items():
        split = (
            chronological_train_validation_test_split(
                df,
                train_fraction=0.70,
                validation_fraction=0.15,
            )
        )

        print(f"\n{asset}")

        print(
            "Train:",
            len(split.train),
            "|",
            split.train["timestamp"].min(),
            "→",
            split.train["timestamp"].max(),
        )

        print(
            "Validation:",
            len(split.validation),
            "|",
            split.validation["timestamp"].min(),
            "→",
            split.validation["timestamp"].max(),
        )

        print(
            "Test:",
            len(split.test),
            "|",
            split.test["timestamp"].min(),
            "→",
            split.test["timestamp"].max(),
        )

        print(
            "Total:",
            (
                len(split.train)
                + len(split.validation)
                + len(split.test)
            ),
        )


if __name__ == "__main__":
    main()