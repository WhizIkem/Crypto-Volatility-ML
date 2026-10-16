from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.features.build_features import (
    CANDIDATE_FEATURES,
    TARGET_COLUMN,
    build_feature_datasets,
)
from crypto_volatility.features.model_dataset import (
    build_model_dataset,
)
from crypto_volatility.utils.config import load_config


def load_clean_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Load cleaned Binance datasets from data/interim.
    """
    data_config = config["data"]

    interim_dir = Path(
        data_config["interim_dir"]
    )

    interval = data_config["interval"]
    symbols = data_config["symbols"]

    datasets = {}

    for asset, symbol in symbols.items():
        path = interim_dir / (
            f"{symbol.lower()}_{interval}_clean.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Clean dataset not found: {path}"
            )

        datasets[asset] = pd.read_parquet(
            path
        )

    return datasets


def validate_feature_output(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """
    Validate the generated feature datasets.
    """
    for asset, df in datasets.items():
        missing_columns = [
            column
            for column in CANDIDATE_FEATURES
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{asset} is missing feature columns: "
                f"{missing_columns}"
            )

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"{asset} is missing target column"
            )

        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"{asset} timestamps are not sorted"
            )

        numeric_columns = (
            df[CANDIDATE_FEATURES]
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
                f"{asset} contains "
                f"{infinite_count} infinite feature values"
            )

        expected_target = (
            df["log_gk_volatility"]
            .shift(-1)
        )

        if not df[TARGET_COLUMN].equals(
            expected_target
        ):
            raise ValueError(
                f"{asset} target alignment is incorrect"
            )

        print(
            asset,
            "| rows:",
            len(df),
            "| feature columns:",
            len(CANDIDATE_FEATURES),
            "| infinite values:",
            infinite_count,
            "| target missing:",
            df[TARGET_COLUMN].isna().sum(),
        )


def print_boundary_missingness(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """
    Print missing-value counts for candidate features.
    """
    for asset, df in datasets.items():
        print(f"\n{asset}")

        missing_counts = (
            df[CANDIDATE_FEATURES + [TARGET_COLUMN]]
            .isna()
            .sum()
        )

        missing_counts = missing_counts[
            missing_counts > 0
        ]

        print(
            missing_counts.to_string()
        )


def main() -> None:
    """
    Build and validate candidate feature datasets.
    """
    config = load_config()

    print(
        "Loading cleaned Binance datasets..."
    )

    datasets = load_clean_datasets(
        config
    )

    print(
        "Building candidate features..."
    )

    feature_datasets = build_feature_datasets(
        datasets
    )

    print(
        "Validating generated features..."
    )

    validate_feature_output(
        feature_datasets
    )

    print(
        "\nBoundary missingness:"
    )

    print_boundary_missingness(
        feature_datasets
    )

    print(
        "\nFeature pipeline validation complete."
    )

    processed_dir = Path(
    config["data"]["processed_dir"]
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nBuilding model-ready datasets..."
    )

    for asset, df in feature_datasets.items():
        model_dataset = build_model_dataset(
            df
        )

        symbol = config["data"]["symbols"][
            asset
        ]

        output_path = (
            processed_dir
            / (
                f"{symbol.lower()}_"
                f"{config['data']['interval']}_"
                "model.parquet"
            )
        )

        model_dataset.to_parquet(
            output_path,
            index=False,
        )

        print(
            asset,
            "| model rows:",
            len(model_dataset),
            "| start:",
            model_dataset["timestamp"].min(),
            "| end:",
            model_dataset["timestamp"].max(),
            "| saved:",
            output_path,
        )


if __name__ == "__main__":
    main()