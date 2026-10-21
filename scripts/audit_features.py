from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.features.build_features import (
    CANDIDATE_FEATURES,
)
from crypto_volatility.features.selection import (
    calculate_target_correlations,
    find_high_correlation_pairs,
)
from crypto_volatility.utils.config import load_config

CORRELATION_THRESHOLD = 0.90


def load_processed_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Load processed model-ready datasets."""
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


def audit_asset(
    asset: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit candidate features using the purged training partition only.
    """
    split = chronological_train_validation_test_split(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_rows=1,
    )

    train = split.train

    target_correlations = (
        calculate_target_correlations(
            train,
            feature_columns=CANDIDATE_FEATURES,
        )
    )

    target_correlations.insert(
        0,
        "asset",
        asset,
    )

    redundancy_pairs = (
        find_high_correlation_pairs(
            train,
            feature_columns=CANDIDATE_FEATURES,
            threshold=CORRELATION_THRESHOLD,
        )
    )

    if not redundancy_pairs.empty:
        redundancy_pairs.insert(
            0,
            "asset",
            asset,
        )

    return (
        target_correlations,
        redundancy_pairs,
    )


def main() -> None:
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    correlation_frames = []
    redundancy_frames = []

    for asset, df in datasets.items():
        (
            target_correlations,
            redundancy_pairs,
        ) = audit_asset(
            asset,
            df,
        )

        correlation_frames.append(
            target_correlations
        )

        if not redundancy_pairs.empty:
            redundancy_frames.append(
                redundancy_pairs
            )

    correlations = pd.concat(
        correlation_frames,
        ignore_index=True,
    )

    mean_correlations = (
        correlations
        .groupby(
            "feature",
            as_index=False,
        )[
            "absolute_target_correlation"
        ]
        .mean()
        .rename(
            columns={
                "absolute_target_correlation":
                    "mean_absolute_target_correlation",
            }
        )
        .sort_values(
            "mean_absolute_target_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if redundancy_frames:
        redundancy = pd.concat(
            redundancy_frames,
            ignore_index=True,
        )
    else:
        redundancy = pd.DataFrame(
            columns=[
                "asset",
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    output_dir = Path(
        "reports/tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlations_path = (
        output_dir
        / "training_feature_target_correlations.csv"
    )

    mean_correlations_path = (
        output_dir
        / "training_mean_feature_target_correlations.csv"
    )

    redundancy_path = (
        output_dir
        / "training_high_correlation_pairs.csv"
    )

    correlations.to_csv(
        correlations_path,
        index=False,
    )

    mean_correlations.to_csv(
        mean_correlations_path,
        index=False,
    )

    redundancy.to_csv(
        redundancy_path,
        index=False,
    )

    print(
        "\nTraining-only mean absolute target correlations:\n"
    )

    print(
        mean_correlations.to_string(
            index=False
        )
    )

    print(
        "\nTraining-only highly correlated feature pairs "
        f"(|corr| >= {CORRELATION_THRESHOLD:.2f}):\n"
    )

    if redundancy.empty:
        print(
            "No highly correlated feature pairs found."
        )
    else:
        print(
            redundancy.to_string(
                index=False
            )
        )

    print(
        "\nSaved feature audit tables to:",
        output_dir,
    )


if __name__ == "__main__":
    main()