from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.data.sequences import (
    build_leakage_safe_sequence_splits,
    build_sequence_train_validation_split,
    inverse_transform_sequence_targets,
    scale_sequence_targets,
)
from crypto_volatility.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from crypto_volatility.features.build_features import (
    CORE_PLUS_DOWNSIDE_FEATURES,
    TARGET_COLUMN,
)
from crypto_volatility.models.lstm import (
    VolatilityLSTM,
    VolatilitySequenceDataset,
)
from crypto_volatility.models.lstm_training import (
    predict_lstm,
    train_lstm_fixed_epochs,
    train_lstm_model,
)
from crypto_volatility.utils.config import load_config

SEQUENCE_LENGTHS = [
    7,
    14,
    30,
]

HIDDEN_SIZES = [
    16,
    32,
]

LEARNING_RATES = [
    0.0005,
    0.001,
]

SEEDS = [
    42,
    123,
    2026,
]

INNER_TRAIN_FRACTION = 0.85

PURGE_ROWS = 1

NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 32

MAX_EPOCHS = 100
PATIENCE = 10

WEIGHT_DECAY = 0.0

DEVICE = "cpu"


def load_processed_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config["interval"]

    datasets = {}

    for asset, symbol in (
        data_config["symbols"].items()
    ):
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


def tune_lstm_configuration(
    outer_train: pd.DataFrame,
) -> tuple[
    dict[str, int | float],
    pd.DataFrame,
]:
    """
    Tune LSTM configuration using inner validation only.
    """
    rows = []

    for sequence_length in SEQUENCE_LENGTHS:
        inner_split = (
            build_sequence_train_validation_split(
                df=outer_train,
                feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
                target_column=TARGET_COLUMN,
                sequence_length=sequence_length,
                train_fraction=INNER_TRAIN_FRACTION,
                purge_rows=PURGE_ROWS,
            )
        )

        scaled_targets = (
            scale_sequence_targets(
                y_train=inner_split.y_train,
                y_validation=inner_split.y_validation,
            )
        )

        for hidden_size in HIDDEN_SIZES:
            for learning_rate in LEARNING_RATES:
                torch.manual_seed(
                    42
                )

                np.random.seed(
                    42
                )

                train_dataset = (
                    VolatilitySequenceDataset(
                        x=inner_split.x_train,
                        y=scaled_targets.y_train,
                    )
                )

                validation_dataset = (
                    VolatilitySequenceDataset(
                        x=inner_split.x_validation,
                        y=scaled_targets.y_validation,
                    )
                )

                model = VolatilityLSTM(
                    n_features=len(
                        CORE_PLUS_DOWNSIDE_FEATURES
                    ),
                    hidden_size=hidden_size,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT,
                )

                result = train_lstm_model(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    learning_rate=learning_rate,
                    batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    weight_decay=WEIGHT_DECAY,
                    device=DEVICE,
                )

                predictions_scaled = predict_lstm(
                    model=result.model,
                    dataset=validation_dataset,
                    batch_size=BATCH_SIZE,
                    device=DEVICE,
                )

                predictions = (
                    inverse_transform_sequence_targets(
                        predictions_scaled,
                        scaled_targets.target_scaler,
                    )
                )

                rmse = root_mean_squared_error(
                    pd.Series(
                        inner_split.y_validation
                    ),
                    pd.Series(
                        predictions
                    ),
                )

                rows.append(
                    {
                        "sequence_length": sequence_length,
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "best_epoch": result.best_epoch,
                        "epochs_run": len(
                            result.validation_losses
                        ),
                        "inner_validation_rmse": rmse,
                    }
                )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        [
            "inner_validation_rmse",
            "sequence_length",
            "hidden_size",
            "learning_rate",
        ]
    ).reset_index(
        drop=True
    )

    best = summary.iloc[
        0
    ]

    config = {
        "sequence_length": int(
            best[
                "sequence_length"
            ]
        ),
        "hidden_size": int(
            best[
                "hidden_size"
            ]
        ),
        "learning_rate": float(
            best[
                "learning_rate"
            ]
        ),
        "best_epoch": int(
            best[
                "best_epoch"
            ]
        ),
    }

    return (
        config,
        summary,
    )


def evaluate_seed(
    df: pd.DataFrame,
    selected_config: dict[str, int | float],
    seed: int,
) -> dict[str, float | int]:
    """
    Refit selected configuration on outer training data and
    evaluate on outer validation for one random seed.
    """
    sequence_length = int(
        selected_config[
            "sequence_length"
        ]
    )

    hidden_size = int(
        selected_config[
            "hidden_size"
        ]
    )

    learning_rate = float(
        selected_config[
            "learning_rate"
        ]
    )

    best_epoch = int(
        selected_config[
            "best_epoch"
        ]
    )

    sequence_split = (
        build_leakage_safe_sequence_splits(
            df=df,
            feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
            target_column=TARGET_COLUMN,
            sequence_length=sequence_length,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=PURGE_ROWS,
        )
    )

    scaled_targets = (
        scale_sequence_targets(
            y_train=sequence_split.y_train,
            y_validation=sequence_split.y_validation,
        )
    )

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    train_dataset = (
        VolatilitySequenceDataset(
            x=sequence_split.x_train,
            y=scaled_targets.y_train,
        )
    )

    validation_dataset = (
        VolatilitySequenceDataset(
            x=sequence_split.x_validation,
            y=scaled_targets.y_validation,
        )
    )

    model = VolatilityLSTM(
        n_features=len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    model = train_lstm_fixed_epochs(
        model=model,
        train_dataset=train_dataset,
        epochs=best_epoch,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    )

    predictions_scaled = predict_lstm(
        model=model,
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    predictions = (
        inverse_transform_sequence_targets(
            predictions_scaled,
            scaled_targets.target_scaler,
        )
    )

    y_true = (
        sequence_split.y_validation
    )

    return {
        "seed": seed,
        "mae": mean_absolute_error(
            pd.Series(
                y_true
            ),
            pd.Series(
                predictions
            ),
        ),
        "rmse": root_mean_squared_error(
            pd.Series(
                y_true
            ),
            pd.Series(
                predictions
            ),
        ),
    }


def evaluate_asset(
    asset: str,
    df: pd.DataFrame,
) -> tuple[
    dict[str, str | float | int],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Tune on outer training data only and evaluate selected setup
    across several fixed random seeds.
    """
    outer_split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=PURGE_ROWS,
        )
    )

    outer_train = (
        outer_split.train
        .copy()
        .reset_index(
            drop=True
        )
    )

    (
        selected_config,
        tuning_summary,
    ) = tune_lstm_configuration(
        outer_train
    )

    seed_rows = []

    for seed in SEEDS:
        seed_result = evaluate_seed(
            df=df,
            selected_config=selected_config,
            seed=seed,
        )

        seed_rows.append(
            seed_result
        )

    seed_results = pd.DataFrame(
        seed_rows
    )

    metrics = {
        "asset": asset,
        "model": "lstm_tuned",
        "feature_set": "core_plus_downside_7",
        "n_features": len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        "sequence_length": selected_config[
            "sequence_length"
        ],
        "hidden_size": selected_config[
            "hidden_size"
        ],
        "learning_rate": selected_config[
            "learning_rate"
        ],
        "best_epoch": selected_config[
            "best_epoch"
        ],
        "n_seeds": len(
            SEEDS
        ),
        "mean_mae": float(
            seed_results[
                "mae"
            ].mean()
        ),
        "std_mae": float(
            seed_results[
                "mae"
            ].std(
                ddof=1
            )
        ),
        "mean_rmse": float(
            seed_results[
                "rmse"
            ].mean()
        ),
        "std_rmse": float(
            seed_results[
                "rmse"
            ].std(
                ddof=1
            )
        ),
    }

    tuning_summary.insert(
        0,
        "asset",
        asset,
    )

    seed_results.insert(
        0,
        "asset",
        asset,
    )

    return (
        metrics,
        tuning_summary,
        seed_results,
    )


def main() -> None:
    config = load_config()

    datasets = load_processed_datasets(
        config
    )

    metric_rows = []
    tuning_frames = []
    seed_frames = []

    for asset, df in datasets.items():
        print(
            f"\nAuditing LSTM for {asset}..."
        )

        (
            metrics,
            tuning_summary,
            seed_results,
        ) = evaluate_asset(
            asset,
            df,
        )

        metric_rows.append(
            metrics
        )

        tuning_frames.append(
            tuning_summary
        )

        seed_frames.append(
            seed_results
        )

        print(
            f"{asset} selected sequence length: "
            f"{metrics['sequence_length']}"
        )

        print(
            f"{asset} selected hidden size: "
            f"{metrics['hidden_size']}"
        )

        print(
            f"{asset} selected learning rate: "
            f"{metrics['learning_rate']}"
        )

        print(
            f"{asset} mean validation RMSE: "
            f"{metrics['mean_rmse']:.6f}"
        )

    metrics = pd.DataFrame(
        metric_rows
    ).sort_values(
        [
            "mean_rmse",
            "asset",
        ]
    ).reset_index(
        drop=True
    )

    tuning = pd.concat(
        tuning_frames,
        ignore_index=True,
    )

    seeds = pd.concat(
        seed_frames,
        ignore_index=True,
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "lstm_tuned_validation_metrics.csv"
    )

    tuning_path = (
        output_dir
        / "lstm_tuning_summary.csv"
    )

    seeds_path = (
        output_dir
        / "lstm_seed_results.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    tuning.to_csv(
        tuning_path,
        index=False,
    )

    seeds.to_csv(
        seeds_path,
        index=False,
    )

    print(
        "\nTuned LSTM validation metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nLSTM seed results:\n"
    )

    print(
        seeds.to_string(
            index=False
        )
    )

    print(
        "\nSaved metrics to:",
        metrics_path,
    )

    print(
        "Saved tuning summary to:",
        tuning_path,
    )

    print(
        "Saved seed results to:",
        seeds_path,
    )


if __name__ == "__main__":
    main()