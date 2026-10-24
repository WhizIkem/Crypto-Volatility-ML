from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from crypto_volatility.data.preprocessing import (
    chronological_train_validation_test_split,
)
from crypto_volatility.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from crypto_volatility.evaluation.robust_validation import (
    expanding_validation_splits,
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
    train_lstm_fixed_epochs,
    train_lstm_model,
)
from crypto_volatility.utils.config import load_config

ASSET = "ETH"

INITIAL_TRAIN_SIZE = 1200
VALIDATION_SIZE = 120
STEP_SIZE = 120
PURGE_ROWS = 1

SEQUENCE_LENGTH = 7
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0

LEARNING_RATE = 0.0005
BATCH_SIZE = 32

MAX_EPOCHS = 100
PATIENCE = 10

INNER_VALIDATION_FRACTION = 0.15

SEEDS = [
    42,
    123,
    2026,
]

DEVICE = "cpu"


def load_eth_dataset(
    config: dict,
) -> pd.DataFrame:
    """
    Load the processed ETH model dataset.
    """
    data_config = config["data"]

    processed_dir = Path(
        data_config["processed_dir"]
    )

    interval = data_config[
        "interval"
    ]

    symbol = data_config[
        "symbols"
    ][
        ASSET
    ]

    path = processed_dir / (
        f"{symbol.lower()}_{interval}_model.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    if "timestamp" not in df.columns:
        raise ValueError(
            "ETH dataset is missing 'timestamp'"
        )

    return (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


def build_development_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
]:
    """
    Keep the final outer test partition completely sealed.
    """
    outer_split = (
        chronological_train_validation_test_split(
            df,
            train_fraction=0.70,
            validation_fraction=0.15,
            purge_rows=PURGE_ROWS,
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

    positions = df.index[
        df[
            "timestamp"
        ].eq(
            final_test_start
        )
    ].tolist()

    if len(
        positions
    ) != 1:
        raise ValueError(
            "Could not uniquely locate final test start"
        )

    test_start_position = positions[
        0
    ]

    development_end = (
        test_start_position
        - PURGE_ROWS
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


def set_seed(
    seed: int,
) -> None:
    """
    Set deterministic seeds used by the LSTM experiment.
    """
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def make_sequences_for_targets(
    scaled_features: np.ndarray,
    targets: np.ndarray,
    target_indices: np.ndarray,
    sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Build sequences whose labels correspond exactly to target_indices.

    Historical feature rows before the target may be used as context.
    """
    x_sequences = []
    y_sequences = []

    for target_index in target_indices:
        sequence_end = (
            target_index + 1
        )

        sequence_start = (
            sequence_end
            - sequence_length
        )

        if sequence_start < 0:
            continue

        x_sequences.append(
            scaled_features[
                sequence_start:
                sequence_end
            ]
        )

        y_sequences.append(
            targets[
                target_index
            ]
        )

    if not x_sequences:
        raise ValueError(
            "No LSTM sequences could be created"
        )

    return (
        np.asarray(
            x_sequences,
            dtype=float,
        ),
        np.asarray(
            y_sequences,
            dtype=float,
        ),
    )


def build_inner_training_data(
    train_df: pd.DataFrame,
) -> tuple[
    VolatilitySequenceDataset,
    VolatilitySequenceDataset,
]:
    """
    Build an inner train/validation split used only to estimate
    the early-stopping epoch.

    Feature and target scalers are fitted only on the inner
    training rows.
    """
    n_rows = len(
        train_df
    )

    inner_validation_size = max(
        1,
        int(
            n_rows
            * INNER_VALIDATION_FRACTION
        ),
    )

    validation_start = (
        n_rows
        - inner_validation_size
    )

    train_target_end = (
        validation_start
        - PURGE_ROWS
    )

    if train_target_end <= SEQUENCE_LENGTH:
        raise ValueError(
            "Inner LSTM training period is too short"
        )

    feature_values = (
        train_df[
            CORE_PLUS_DOWNSIDE_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )

    target_values = (
        train_df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        feature_values[
            :train_target_end
        ]
    )

    scaled_features = (
        feature_scaler.transform(
            feature_values
        )
    )

    target_scaler = StandardScaler()

    target_scaler.fit(
        target_values[
            :train_target_end
        ].reshape(
            -1,
            1,
        )
    )

    scaled_targets = (
        target_scaler
        .transform(
            target_values.reshape(
                -1,
                1,
            )
        )
        .reshape(
            -1
        )
    )

    train_indices = np.arange(
        SEQUENCE_LENGTH - 1,
        train_target_end,
    )

    validation_indices = np.arange(
        validation_start,
        n_rows,
    )

    (
        x_train,
        y_train,
    ) = make_sequences_for_targets(
        scaled_features=scaled_features,
        targets=scaled_targets,
        target_indices=train_indices,
        sequence_length=SEQUENCE_LENGTH,
    )

    (
        x_validation,
        y_validation,
    ) = make_sequences_for_targets(
        scaled_features=scaled_features,
        targets=scaled_targets,
        target_indices=validation_indices,
        sequence_length=SEQUENCE_LENGTH,
    )

    return (
        VolatilitySequenceDataset(
            x_train,
            y_train,
        ),
        VolatilitySequenceDataset(
            x_validation,
            y_validation,
        ),
    )


def estimate_best_epoch(
    train_df: pd.DataFrame,
    seed: int,
) -> int:
    """
    Estimate training duration using only data inside the outer
    training fold.
    """
    set_seed(
        seed
    )

    (
        train_dataset,
        validation_dataset,
    ) = build_inner_training_data(
        train_df
    )

    model = VolatilityLSTM(
        n_features=len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    result = train_lstm_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
    )

    return result.best_epoch


def build_outer_datasets(
    train_df: pd.DataFrame,
    context_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[
    VolatilitySequenceDataset,
    VolatilitySequenceDataset,
    StandardScaler,
]:
    """
    Build the final training and outer-validation sequence datasets.

    Feature and target scalers are fitted exclusively on the outer
    training fold.

    The purge/context row may be used as observed feature history,
    but is never used as a training or validation target.
    """
    combined_df = pd.concat(
        [
            train_df,
            context_df,
            validation_df,
        ],
        ignore_index=True,
    )

    n_train = len(
        train_df
    )

    n_context = len(
        context_df
    )

    feature_values = (
        combined_df[
            CORE_PLUS_DOWNSIDE_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )

    target_values = (
        combined_df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        feature_values[
            :n_train
        ]
    )

    scaled_features = (
        feature_scaler.transform(
            feature_values
        )
    )

    target_scaler = StandardScaler()

    target_scaler.fit(
        target_values[
            :n_train
        ].reshape(
            -1,
            1,
        )
    )

    scaled_targets = (
        target_scaler
        .transform(
            target_values.reshape(
                -1,
                1,
            )
        )
        .reshape(
            -1
        )
    )

    train_indices = np.arange(
        SEQUENCE_LENGTH - 1,
        n_train,
    )

    validation_start = (
        n_train
        + n_context
    )

    validation_indices = np.arange(
        validation_start,
        len(
            combined_df
        ),
    )

    (
        x_train,
        y_train,
    ) = make_sequences_for_targets(
        scaled_features=scaled_features,
        targets=scaled_targets,
        target_indices=train_indices,
        sequence_length=SEQUENCE_LENGTH,
    )

    (
        x_validation,
        y_validation,
    ) = make_sequences_for_targets(
        scaled_features=scaled_features,
        targets=scaled_targets,
        target_indices=validation_indices,
        sequence_length=SEQUENCE_LENGTH,
    )

    return (
        VolatilitySequenceDataset(
            x_train,
            y_train,
        ),
        VolatilitySequenceDataset(
            x_validation,
            y_validation,
        ),
        target_scaler,
    )


def predict_scaled(
    model: VolatilityLSTM,
    dataset: VolatilitySequenceDataset,
) -> np.ndarray:
    """
    Generate predictions from a trained LSTM.
    """
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        DEVICE
    )

    model = model.to(
        device
    )

    model.eval()

    predictions = []

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(
                device
            )

            batch_predictions = model(
                x_batch
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(
        predictions,
        dtype=float,
    )


def evaluate_seed(
    train_df: pd.DataFrame,
    context_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed: int,
) -> tuple[
    float,
    float,
    int,
]:
    """
    Train and evaluate one ETH LSTM seed on one outer fold.
    """
    best_epoch = estimate_best_epoch(
        train_df=train_df,
        seed=seed,
    )

    (
        train_dataset,
        validation_dataset,
        target_scaler,
    ) = build_outer_datasets(
        train_df=train_df,
        context_df=context_df,
        validation_df=validation_df,
    )

    set_seed(
        seed
    )

    model = VolatilityLSTM(
        n_features=len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    model = train_lstm_fixed_epochs(
        model=model,
        train_dataset=train_dataset,
        epochs=best_epoch,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    scaled_predictions = predict_scaled(
        model=model,
        dataset=validation_dataset,
    )

    predictions = (
        target_scaler
        .inverse_transform(
            scaled_predictions.reshape(
                -1,
                1,
            )
        )
        .reshape(
            -1
        )
    )

    y_true = (
        validation_df[
            TARGET_COLUMN
        ]
        .reset_index(
            drop=True
        )
    )

    y_pred = pd.Series(
        predictions,
        index=y_true.index,
        name="lstm_prediction",
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = root_mean_squared_error(
        y_true,
        y_pred,
    )

    return (
        mae,
        rmse,
        best_epoch,
    )


def evaluate_fold(
    fold: int,
    train_df: pd.DataFrame,
    context_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate all fixed seeds for one robust-validation fold.
    """
    rows = []

    for seed in SEEDS:
        (
            mae,
            rmse,
            best_epoch,
        ) = evaluate_seed(
            train_df=train_df,
            context_df=context_df,
            validation_df=validation_df,
            seed=seed,
        )

        rows.append(
            {
                "asset": ASSET,
                "fold": fold,
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_start": (
                    validation_df[
                        "timestamp"
                    ].iloc[0]
                ),
                "validation_end": (
                    validation_df[
                        "timestamp"
                    ].iloc[-1]
                ),
                "mae": mae,
                "rmse": rmse,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_fold_summary(
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the three seeds inside each fold.
    """
    return (
        seed_metrics
        .groupby(
            [
                "asset",
                "fold",
            ],
            as_index=False,
        )
        .agg(
            mean_mae=(
                "mae",
                "mean",
            ),
            std_mae=(
                "mae",
                "std",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            std_rmse=(
                "rmse",
                "std",
            ),
            mean_best_epoch=(
                "best_epoch",
                "mean",
            ),
        )
        .sort_values(
            "fold"
        )
        .reset_index(
            drop=True
        )
    )


def build_overall_summary(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate performance across the four historical folds.
    """
    return pd.DataFrame(
        [
            {
                "asset": ASSET,
                "model": "LSTM",
                "mean_mae": (
                    fold_summary[
                        "mean_mae"
                    ].mean()
                ),
                "std_mae_across_folds": (
                    fold_summary[
                        "mean_mae"
                    ].std()
                ),
                "mean_rmse": (
                    fold_summary[
                        "mean_rmse"
                    ].mean()
                ),
                "std_rmse_across_folds": (
                    fold_summary[
                        "mean_rmse"
                    ].std()
                ),
                "min_fold_rmse": (
                    fold_summary[
                        "mean_rmse"
                    ].min()
                ),
                "max_fold_rmse": (
                    fold_summary[
                        "mean_rmse"
                    ].max()
                ),
                "folds": len(
                    fold_summary
                ),
                "seeds_per_fold": len(
                    SEEDS
                ),
            }
        ]
    )


def main() -> None:
    config = load_config()

    df = load_eth_dataset(
        config
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

    print(
        "\nPhase 7 ETH LSTM robust validation"
    )

    print(
        "Sequence length:",
        SEQUENCE_LENGTH,
    )

    print(
        "Hidden size:",
        HIDDEN_SIZE,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Number of folds:",
        len(
            splits
        ),
    )

    print(
        "Final sealed test begins:",
        final_test_start,
    )

    fold_results = []

    for split in splits:
        print(
            f"\nEvaluating fold {split.fold}..."
        )

        result = evaluate_fold(
            fold=split.fold,
            train_df=split.train,
            context_df=split.context,
            validation_df=split.validation,
        )

        fold_results.append(
            result
        )

    seed_metrics = pd.concat(
        fold_results,
        ignore_index=True,
    )

    fold_summary = build_fold_summary(
        seed_metrics
    )

    overall_summary = build_overall_summary(
        fold_summary
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_path = (
        output_dir
        / "robust_lstm_eth_seed_metrics.csv"
    )

    fold_path = (
        output_dir
        / "robust_lstm_eth_fold_summary.csv"
    )

    summary_path = (
        output_dir
        / "robust_lstm_eth_summary.csv"
    )

    seed_metrics.to_csv(
        seed_path,
        index=False,
    )

    fold_summary.to_csv(
        fold_path,
        index=False,
    )

    overall_summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nETH LSTM seed metrics:\n"
    )

    print(
        seed_metrics[
            [
                "fold",
                "seed",
                "best_epoch",
                "mae",
                "rmse",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nETH LSTM fold summary:\n"
    )

    print(
        fold_summary.to_string(
            index=False
        )
    )

    print(
        "\nETH LSTM overall robust summary:\n"
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved seed metrics to:",
        seed_path,
    )

    print(
        "Saved fold summary to:",
        fold_path,
    )

    print(
        "Saved overall summary to:",
        summary_path,
    )


if __name__ == "__main__":
    main()