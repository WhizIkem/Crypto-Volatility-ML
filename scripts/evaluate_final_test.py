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
from crypto_volatility.data.var_panel import (
    build_var_volatility_panel,
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
    train_lstm_fixed_epochs,
)
from crypto_volatility.models.ridge import (
    fit_ridge_model,
    predict_ridge,
)
from crypto_volatility.models.var import (
    fit_var_model,
    forecast_var_one_step,
)
from crypto_volatility.utils.config import load_config

PURGE_ROWS = 1

VAR_LAG = 7

BNB_RIDGE_ALPHA = 100.0

ETH_SEQUENCE_LENGTH = 7
ETH_HIDDEN_SIZE = 32
ETH_NUM_LAYERS = 1
ETH_DROPOUT = 0.0
ETH_LEARNING_RATE = 0.0005
ETH_BATCH_SIZE = 32
ETH_EPOCHS = 52
ETH_SEED = 42

DEVICE = "cpu"


def load_model_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Load all processed per-asset model datasets.
    """
    data_config = config[
        "data"
    ]

    processed_dir = Path(
        data_config[
            "processed_dir"
        ]
    )

    interval = data_config[
        "interval"
    ]

    datasets = {}

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
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

        required_columns = {
            "timestamp",
            "log_gk_volatility",
            TARGET_COLUMN,
        }

        missing_columns = (
            required_columns
            - set(
                df.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{asset} missing columns: "
                f"{sorted(missing_columns)}"
            )

        datasets[
            asset
        ] = (
            df
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

    return datasets


def split_final_periods(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Return:

    development:
        All safe supervised observations before the final-test purge.

    context:
        Purged boundary row. Its target is never used for training,
        but its observed features may be used as historical context.

    test:
        Final untouched test partition.
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
            "Final test partition is empty"
        )

    test_start = (
        outer_split.test[
            "timestamp"
        ].iloc[0]
    )

    positions = df.index[
        df[
            "timestamp"
        ].eq(
            test_start
        )
    ].tolist()

    if len(
        positions
    ) != 1:
        raise ValueError(
            "Could not uniquely locate final test start"
        )

    test_start_position = (
        positions[
            0
        ]
    )

    context_start = (
        test_start_position
        - PURGE_ROWS
    )

    development = (
        df
        .iloc[
            :context_start
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    context = (
        df
        .iloc[
            context_start:
            test_start_position
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        df
        .iloc[
            test_start_position:
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if development.empty:
        raise ValueError(
            "Development partition is empty"
        )

    if len(
        context
    ) != PURGE_ROWS:
        raise ValueError(
            "Unexpected number of context rows"
        )

    if test.empty:
        raise ValueError(
            "Test partition is empty"
        )

    return (
        development,
        context,
        test,
    )


def validate_common_test_dates(
    partitions: dict[
        str,
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ],
    ],
) -> None:
    """
    Ensure all assets share the same sealed test dates.
    """
    assets = list(
        partitions.keys()
    )

    reference = (
        partitions[
            assets[0]
        ][2][
            "timestamp"
        ]
        .reset_index(
            drop=True
        )
    )

    for asset in assets[
        1:
    ]:
        timestamps = (
            partitions[
                asset
            ][2][
                "timestamp"
            ]
            .reset_index(
                drop=True
            )
        )

        if not reference.equals(
            timestamps
        ):
            raise ValueError(
                "Final test timestamps are not aligned "
                f"for {assets[0]} and {asset}"
            )


def build_var_panel_with_targets(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Build aligned VAR volatility series and t+1 targets.
    """
    panel = build_var_volatility_panel(
        datasets=datasets,
        time_column="timestamp",
        volatility_column="log_gk_volatility",
    )

    for asset, df in (
        datasets.items()
    ):
        target_frame = (
            df[
                [
                    "timestamp",
                    TARGET_COLUMN,
                ]
            ]
            .copy()
            .rename(
                columns={
                    TARGET_COLUMN: (
                        f"{asset}_target"
                    ),
                }
            )
        )

        panel = panel.merge(
            target_frame,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

    return (
        panel
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


def evaluate_final_var(
    development_datasets: dict[
        str,
        pd.DataFrame,
    ],
    context_datasets: dict[
        str,
        pd.DataFrame,
    ],
    test_datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Produce corrected t -> t+1 expanding VAR forecasts.

    VAR forecasts all four assets jointly, although only BTC and SOL
    are selected as champion VAR outputs.
    """
    development_panel = (
        build_var_panel_with_targets(
            development_datasets
        )
    )

    context_panel = (
        build_var_panel_with_targets(
            context_datasets
        )
    )

    test_panel = (
        build_var_panel_with_targets(
            test_datasets
        )
    )

    asset_columns = list(
        development_datasets.keys()
    )

    history = pd.concat(
        [
            development_panel[
                asset_columns
            ],
            context_panel[
                asset_columns
            ],
        ],
        ignore_index=True,
    )

    rows = []

    for index in range(
        len(
            test_panel
        )
    ):
        current_row = (
            test_panel
            .iloc[
                index
            ]
        )

        current_observation = pd.DataFrame(
            [
                {
                    asset: current_row[
                        asset
                    ]
                    for asset in asset_columns
                }
            ]
        )

        # At forecast origin t, current volatility is already observed.
        history = pd.concat(
            [
                history,
                current_observation,
            ],
            ignore_index=True,
        )

        model = fit_var_model(
            train_df=history,
            columns=asset_columns,
            maxlags=VAR_LAG,
            ic=None,
        )

        forecast = forecast_var_one_step(
            model=model,
            history_df=history,
            columns=asset_columns,
        )

        row = {
            "timestamp": current_row[
                "timestamp"
            ],
        }

        for asset in asset_columns:
            row[
                f"{asset}_actual"
            ] = float(
                current_row[
                    f"{asset}_target"
                ]
            )

            row[
                f"{asset}_forecast"
            ] = float(
                forecast[
                    asset
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def evaluate_bnb_ridge(
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train the frozen BNB Ridge model on the complete development
    period and evaluate once on the sealed final test.
    """
    model = fit_ridge_model(
        train_df=development,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
        target_column=TARGET_COLUMN,
        alpha=BNB_RIDGE_ALPHA,
    )

    predictions = predict_ridge(
        model=model,
        df=test,
        feature_columns=CORE_PLUS_DOWNSIDE_FEATURES,
    )

    return pd.DataFrame(
        {
            "timestamp": test[
                "timestamp"
            ].to_numpy(),
            "BNB_actual": (
                test[
                    TARGET_COLUMN
                ]
                .to_numpy(
                    dtype=float
                )
            ),
            "BNB_forecast": (
                predictions
                .to_numpy(
                    dtype=float
                )
            ),
        }
    )


def set_seed(
    seed: int,
) -> None:
    """
    Set deterministic random seeds for the production LSTM.
    """
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def make_sequences_for_indices(
    scaled_features: np.ndarray,
    scaled_targets: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Build fixed-length LSTM sequences for specified target rows.
    """
    x_sequences = []
    y_sequences = []

    for target_index in target_indices:
        sequence_end = (
            target_index
            + 1
        )

        sequence_start = (
            sequence_end
            - ETH_SEQUENCE_LENGTH
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
            scaled_targets[
                target_index
            ]
        )

    if not x_sequences:
        raise ValueError(
            "No ETH LSTM sequences were created"
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


def predict_lstm_dataset(
    model: VolatilityLSTM,
    dataset: VolatilitySequenceDataset,
) -> np.ndarray:
    """
    Generate CPU LSTM predictions.
    """
    loader = DataLoader(
        dataset,
        batch_size=ETH_BATCH_SIZE,
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


def evaluate_eth_lstm(
    development: pd.DataFrame,
    context: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train the frozen ETH production LSTM using the complete
    development period.

    Scalers are fitted on development data only.

    The purge row and previous test feature rows may be used as
    observed sequence context, but test targets are never used for
    training.
    """
    combined = pd.concat(
        [
            development,
            context,
            test,
        ],
        ignore_index=True,
    )

    n_development = len(
        development
    )

    n_context = len(
        context
    )

    feature_values = (
        combined[
            CORE_PLUS_DOWNSIDE_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )

    target_values = (
        combined[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        feature_values[
            :n_development
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
            :n_development
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
        ETH_SEQUENCE_LENGTH - 1,
        n_development,
    )

    test_start = (
        n_development
        + n_context
    )

    test_indices = np.arange(
        test_start,
        len(
            combined
        ),
    )

    (
        x_train,
        y_train,
    ) = make_sequences_for_indices(
        scaled_features=scaled_features,
        scaled_targets=scaled_targets,
        target_indices=train_indices,
    )

    (
        x_test,
        y_test,
    ) = make_sequences_for_indices(
        scaled_features=scaled_features,
        scaled_targets=scaled_targets,
        target_indices=test_indices,
    )

    train_dataset = (
        VolatilitySequenceDataset(
            x_train,
            y_train,
        )
    )

    test_dataset = (
        VolatilitySequenceDataset(
            x_test,
            y_test,
        )
    )

    set_seed(
        ETH_SEED
    )

    model = VolatilityLSTM(
        n_features=len(
            CORE_PLUS_DOWNSIDE_FEATURES
        ),
        hidden_size=ETH_HIDDEN_SIZE,
        num_layers=ETH_NUM_LAYERS,
        dropout=ETH_DROPOUT,
    )

    model = train_lstm_fixed_epochs(
        model=model,
        train_dataset=train_dataset,
        epochs=ETH_EPOCHS,
        learning_rate=ETH_LEARNING_RATE,
        batch_size=ETH_BATCH_SIZE,
        device=DEVICE,
    )

    scaled_predictions = (
        predict_lstm_dataset(
            model=model,
            dataset=test_dataset,
        )
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

    return pd.DataFrame(
        {
            "timestamp": test[
                "timestamp"
            ].to_numpy(),
            "ETH_actual": (
                test[
                    TARGET_COLUMN
                ]
                .to_numpy(
                    dtype=float
                )
            ),
            "ETH_forecast": predictions,
        }
    )


def calculate_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate final champion-model metrics.
    """
    model_by_asset = {
        "BTC": "VAR",
        "ETH": "LSTM",
        "SOL": "VAR",
        "BNB": "Ridge",
    }

    rows = []

    for asset, model_name in (
        model_by_asset.items()
    ):
        y_true = predictions[
            f"{asset}_actual"
        ]

        y_pred = predictions[
            f"{asset}_forecast"
        ]

        rows.append(
            {
                "asset": asset,
                "model": model_name,
                "n_observations": len(
                    predictions
                ),
                "mae": mean_absolute_error(
                    y_true,
                    y_pred,
                ),
                "rmse": root_mean_squared_error(
                    y_true,
                    y_pred,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    config = load_config()

    datasets = load_model_datasets(
        config
    )

    partitions = {}

    for asset, df in (
        datasets.items()
    ):
        partitions[
            asset
        ] = split_final_periods(
            df
        )

    validate_common_test_dates(
        partitions
    )

    development_datasets = {
        asset: parts[0]
        for asset, parts in (
            partitions.items()
        )
    }

    context_datasets = {
        asset: parts[1]
        for asset, parts in (
            partitions.items()
        )
    }

    test_datasets = {
        asset: parts[2]
        for asset, parts in (
            partitions.items()
        )
    }

    test_start = (
        test_datasets[
            "BTC"
        ][
            "timestamp"
        ].iloc[0]
    )

    test_end = (
        test_datasets[
            "BTC"
        ][
            "timestamp"
        ].iloc[-1]
    )

    print(
        "\nFINAL UNTOUCHED TEST EVALUATION"
    )

    print(
        "Test start:",
        test_start,
    )

    print(
        "Test end:",
        test_end,
    )

    print(
        "Test rows:",
        len(
            test_datasets[
                "BTC"
            ]
        ),
    )

    print(
        "\nFrozen champions:"
    )

    print(
        "BTC -> VAR lag 7"
    )

    print(
        "ETH -> LSTM "
        "seq 7 / hidden 32 / "
        "lr 0.0005 / epochs 52 / seed 42"
    )

    print(
        "SOL -> VAR lag 7"
    )

    print(
        "BNB -> Ridge alpha 100"
    )

    var_predictions = (
        evaluate_final_var(
            development_datasets=development_datasets,
            context_datasets=context_datasets,
            test_datasets=test_datasets,
        )
    )

    eth_predictions = (
        evaluate_eth_lstm(
            development=development_datasets[
                "ETH"
            ],
            context=context_datasets[
                "ETH"
            ],
            test=test_datasets[
                "ETH"
            ],
        )
    )

    bnb_predictions = (
        evaluate_bnb_ridge(
            development=development_datasets[
                "BNB"
            ],
            test=test_datasets[
                "BNB"
            ],
        )
    )

    final_predictions = (
        var_predictions[
            [
                "timestamp",
                "BTC_actual",
                "BTC_forecast",
                "SOL_actual",
                "SOL_forecast",
            ]
        ]
        .merge(
            eth_predictions,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            bnb_predictions,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )
    )

    expected_rows = len(
        test_datasets[
            "BTC"
        ]
    )

    if len(
        final_predictions
    ) != expected_rows:
        raise ValueError(
            "Final prediction rows do not match "
            "the sealed test partition"
        )

    metrics = calculate_metrics(
        final_predictions
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "final_test_predictions.csv"
    )

    metrics_path = (
        output_dir
        / "final_test_metrics.csv"
    )

    final_predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\nFinal untouched test metrics:\n"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\nSaved predictions to:",
        predictions_path,
    )

    print(
        "Saved metrics to:",
        metrics_path,
    )


if __name__ == "__main__":
    main()