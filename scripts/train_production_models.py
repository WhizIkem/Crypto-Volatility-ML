from __future__ import annotations

import json
import pickle
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crypto_volatility.models.lstm import VolatilityLSTM
from crypto_volatility.utils.config import load_config

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]

FEATURE_COLUMNS = [
    "log_gk_volatility",
    "har_vol_weekly",
    "har_vol_monthly",
    "absolute_return",
    "log_volume",
    "cross_asset_mean_log_volatility",
    "downside_return",
]

TARGET_COLUMN = (
    "target_log_gk_vol_t_plus_1"
)

TIMESTAMP_COLUMN = "timestamp"

# ------------------------------------------------------------------
# Frozen production specifications.
# These must not be tuned in this script.
# ------------------------------------------------------------------

VAR_LAG = 7

ETH_SEQUENCE_LENGTH = 7
ETH_HIDDEN_SIZE = 32
ETH_NUM_LAYERS = 1
ETH_DROPOUT = 0.0
ETH_LEARNING_RATE = 0.0005
ETH_BATCH_SIZE = 32
ETH_EPOCHS = 52
ETH_SEED = 42

BNB_RIDGE_ALPHA = 100.0

PRODUCTION_DIR = Path(
    "models/production"
)


def set_seed(
    seed: int,
) -> None:
    """
    Make the production LSTM training run reproducible.
    """
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def load_asset_dataset(
    *,
    asset: str,
    symbol: str,
    interval: str,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Load one model-ready asset dataset.
    """
    path = (
        processed_dir
        / f"{symbol.lower()}_{interval}_model.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{asset} model dataset not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    required_columns = {
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{asset} dataset missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    if df[
        TIMESTAMP_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"{asset} contains duplicate timestamps"
        )

    required_numeric = [
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]

    if df[
        required_numeric
    ].isna().any().any():
        raise ValueError(
            f"{asset} production dataset contains missing values"
        )

    numeric_values = (
        df[
            required_numeric
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            f"{asset} production dataset contains non-finite values"
        )

    return df


def load_all_datasets(
    config: dict,
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Load all four model-ready datasets.
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
        datasets[
            asset
        ] = load_asset_dataset(
            asset=asset,
            symbol=symbol,
            interval=interval,
            processed_dir=processed_dir,
        )

    reference = (
        datasets[
            "BTC"
        ][
            TIMESTAMP_COLUMN
        ]
        .reset_index(
            drop=True
        )
    )

    for asset in ASSETS[
        1:
    ]:
        timestamps = (
            datasets[
                asset
            ][
                TIMESTAMP_COLUMN
            ]
            .reset_index(
                drop=True
            )
        )

        if not timestamps.equals(
            reference
        ):
            raise ValueError(
                "Production datasets are not timestamp aligned"
            )

    return datasets


def save_pickle(
    value: object,
    path: Path,
) -> None:
    """
    Persist a Python object.
    """
    with path.open(
        "wb"
    ) as file:
        pickle.dump(
            value,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def train_production_var(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> dict[str, object]:
    """
    Train one joint VAR model.

    The production system will use the BTC and SOL equations,
    although all four volatility series are required by the VAR.
    """
    frames = []

    for asset in ASSETS:
        frame = (
            datasets[
                asset
            ][
                [
                    TIMESTAMP_COLUMN,
                    "log_gk_volatility",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_gk_volatility": asset,
                }
            )
        )

        frames.append(
            frame
        )

    panel = frames[
        0
    ]

    for frame in frames[
        1:
    ]:
        panel = panel.merge(
            frame,
            on=TIMESTAMP_COLUMN,
            how="inner",
            validate="one_to_one",
        )

    panel = (
        panel
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    values = (
        panel[
            ASSETS
        ]
        .astype(
            float
        )
    )

    model = VAR(
        values
    ).fit(
        VAR_LAG
    )

    model_path = (
        PRODUCTION_DIR
        / "var_model.pkl"
    )

    save_pickle(
        model,
        model_path,
    )

    history_path = (
        PRODUCTION_DIR
        / "var_history_tail.parquet"
    )

    panel[
        [
            TIMESTAMP_COLUMN,
            *ASSETS,
        ]
    ].tail(
        30
    ).to_parquet(
        history_path,
        index=False,
    )

    return {
        "artifact": str(
            model_path
        ),
        "history_artifact": str(
            history_path
        ),
        "lag": VAR_LAG,
        "columns": ASSETS,
        "n_observations": int(
            len(
                panel
            )
        ),
        "training_start": str(
            panel[
                TIMESTAMP_COLUMN
            ].iloc[
                0
            ]
        ),
        "training_end": str(
            panel[
                TIMESTAMP_COLUMN
            ].iloc[
                -1
            ]
        ),
    }


def train_production_ridge(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Train the frozen BNB Ridge model on all available data.
    """
    x = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    scaler = StandardScaler()

    x_scaled = (
        scaler.fit_transform(
            x
        )
    )

    model = Ridge(
        alpha=BNB_RIDGE_ALPHA
    )

    model.fit(
        x_scaled,
        y,
    )

    model_path = (
        PRODUCTION_DIR
        / "bnb_ridge_model.pkl"
    )

    scaler_path = (
        PRODUCTION_DIR
        / "bnb_feature_scaler.pkl"
    )

    save_pickle(
        model,
        model_path,
    )

    save_pickle(
        scaler,
        scaler_path,
    )

    return {
        "artifact": str(
            model_path
        ),
        "feature_scaler": str(
            scaler_path
        ),
        "alpha": BNB_RIDGE_ALPHA,
        "features": FEATURE_COLUMNS,
        "n_observations": int(
            len(
                df
            )
        ),
        "training_start": str(
            df[
                TIMESTAMP_COLUMN
            ].iloc[
                0
            ]
        ),
        "training_end": str(
            df[
                TIMESTAMP_COLUMN
            ].iloc[
                -1
            ]
        ),
    }


def build_lstm_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Build sequences where features through day t predict the
    stored t+1 volatility target for day t.
    """
    x_sequences = []
    y_sequences = []

    for end_position in range(
        sequence_length - 1,
        len(
            features
        ),
    ):
        start_position = (
            end_position
            - sequence_length
            + 1
        )

        x_sequences.append(
            features[
                start_position:
                end_position
                + 1
            ]
        )

        y_sequences.append(
            targets[
                end_position
            ]
        )

    return (
        np.asarray(
            x_sequences,
            dtype=np.float32,
        ),
        np.asarray(
            y_sequences,
            dtype=np.float32,
        ),
    )


def train_production_lstm(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Train the frozen ETH LSTM specification on all available
    historical observations.
    """
    set_seed(
        ETH_SEED
    )

    device = torch.device(
        "cpu"
    )

    raw_features = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    raw_targets = (
        df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
        .reshape(
            -1,
            1,
        )
    )

    feature_scaler = (
        StandardScaler()
    )

    target_scaler = (
        StandardScaler()
    )

    scaled_features = (
        feature_scaler.fit_transform(
            raw_features
        )
    )

    scaled_targets = (
        target_scaler.fit_transform(
            raw_targets
        )
        .ravel()
    )

    (
        x_sequences,
        y_sequences,
    ) = build_lstm_sequences(
        features=scaled_features,
        targets=scaled_targets,
        sequence_length=(
            ETH_SEQUENCE_LENGTH
        ),
    )

    dataset = TensorDataset(
        torch.from_numpy(
            x_sequences
        ),
        torch.from_numpy(
            y_sequences
        ),
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        ETH_SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=ETH_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = VolatilityLSTM(
        n_features=len(
            FEATURE_COLUMNS
        ),
        hidden_size=(
            ETH_HIDDEN_SIZE
        ),
        num_layers=(
            ETH_NUM_LAYERS
        ),
        dropout=(
            ETH_DROPOUT
        ),
    ).to(
        device
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=ETH_LEARNING_RATE,
        )
    )

    loss_function = (
        nn.MSELoss()
    )

    model.train()

    final_loss = float(
        "nan"
    )

    for epoch in range(
        1,
        ETH_EPOCHS + 1,
    ):
        epoch_losses = []

        for (
            x_batch,
            y_batch,
        ) in loader:
            x_batch = (
                x_batch.to(
                    device
                )
            )

            y_batch = (
                y_batch.to(
                    device
                )
            )

            optimizer.zero_grad()

            predictions = model(
                x_batch
            )

            loss = loss_function(
                predictions,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        final_loss = float(
            np.mean(
                epoch_losses
            )
        )

        if (
            epoch == 1
            or epoch
            % 10
            == 0
            or epoch
            == ETH_EPOCHS
        ):
            print(
                "ETH LSTM epoch "
                f"{epoch:>3}/{ETH_EPOCHS} "
                f"loss={final_loss:.6f}"
            )

    checkpoint_path = (
        PRODUCTION_DIR
        / "eth_lstm.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "input_size": len(
                FEATURE_COLUMNS
            ),
            "hidden_size": (
                ETH_HIDDEN_SIZE
            ),
            "num_layers": (
                ETH_NUM_LAYERS
            ),
            "dropout": (
                ETH_DROPOUT
            ),
            "sequence_length": (
                ETH_SEQUENCE_LENGTH
            ),
            "features": (
                FEATURE_COLUMNS
            ),
            "target_column": (
                TARGET_COLUMN
            ),
            "epochs": (
                ETH_EPOCHS
            ),
            "learning_rate": (
                ETH_LEARNING_RATE
            ),
            "seed": (
                ETH_SEED
            ),
        },
        checkpoint_path,
    )

    feature_scaler_path = (
        PRODUCTION_DIR
        / "eth_feature_scaler.pkl"
    )

    target_scaler_path = (
        PRODUCTION_DIR
        / "eth_target_scaler.pkl"
    )

    save_pickle(
        feature_scaler,
        feature_scaler_path,
    )

    save_pickle(
        target_scaler,
        target_scaler_path,
    )

    return {
        "artifact": str(
            checkpoint_path
        ),
        "feature_scaler": str(
            feature_scaler_path
        ),
        "target_scaler": str(
            target_scaler_path
        ),
        "sequence_length": (
            ETH_SEQUENCE_LENGTH
        ),
        "hidden_size": (
            ETH_HIDDEN_SIZE
        ),
        "num_layers": (
            ETH_NUM_LAYERS
        ),
        "dropout": (
            ETH_DROPOUT
        ),
        "learning_rate": (
            ETH_LEARNING_RATE
        ),
        "epochs": (
            ETH_EPOCHS
        ),
        "seed": (
            ETH_SEED
        ),
        "device": "cpu",
        "features": (
            FEATURE_COLUMNS
        ),
        "n_sequences": int(
            len(
                x_sequences
            )
        ),
        "final_training_loss": (
            final_loss
        ),
        "training_start": str(
            df[
                TIMESTAMP_COLUMN
            ].iloc[
                0
            ]
        ),
        "training_end": str(
            df[
                TIMESTAMP_COLUMN
            ].iloc[
                -1
            ]
        ),
    }


def save_manifest(
    *,
    datasets: dict[
        str,
        pd.DataFrame,
    ],
    var_metadata: dict[
        str,
        object,
    ],
    eth_metadata: dict[
        str,
        object,
    ],
    bnb_metadata: dict[
        str,
        object,
    ],
) -> Path:
    """
    Save the production-model manifest used by later inference.
    """
    manifest = {
        "created_at_utc": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "forecast_target": (
            TARGET_COLUMN
        ),
        "forecast_horizon": (
            "1 day ahead"
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "assets": ASSETS,
        "champions": {
            "BTC": {
                "model": "VAR",
                "artifact": (
                    var_metadata[
                        "artifact"
                    ]
                ),
                "lag": VAR_LAG,
            },
            "ETH": {
                "model": "LSTM",
                "artifact": (
                    eth_metadata[
                        "artifact"
                    ]
                ),
                "sequence_length": (
                    ETH_SEQUENCE_LENGTH
                ),
                "hidden_size": (
                    ETH_HIDDEN_SIZE
                ),
            },
            "SOL": {
                "model": "VAR",
                "artifact": (
                    var_metadata[
                        "artifact"
                    ]
                ),
                "lag": VAR_LAG,
            },
            "BNB": {
                "model": "Ridge",
                "artifact": (
                    bnb_metadata[
                        "artifact"
                    ]
                ),
                "alpha": (
                    BNB_RIDGE_ALPHA
                ),
            },
        },
        "models": {
            "var": (
                var_metadata
            ),
            "eth_lstm": (
                eth_metadata
            ),
            "bnb_ridge": (
                bnb_metadata
            ),
        },
        "portfolio": {
            "annual_volatility_target": (
                0.20
            ),
            "max_asset_weight": (
                0.40
            ),
            "correlation_lookback_days": (
                30
            ),
            "minimum_turnover_threshold": (
                0.05
            ),
            "transaction_cost_bps": (
                10.0
            ),
            "max_risky_exposure": (
                1.0
            ),
            "leverage": False,
        },
        "data": {
            asset: {
                "rows": int(
                    len(
                        df
                    )
                ),
                "start": str(
                    df[
                        TIMESTAMP_COLUMN
                    ].iloc[
                        0
                    ]
                ),
                "end": str(
                    df[
                        TIMESTAMP_COLUMN
                    ].iloc[
                        -1
                    ]
                ),
            }
            for asset, df in (
                datasets.items()
            )
        },
    }

    path = (
        PRODUCTION_DIR
        / "manifest.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    return path


def main() -> None:
    print(
        "\nProduction model training"
    )

    print(
        "\nFrozen specifications:"
    )

    print(
        "BTC -> VAR lag 7"
    )

    print(
        "ETH -> LSTM seq 7 / hidden 32 "
        "/ lr 0.0005 / epochs 52 / seed 42"
    )

    print(
        "SOL -> VAR lag 7"
    )

    print(
        "BNB -> Ridge alpha 100"
    )

    PRODUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config()

    datasets = (
        load_all_datasets(
            config
        )
    )

    reference = datasets[
        "BTC"
    ]

    print(
        "\nTraining data:"
    )

    print(
        "Rows:",
        len(
            reference
        ),
    )

    print(
        "Start:",
        reference[
            TIMESTAMP_COLUMN
        ].iloc[
            0
        ],
    )

    print(
        "End:",
        reference[
            TIMESTAMP_COLUMN
        ].iloc[
            -1
        ],
    )

    print(
        "\nTraining joint VAR..."
    )

    var_metadata = (
        train_production_var(
            datasets
        )
    )

    print(
        "VAR saved."
    )

    print(
        "\nTraining BNB Ridge..."
    )

    bnb_metadata = (
        train_production_ridge(
            datasets[
                "BNB"
            ]
        )
    )

    print(
        "BNB Ridge saved."
    )

    print(
        "\nTraining ETH LSTM on CPU..."
    )

    eth_metadata = (
        train_production_lstm(
            datasets[
                "ETH"
            ]
        )
    )

    print(
        "ETH LSTM saved."
    )

    manifest_path = (
        save_manifest(
            datasets=datasets,
            var_metadata=(
                var_metadata
            ),
            eth_metadata=(
                eth_metadata
            ),
            bnb_metadata=(
                bnb_metadata
            ),
        )
    )

    print(
        "\nProduction training complete."
    )

    print(
        "\nArtifacts:"
    )

    for path in sorted(
        PRODUCTION_DIR.iterdir()
    ):
        print(
            " -",
            path,
        )

    print(
        "\nManifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()