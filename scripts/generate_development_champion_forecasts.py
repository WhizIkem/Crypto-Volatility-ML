from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]

SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
}

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

INITIAL_TRAIN_SIZE = 1200
VALIDATION_SIZE = 120
STEP_SIZE = 120
PURGE_ROWS = 1

N_FOLDS = 4

VAR_LAG = 7
BNB_RIDGE_ALPHA = 100.0

ETH_SEQUENCE_LENGTH = 7
ETH_HIDDEN_SIZE = 32
ETH_LEARNING_RATE = 0.0005
ETH_BATCH_SIZE = 32
ETH_MAX_EPOCHS = 100
ETH_PATIENCE = 10

ETH_SEEDS = [
    42,
    123,
    2026,
]

# Same development boundary used in Phase 7.
FINAL_TEST_START = pd.Timestamp(
    "2025-11-28",
    tz="UTC",
)


class DevelopmentLSTM(
    nn.Module
):
    """
    Same simple one-layer LSTM structure used for ETH.

    Sequence:
        7 days x 7 features

    Hidden size:
        32
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        _, (
            hidden,
            _,
        ) = self.lstm(
            x
        )

        prediction = self.output(
            hidden[
                -1
            ]
        )

        return prediction.squeeze(
            -1
        )


def set_seed(
    seed: int,
) -> None:
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def load_asset_dataset(
    asset: str,
) -> pd.DataFrame:
    symbol = SYMBOLS[
        asset
    ]

    path = Path(
        "data/processed"
    ) / (
        f"{symbol.lower()}_1d_model.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    required = {
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        "log_gk_volatility",
        *FEATURE_COLUMNS,
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{asset} is missing columns: "
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

    return df


def load_all_datasets() -> dict[
    str,
    pd.DataFrame,
]:
    datasets = {
        asset: load_asset_dataset(
            asset
        )
        for asset in ASSETS
    }

    reference_timestamps = (
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
            reference_timestamps
        ):
            raise ValueError(
                "Asset model datasets are not "
                "timestamp-aligned"
            )

    return datasets


def get_development_length(
    df: pd.DataFrame,
) -> int:
    test_positions = np.flatnonzero(
        (
            df[
                TIMESTAMP_COLUMN
            ]
            >= FINAL_TEST_START
        ).to_numpy()
    )

    if len(
        test_positions
    ) == 0:
        raise ValueError(
            "Could not locate final-test start"
        )

    first_test_position = int(
        test_positions[
            0
        ]
    )

    # Row immediately before final test is the
    # purge/context boundary.
    development_length = (
        first_test_position
        - PURGE_ROWS
    )

    return development_length


def build_fold_positions(
    development_length: int,
) -> list[
    tuple[
        np.ndarray,
        int,
        np.ndarray,
    ]
]:
    """
    Return:

        train positions,
        context position,
        validation positions

    matching the existing Phase 7 expanding-window design.
    """
    folds = []

    for fold in range(
        N_FOLDS
    ):
        train_end = (
            INITIAL_TRAIN_SIZE
            + (
                fold
                * STEP_SIZE
            )
        )

        context_position = (
            train_end
        )

        validation_start = (
            train_end
            + PURGE_ROWS
        )

        validation_end = (
            validation_start
            + VALIDATION_SIZE
        )

        if validation_end > development_length:
            raise ValueError(
                "Fold exceeds development period"
            )

        train_positions = np.arange(
            0,
            train_end,
        )

        validation_positions = np.arange(
            validation_start,
            validation_end,
        )

        folds.append(
            (
                train_positions,
                context_position,
                validation_positions,
            )
        )

    return folds


def ridge_predictions(
    df: pd.DataFrame,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> np.ndarray:
    """
    BNB Ridge predictions.

    Feature scaler is fitted on the outer training fold only.
    """
    train = df.iloc[
        train_positions
    ]

    validation = df.iloc[
        validation_positions
    ]

    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        train[
            FEATURE_COLUMNS
        ]
    )

    x_validation = scaler.transform(
        validation[
            FEATURE_COLUMNS
        ]
    )

    y_train = train[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    model = Ridge(
        alpha=BNB_RIDGE_ALPHA
    )

    model.fit(
        x_train,
        y_train,
    )

    return model.predict(
        x_validation
    )


def build_var_panel(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    frames = []

    for asset in ASSETS:
        frame = (
            datasets[
                asset
            ][
                [
                    TIMESTAMP_COLUMN,
                    "log_gk_volatility",
                    TARGET_COLUMN,
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_gk_volatility": (
                        f"{asset}_vol"
                    ),
                    TARGET_COLUMN: (
                        f"{asset}_target"
                    ),
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

    return (
        panel
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )


def var_fold_predictions(
    panel: pd.DataFrame,
    train_positions: np.ndarray,
    context_position: int,
    validation_positions: np.ndarray,
) -> pd.DataFrame:
    """
    Generate VAR t -> t+1 forecasts.

    At each validation forecast origin t:

        history includes observed volatility through t,
        VAR forecasts volatility for t+1.

    This matches the corrected Phase 7 VAR alignment.
    """
    var_columns = [
        f"{asset}_vol"
        for asset in ASSETS
    ]

    history_positions = np.concatenate(
        [
            train_positions,
            np.array(
                [
                    context_position
                ]
            ),
        ]
    )

    history = (
        panel.loc[
            history_positions,
            var_columns,
        ]
        .astype(
            float
        )
        .reset_index(
            drop=True
        )
    )

    predictions = []

    for position in validation_positions:
        current_values = (
            panel.loc[
                position,
                var_columns,
            ]
            .to_numpy(
                dtype=float
            )
        )

        current_row = pd.DataFrame(
            [
                current_values
            ],
            columns=var_columns,
            dtype=float,
        )

        # At the end of day t, volatility at t is observed
        # and can be included when forecasting t+1.
        history = pd.concat(
            [
                history,
                current_row,
            ],
            ignore_index=True,
        )

        # Defensive dtype enforcement for statsmodels.
        history = history.astype(
            float
        )

        model = VAR(
            history
        ).fit(
            VAR_LAG
        )

        lagged_values = (
            history[
                var_columns
            ]
            .iloc[
                -VAR_LAG:
            ]
            .to_numpy(
                dtype=float
            )
        )

        forecast = model.forecast(
            y=lagged_values,
            steps=1,
        )[
            0
        ]

        predictions.append(
            {
                TIMESTAMP_COLUMN: (
                    panel.loc[
                        position,
                        TIMESTAMP_COLUMN,
                    ]
                ),
                "BTC_forecast": float(
                    forecast[
                        0
                    ]
                ),
                "SOL_forecast": float(
                    forecast[
                        2
                    ]
                ),
            }
        )

    return pd.DataFrame(
        predictions
    )

def build_sequences(
    x: np.ndarray,
    y: np.ndarray,
    target_positions: np.ndarray,
    sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    x_sequences = []
    y_values = []

    for position in target_positions:
        sequence_start = (
            position
            - sequence_length
            + 1
        )

        if sequence_start < 0:
            raise ValueError(
                "Not enough history for sequence"
            )

        x_sequences.append(
            x[
                sequence_start:
                position
                + 1
            ]
        )

        y_values.append(
            y[
                position
            ]
        )

    return (
        np.asarray(
            x_sequences,
            dtype=np.float32,
        ),
        np.asarray(
            y_values,
            dtype=np.float32,
        ),
    )


def train_lstm_with_early_stopping(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[
    DevelopmentLSTM,
    int,
]:
    set_seed(
        seed
    )

    device = torch.device(
        "cpu"
    )

    model = DevelopmentLSTM(
        input_size=(
            x_train.shape[
                2
            ]
        ),
        hidden_size=(
            ETH_HIDDEN_SIZE
        ),
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ETH_LEARNING_RATE,
    )

    loss_function = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(
            x_train
        ),
        torch.from_numpy(
            y_train
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        dataset,
        batch_size=ETH_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    x_validation_tensor = (
        torch.from_numpy(
            x_validation
        ).to(
            device
        )
    )

    y_validation_tensor = (
        torch.from_numpy(
            y_validation
        ).to(
            device
        )
    )

    best_state = None
    best_loss = float(
        "inf"
    )
    best_epoch = 1

    epochs_without_improvement = 0

    for epoch in range(
        1,
        ETH_MAX_EPOCHS + 1,
    ):
        model.train()

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
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

        model.eval()

        with torch.no_grad():
            validation_predictions = model(
                x_validation_tensor
            )

            validation_loss = float(
                loss_function(
                    validation_predictions,
                    y_validation_tensor,
                ).item()
            )

        if validation_loss < (
            best_loss
            - 1e-8
        ):
            best_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value in (
                    model
                    .state_dict()
                    .items()
                )
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= ETH_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "LSTM training did not produce a best state"
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
    )


def train_lstm_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    seed: int,
) -> DevelopmentLSTM:
    """
    Refit using the complete outer training fold for the epoch
    count selected only from the inner training-validation split.
    """
    set_seed(
        seed
    )

    device = torch.device(
        "cpu"
    )

    model = DevelopmentLSTM(
        input_size=(
            x_train.shape[
                2
            ]
        ),
        hidden_size=(
            ETH_HIDDEN_SIZE
        ),
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ETH_LEARNING_RATE,
    )

    loss_function = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(
            x_train
        ),
        torch.from_numpy(
            y_train
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        dataset,
        batch_size=ETH_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    for _ in range(
        epochs
    ):
        model.train()

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
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

    return model


def eth_lstm_fold_predictions(
    df: pd.DataFrame,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the Phase 7 ETH procedure.

    For every outer fold:

    1. scale using outer training only;
    2. reserve an inner chronological holdout;
    3. find best epoch separately for each seed;
    4. refit the entire outer training fold;
    5. predict the untouched outer validation fold;
    6. average predictions across the three seeds.

    This prevents portfolio research from seeing validation
    targets when producing the forecasts.
    """
    outer_train_end = int(
        train_positions[
            -1
        ]
    ) + 1

    # Use the final 15% of the outer training fold as the
    # inner early-stopping validation period.
    #
    # If your original evaluate_robust_lstm_eth.py used a
    # different exact inner split, use that same value here.
    inner_validation_size = int(
        np.floor(
            outer_train_end
            * 0.15
        )
    )

    if inner_validation_size < (
        ETH_SEQUENCE_LENGTH
        + 1
    ):
        raise ValueError(
            "Inner validation period too small"
        )

    inner_train_end = (
        outer_train_end
        - inner_validation_size
        - PURGE_ROWS
    )

    inner_validation_start = (
        inner_train_end
        + PURGE_ROWS
    )

    x_values = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_values = (
        df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    feature_scaler = (
        StandardScaler()
    )

    feature_scaler.fit(
        x_values[
            :outer_train_end
        ]
    )

    scaled_x = (
        feature_scaler.transform(
            x_values
        )
    )

    target_scaler = (
        StandardScaler()
    )

    target_scaler.fit(
        y_values[
            :outer_train_end
        ].reshape(
            -1,
            1,
        )
    )

    scaled_y = (
        target_scaler.transform(
            y_values.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )

    inner_train_targets = np.arange(
        ETH_SEQUENCE_LENGTH
        - 1,
        inner_train_end,
    )

    inner_validation_targets = np.arange(
        inner_validation_start,
        outer_train_end,
    )

    (
        inner_x_train,
        inner_y_train,
    ) = build_sequences(
        scaled_x,
        scaled_y,
        inner_train_targets,
        ETH_SEQUENCE_LENGTH,
    )

    (
        inner_x_validation,
        inner_y_validation,
    ) = build_sequences(
        scaled_x,
        scaled_y,
        inner_validation_targets,
        ETH_SEQUENCE_LENGTH,
    )

    outer_train_targets = np.arange(
        ETH_SEQUENCE_LENGTH
        - 1,
        outer_train_end,
    )

    (
        outer_x_train,
        outer_y_train,
    ) = build_sequences(
        scaled_x,
        scaled_y,
        outer_train_targets,
        ETH_SEQUENCE_LENGTH,
    )

    (
        outer_x_validation,
        _,
    ) = build_sequences(
        scaled_x,
        scaled_y,
        validation_positions,
        ETH_SEQUENCE_LENGTH,
    )

    seed_predictions = []

    for seed in ETH_SEEDS:
        _, best_epoch = (
            train_lstm_with_early_stopping(
                inner_x_train,
                inner_y_train,
                inner_x_validation,
                inner_y_validation,
                seed,
            )
        )

        model = (
            train_lstm_fixed_epochs(
                outer_x_train,
                outer_y_train,
                epochs=best_epoch,
                seed=seed,
            )
        )

        model.eval()

        with torch.no_grad():
            scaled_predictions = (
                model(
                    torch.from_numpy(
                        outer_x_validation
                    )
                )
                .cpu()
                .numpy()
            )

        predictions = (
            target_scaler
            .inverse_transform(
                scaled_predictions.reshape(
                    -1,
                    1,
                )
            )
            .ravel()
        )

        seed_predictions.append(
            predictions
        )

    return np.mean(
        np.vstack(
            seed_predictions
        ),
        axis=0,
    )


def main() -> None:
    datasets = (
        load_all_datasets()
    )

    reference = datasets[
        "BTC"
    ]

    development_length = (
        get_development_length(
            reference
        )
    )

    development_end = (
        reference.iloc[
            development_length
            - 1
        ][
            TIMESTAMP_COLUMN
        ]
    )

    print(
        "\nDevelopment champion forecast generation"
    )

    print(
        "Development rows:",
        development_length,
    )

    print(
        "Development end:",
        development_end,
    )

    print(
        "\nFrozen champion architecture:"
    )

    print(
        "BTC -> VAR lag 7"
    )

    print(
        "ETH -> LSTM seq 7 / hidden 32 / lr 0.0005"
    )

    print(
        "SOL -> VAR lag 7"
    )

    print(
        "BNB -> Ridge alpha 100"
    )

    folds = build_fold_positions(
        development_length
    )

    var_panel = build_var_panel(
        datasets
    )

    fold_outputs = []

    for fold_number, (
        train_positions,
        context_position,
        validation_positions,
    ) in enumerate(
        folds,
        start=1,
    ):
        print(
            f"\nGenerating fold {fold_number}/{N_FOLDS}"
        )

        validation_dates = (
            reference.iloc[
                validation_positions
            ][
                TIMESTAMP_COLUMN
            ]
            .reset_index(
                drop=True
            )
        )

        var_predictions = (
            var_fold_predictions(
                panel=var_panel,
                train_positions=(
                    train_positions
                ),
                context_position=(
                    context_position
                ),
                validation_positions=(
                    validation_positions
                ),
            )
        )

        bnb_predictions = (
            ridge_predictions(
                df=datasets[
                    "BNB"
                ],
                train_positions=(
                    train_positions
                ),
                validation_positions=(
                    validation_positions
                ),
            )
        )

        eth_predictions = (
            eth_lstm_fold_predictions(
                df=datasets[
                    "ETH"
                ],
                train_positions=(
                    train_positions
                ),
                validation_positions=(
                    validation_positions
                ),
            )
        )

        if not (
            len(
                validation_dates
            )
            == len(
                var_predictions
            )
            == len(
                bnb_predictions
            )
            == len(
                eth_predictions
            )
        ):
            raise ValueError(
                "Fold prediction lengths do not match"
            )

        fold_output = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: (
                    validation_dates
                ),
                "BTC_forecast": (
                    var_predictions[
                        "BTC_forecast"
                    ].to_numpy()
                ),
                "ETH_forecast": (
                    eth_predictions
                ),
                "SOL_forecast": (
                    var_predictions[
                        "SOL_forecast"
                    ].to_numpy()
                ),
                "BNB_forecast": (
                    bnb_predictions
                ),
                "fold": fold_number,
            }
        )

        fold_outputs.append(
            fold_output
        )

        print(
            "Rows:",
            len(
                fold_output
            ),
        )

        print(
            "Start:",
            fold_output[
                TIMESTAMP_COLUMN
            ].iloc[
                0
            ],
        )

        print(
            "End:",
            fold_output[
                TIMESTAMP_COLUMN
            ].iloc[
                -1
            ],
        )

    predictions = (
        pd.concat(
            fold_outputs,
            ignore_index=True,
        )
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        predictions
    ) != (
        N_FOLDS
        * VALIDATION_SIZE
    ):
        raise ValueError(
            "Expected "
            f"{N_FOLDS * VALIDATION_SIZE} "
            "development predictions, got "
            f"{len(predictions)}"
        )

    if predictions[
        TIMESTAMP_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Duplicate prediction timestamps detected"
        )

    forecast_columns = [
        f"{asset}_forecast"
        for asset in ASSETS
    ]

    if predictions[
        forecast_columns
    ].isna().any().any():
        raise ValueError(
            "Missing development forecasts detected"
        )

    output_path = Path(
        "reports/metrics/"
        "development_champion_predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nDevelopment forecast generation complete."
    )

    print(
        "Total OOS rows:",
        len(
            predictions
        ),
    )

    print(
        "Start:",
        predictions[
            TIMESTAMP_COLUMN
        ].iloc[
            0
        ],
    )

    print(
        "End:",
        predictions[
            TIMESTAMP_COLUMN
        ].iloc[
            -1
        ],
    )

    print(
        "Saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()