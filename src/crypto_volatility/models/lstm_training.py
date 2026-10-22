from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from crypto_volatility.models.lstm import (
    VolatilityLSTM,
    VolatilitySequenceDataset,
)


@dataclass(frozen=True)
class LSTMTrainingResult:
    model: VolatilityLSTM
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float


def train_lstm_model(
    model: VolatilityLSTM,
    train_dataset: VolatilitySequenceDataset,
    validation_dataset: VolatilitySequenceDataset,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 10,
    weight_decay: float = 0.0,
    device: str | None = None,
) -> LSTMTrainingResult:
    """
    Train an LSTM and monitor validation loss for early stopping.

    Training samples are shuffled between epochs because each input
    sample is already an independent, chronologically ordered sequence.

    Validation samples remain unshuffled.
    """
    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero"
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    if max_epochs < 1:
        raise ValueError(
            "max_epochs must be at least 1"
        )

    if patience < 1:
        raise ValueError(
            "patience must be at least 1"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative"
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "train_dataset cannot be empty"
        )

    if len(validation_dataset) == 0:
        raise ValueError(
            "validation_dataset cannot be empty"
        )

    if device is None:
        device = "cpu"

    torch_device = torch.device(
        device
    )

    model = model.to(
        torch_device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_losses = []
    validation_losses = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        train_loss_total = 0.0
        train_observations = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(
                torch_device
            )

            y_batch = y_batch.to(
                torch_device
            )

            optimizer.zero_grad()

            predictions = model(
                x_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_size_actual = len(
                x_batch
            )

            train_loss_total += (
                loss.item()
                * batch_size_actual
            )

            train_observations += (
                batch_size_actual
            )

        train_loss = (
            train_loss_total
            / train_observations
        )

        train_losses.append(
            float(
                train_loss
            )
        )

        model.eval()

        validation_loss_total = 0.0
        validation_observations = 0

        with torch.no_grad():
            for (
                x_batch,
                y_batch,
            ) in validation_loader:
                x_batch = x_batch.to(
                    torch_device
                )

                y_batch = y_batch.to(
                    torch_device
                )

                predictions = model(
                    x_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                batch_size_actual = len(
                    x_batch
                )

                validation_loss_total += (
                    loss.item()
                    * batch_size_actual
                )

                validation_observations += (
                    batch_size_actual
                )

        validation_loss = (
            validation_loss_total
            / validation_observations
        )

        validation_losses.append(
            float(
                validation_loss
            )
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = float(
                validation_loss
            )

            best_epoch = epoch

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            break

    model.load_state_dict(
        best_state
    )

    return LSTMTrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )


def train_lstm_fixed_epochs(
    model: VolatilityLSTM,
    train_dataset: VolatilitySequenceDataset,
    epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    device: str | None = None,
) -> VolatilityLSTM:
    """
    Train an LSTM for a fixed number of epochs.
    """
    if epochs < 1:
        raise ValueError(
            "epochs must be at least 1"
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero"
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative"
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "train_dataset cannot be empty"
        )

    if device is None:
        device = "cpu"

    torch_device = torch.device(
        device
    )

    model = model.to(
        torch_device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in range(
        epochs
    ):
        model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(
                torch_device
            )

            y_batch = y_batch.to(
                torch_device
            )

            optimizer.zero_grad()

            predictions = model(
                x_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            optimizer.step()

    return model


def predict_lstm(
    model: VolatilityLSTM,
    dataset: VolatilitySequenceDataset,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """
    Generate predictions from a fitted LSTM model.
    """
    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    if len(dataset) == 0:
        raise ValueError(
            "dataset cannot be empty"
        )

    if device is None:
        device = "cpu"

    torch_device = torch.device(
        device
    )

    model = model.to(
        torch_device
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []

    model.eval()

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(
                torch_device
            )

            batch_predictions = model(
                x_batch
            )

            predictions.append(
                batch_predictions
                .cpu()
                .numpy()
            )

    return np.concatenate(
        predictions
    )