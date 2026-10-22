from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class VolatilitySequenceDataset(Dataset):
    """
    PyTorch dataset for volatility sequence forecasting.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape "
                "(samples, sequence_length, features)"
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be one-dimensional"
            )

        if len(x) != len(y):
            raise ValueError(
                "x and y must contain the same number of samples"
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "x must contain only finite values"
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "y must contain only finite values"
            )

        self.x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.y[index],
        )


class VolatilityLSTM(nn.Module):
    """
    LSTM regression model for next-day volatility forecasting.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_features < 1:
            raise ValueError(
                "n_features must be at least 1"
            )

        if hidden_size < 1:
            raise ValueError(
                "hidden_size must be at least 1"
            )

        if num_layers < 1:
            raise ValueError(
                "num_layers must be at least 1"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        effective_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use the final LSTM hidden representation to predict
        next-day log volatility.
        """
        lstm_output, _ = self.lstm(
            x
        )

        final_hidden = lstm_output[
            :,
            -1,
            :,
        ]

        predictions = self.output(
            final_hidden
        )

        return predictions.squeeze(
            -1
        )