import numpy as np
import pytest
import torch

from crypto_volatility.models.lstm import (
    VolatilityLSTM,
    VolatilitySequenceDataset,
)


def make_sequence_arrays():
    rng = np.random.default_rng(
        42
    )

    x = rng.normal(
        size=(
            20,
            10,
            7,
        )
    )

    y = rng.normal(
        size=20
    )

    return x, y


def test_sequence_dataset_returns_expected_length():
    x, y = make_sequence_arrays()

    dataset = VolatilitySequenceDataset(
        x=x,
        y=y,
    )

    assert len(dataset) == 20


def test_sequence_dataset_returns_tensors():
    x, y = make_sequence_arrays()

    dataset = VolatilitySequenceDataset(
        x=x,
        y=y,
    )

    x_item, y_item = dataset[0]

    assert isinstance(
        x_item,
        torch.Tensor,
    )

    assert isinstance(
        y_item,
        torch.Tensor,
    )

    assert x_item.shape == (
        10,
        7,
    )

    assert y_item.ndim == 0


def test_lstm_forward_returns_expected_shape():
    model = VolatilityLSTM(
        n_features=7,
        hidden_size=16,
        num_layers=1,
    )

    x = torch.randn(
        8,
        10,
        7,
    )

    predictions = model(
        x
    )

    assert predictions.shape == (
        8,
    )


def test_lstm_forward_returns_finite_values():
    model = VolatilityLSTM(
        n_features=7,
        hidden_size=16,
        num_layers=1,
    )

    x = torch.randn(
        8,
        10,
        7,
    )

    predictions = model(
        x
    )

    assert torch.isfinite(
        predictions
    ).all()


def test_dataset_rejects_invalid_x_shape():
    x = np.ones(
        (
            20,
            7,
        )
    )

    y = np.ones(
        20
    )

    with pytest.raises(
        ValueError,
        match="x must have shape",
    ):
        VolatilitySequenceDataset(
            x=x,
            y=y,
        )


def test_dataset_rejects_mismatched_lengths():
    x = np.ones(
        (
            20,
            10,
            7,
        )
    )

    y = np.ones(
        19
    )

    with pytest.raises(
        ValueError,
        match="same number",
    ):
        VolatilitySequenceDataset(
            x=x,
            y=y,
        )


def test_lstm_rejects_invalid_feature_count():
    with pytest.raises(
        ValueError,
        match="n_features must be at least 1",
    ):
        VolatilityLSTM(
            n_features=0,
        )


def test_lstm_disables_dropout_for_single_layer():
    model = VolatilityLSTM(
        n_features=7,
        num_layers=1,
        dropout=0.3,
    )

    assert model.lstm.dropout == 0.0