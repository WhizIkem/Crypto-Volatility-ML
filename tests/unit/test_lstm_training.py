import numpy as np
import pytest
import torch

from crypto_volatility.models.lstm import (
    VolatilityLSTM,
    VolatilitySequenceDataset,
)
from crypto_volatility.models.lstm_training import (
    predict_lstm,
    train_lstm_fixed_epochs,
    train_lstm_model,
)


def make_lstm_datasets():
    rng = np.random.default_rng(
        42
    )

    x_train = rng.normal(
        size=(
            80,
            5,
            3,
        )
    )

    y_train = (
        0.5
        * x_train[:, -1, 0]
        - 0.2
        * x_train[:, -1, 1]
    )

    x_validation = rng.normal(
        size=(
            20,
            5,
            3,
        )
    )

    y_validation = (
        0.5
        * x_validation[:, -1, 0]
        - 0.2
        * x_validation[:, -1, 1]
    )

    train_dataset = (
        VolatilitySequenceDataset(
            x=x_train,
            y=y_train,
        )
    )

    validation_dataset = (
        VolatilitySequenceDataset(
            x=x_validation,
            y=y_validation,
        )
    )

    return (
        train_dataset,
        validation_dataset,
    )


def test_train_lstm_model_returns_training_result():
    torch.manual_seed(
        42
    )

    (
        train_dataset,
        validation_dataset,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
        hidden_size=8,
    )

    result = train_lstm_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=10,
        patience=3,
        device="cpu",
    )

    assert result.best_epoch >= 1

    assert len(
        result.train_losses
    ) >= 1

    assert len(
        result.validation_losses
    ) >= 1

    assert np.isfinite(
        result.best_validation_loss
    )


def test_train_lstm_respects_max_epochs():
    torch.manual_seed(
        42
    )

    (
        train_dataset,
        validation_dataset,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
        hidden_size=8,
    )

    result = train_lstm_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        max_epochs=4,
        patience=4,
        device="cpu",
    )

    assert len(
        result.train_losses
    ) <= 4

    assert len(
        result.validation_losses
    ) <= 4


def test_predict_lstm_returns_expected_shape():
    torch.manual_seed(
        42
    )

    (
        train_dataset,
        validation_dataset,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
        hidden_size=8,
    )

    result = train_lstm_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        max_epochs=3,
        patience=2,
        device="cpu",
    )

    predictions = predict_lstm(
        model=result.model,
        dataset=validation_dataset,
        device="cpu",
    )

    assert predictions.shape == (
        len(
            validation_dataset
        ),
    )

    assert np.isfinite(
        predictions
    ).all()


def test_train_lstm_rejects_invalid_learning_rate():
    (
        train_dataset,
        validation_dataset,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
    )

    with pytest.raises(
        ValueError,
        match="learning_rate must be greater than zero",
    ):
        train_lstm_model(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            learning_rate=0.0,
            device="cpu",
        )


def test_train_lstm_rejects_invalid_patience():
    (
        train_dataset,
        validation_dataset,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
    )

    with pytest.raises(
        ValueError,
        match="patience must be at least 1",
    ):
        train_lstm_model(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            patience=0,
            device="cpu",
        )

def test_train_lstm_fixed_epochs_returns_model():
    torch.manual_seed(
        42
    )

    (
        train_dataset,
        _,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
        hidden_size=8,
    )

    fitted_model = train_lstm_fixed_epochs(
        model=model,
        train_dataset=train_dataset,
        epochs=3,
        learning_rate=0.01,
        batch_size=16,
        device="cpu",
    )

    assert isinstance(
        fitted_model,
        VolatilityLSTM,
    )


def test_train_lstm_fixed_epochs_changes_parameters():
    torch.manual_seed(
        42
    )

    (
        train_dataset,
        _,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
        hidden_size=8,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    fitted_model = train_lstm_fixed_epochs(
        model=model,
        train_dataset=train_dataset,
        epochs=2,
        learning_rate=0.01,
        batch_size=16,
        device="cpu",
    )

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter
        in fitted_model.named_parameters()
    )

    assert changed


def test_train_lstm_fixed_epochs_rejects_invalid_epochs():
    (
        train_dataset,
        _,
    ) = make_lstm_datasets()

    model = VolatilityLSTM(
        n_features=3,
    )

    with pytest.raises(
        ValueError,
        match="epochs must be at least 1",
    ):
        train_lstm_fixed_epochs(
            model=model,
            train_dataset=train_dataset,
            epochs=0,
            device="cpu",
        )