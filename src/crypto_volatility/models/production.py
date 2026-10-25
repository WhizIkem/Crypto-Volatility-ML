from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from crypto_volatility.models.lstm import VolatilityLSTM

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


@dataclass(frozen=True)
class ProductionForecasts:
    """
    Production next-day log-volatility forecasts.
    """

    BTC: float
    ETH: float
    SOL: float
    BNB: float

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "BTC": self.BTC,
                "ETH": self.ETH,
                "SOL": self.SOL,
                "BNB": self.BNB,
            },
            name="forecast_log_volatility",
            dtype=float,
        )


@dataclass(frozen=True)
class ProductionModelBundle:
    """
    Loaded production models and preprocessing artifacts.
    """

    var_model: Any
    bnb_model: Any
    bnb_feature_scaler: StandardScaler
    eth_model: VolatilityLSTM
    eth_feature_scaler: StandardScaler
    eth_target_scaler: StandardScaler
    manifest: dict[str, Any]


def load_pickle(
    path: Path,
) -> Any:
    """
    Load a persisted Python model or preprocessing object.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    with path.open(
        "rb"
    ) as file:
        return pickle.load(
            file
        )


def load_manifest(
    production_dir: Path,
) -> dict[str, Any]:
    """
    Load the production manifest.
    """
    path = (
        production_dir
        / "manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Production manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(
            file
        )

    return manifest


def load_eth_lstm(
    production_dir: Path,
) -> VolatilityLSTM:
    """
    Reconstruct the ETH LSTM and load its production weights.
    """
    checkpoint_path = (
        production_dir
        / "eth_lstm.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ETH LSTM checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    required = {
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "model_state_dict",
    }

    missing = (
        required
        - set(
            checkpoint
        )
    )

    if missing:
        raise ValueError(
            "ETH checkpoint missing fields: "
            f"{sorted(missing)}"
        )

    model = VolatilityLSTM(
        n_features=int(
            checkpoint[
                "input_size"
            ]
        ),
        hidden_size=int(
            checkpoint[
                "hidden_size"
            ]
        ),
        num_layers=int(
            checkpoint[
                "num_layers"
            ]
        ),
        dropout=float(
            checkpoint[
                "dropout"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        torch.device(
            "cpu"
        )
    )

    model.eval()

    return model


def load_production_models(
    production_dir: str | Path = "models/production",
) -> ProductionModelBundle:
    """
    Load all production forecasting artifacts.

    No model fitting occurs here.
    """
    production_path = Path(
        production_dir
    )

    manifest = load_manifest(
        production_path
    )

    var_model = load_pickle(
        production_path
        / "var_model.pkl"
    )

    bnb_model = load_pickle(
        production_path
        / "bnb_ridge_model.pkl"
    )

    bnb_feature_scaler = (
        load_pickle(
            production_path
            / "bnb_feature_scaler.pkl"
        )
    )

    eth_feature_scaler = (
        load_pickle(
            production_path
            / "eth_feature_scaler.pkl"
        )
    )

    eth_target_scaler = (
        load_pickle(
            production_path
            / "eth_target_scaler.pkl"
        )
    )

    eth_model = load_eth_lstm(
        production_path
    )

    return ProductionModelBundle(
        var_model=var_model,
        bnb_model=bnb_model,
        bnb_feature_scaler=(
            bnb_feature_scaler
        ),
        eth_model=eth_model,
        eth_feature_scaler=(
            eth_feature_scaler
        ),
        eth_target_scaler=(
            eth_target_scaler
        ),
        manifest=manifest,
    )


def predict_var(
    bundle: ProductionModelBundle,
    volatility_history: pd.DataFrame,
) -> tuple[
    float,
    float,
]:
    """
    Generate BTC and SOL forecasts from the production VAR.

    Parameters
    ----------
    volatility_history:
        Historical log Garman-Klass volatility with columns:

            BTC
            ETH
            SOL
            BNB

        Must contain at least 7 observations.
    """
    missing = (
        set(
            ASSETS
        )
        - set(
            volatility_history.columns
        )
    )

    if missing:
        raise ValueError(
            "VAR history missing columns: "
            f"{sorted(missing)}"
        )

    lag = int(
        bundle.manifest[
            "champions"
        ][
            "BTC"
        ][
            "lag"
        ]
    )

    if len(
        volatility_history
    ) < lag:
        raise ValueError(
            f"VAR requires at least {lag} history rows"
        )

    values = (
        volatility_history[
            ASSETS
        ]
        .tail(
            lag
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "VAR history contains non-finite values"
        )

    forecast = (
        bundle
        .var_model
        .forecast(
            y=values,
            steps=1,
        )[
            0
        ]
    )

    btc_forecast = float(
        forecast[
            ASSETS.index(
                "BTC"
            )
        ]
    )

    sol_forecast = float(
        forecast[
            ASSETS.index(
                "SOL"
            )
        ]
    )

    return (
        btc_forecast,
        sol_forecast,
    )


def predict_bnb(
    bundle: ProductionModelBundle,
    latest_features: pd.Series,
) -> float:
    """
    Generate the BNB next-day log-volatility forecast.
    """
    missing = (
        set(
            FEATURE_COLUMNS
        )
        - set(
            latest_features.index
        )
    )

    if missing:
        raise ValueError(
            "BNB features missing columns: "
            f"{sorted(missing)}"
        )

    values = (
        latest_features[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
        .reshape(
            1,
            -1,
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "BNB features contain non-finite values"
        )

    scaled = (
        bundle
        .bnb_feature_scaler
        .transform(
            values
        )
    )

    prediction = (
        bundle
        .bnb_model
        .predict(
            scaled
        )[
            0
        ]
    )

    return float(
        prediction
    )


def predict_eth(
    bundle: ProductionModelBundle,
    feature_history: pd.DataFrame,
) -> float:
    """
    Generate the ETH next-day log-volatility forecast.

    Uses the final seven feature rows.
    """
    sequence_length = int(
        bundle.manifest[
            "champions"
        ][
            "ETH"
        ][
            "sequence_length"
        ]
    )

    missing = (
        set(
            FEATURE_COLUMNS
        )
        - set(
            feature_history.columns
        )
    )

    if missing:
        raise ValueError(
            "ETH feature history missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        feature_history
    ) < sequence_length:
        raise ValueError(
            "ETH requires at least "
            f"{sequence_length} feature rows"
        )

    values = (
        feature_history[
            FEATURE_COLUMNS
        ]
        .tail(
            sequence_length
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "ETH feature history contains non-finite values"
        )

    scaled = (
        bundle
        .eth_feature_scaler
        .transform(
            values
        )
        .astype(
            np.float32
        )
    )

    tensor = (
        torch.from_numpy(
            scaled
        )
        .unsqueeze(
            0
        )
    )

    bundle.eth_model.eval()

    with torch.no_grad():
        scaled_prediction = (
            bundle
            .eth_model(
                tensor
            )
            .cpu()
            .numpy()
            .reshape(
                -1,
                1,
            )
        )

    prediction = (
        bundle
        .eth_target_scaler
        .inverse_transform(
            scaled_prediction
        )[
            0,
            0,
        ]
    )

    return float(
        prediction
    )


def predict_all_assets(
    bundle: ProductionModelBundle,
    *,
    volatility_history: pd.DataFrame,
    eth_feature_history: pd.DataFrame,
    bnb_latest_features: pd.Series,
) -> ProductionForecasts:
    """
    Generate all four production forecasts.

    BTC -> VAR
    ETH -> LSTM
    SOL -> VAR
    BNB -> Ridge
    """
    (
        btc_forecast,
        sol_forecast,
    ) = predict_var(
        bundle=bundle,
        volatility_history=(
            volatility_history
        ),
    )

    eth_forecast = (
        predict_eth(
            bundle=bundle,
            feature_history=(
                eth_feature_history
            ),
        )
    )

    bnb_forecast = (
        predict_bnb(
            bundle=bundle,
            latest_features=(
                bnb_latest_features
            ),
        )
    )

    forecasts = ProductionForecasts(
        BTC=btc_forecast,
        ETH=eth_forecast,
        SOL=sol_forecast,
        BNB=bnb_forecast,
    )

    values = (
        forecasts
        .to_series()
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Production forecasts contain non-finite values"
        )

    return forecasts