from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]


@dataclass
class PortfolioState:
    """
    Persistent paper-trading portfolio state.

    Quantities represent units of each crypto asset.

    Cash is denominated in the portfolio quote currency,
    currently USDT.
    """

    cash: float
    quantities: dict[str, float]
    last_prices: dict[str, float]
    created_at: str
    updated_at: str
    last_forecast_timestamp: str | None = None
    last_rebalance_timestamp: str | None = None

    def validate(self) -> None:
        """
        Validate state consistency.
        """
        if not np.isfinite(
            self.cash
        ):
            raise ValueError(
                "cash must be finite"
            )

        if self.cash < 0:
            raise ValueError(
                "cash cannot be negative"
            )

        missing_quantities = (
            set(ASSETS)
            - set(
                self.quantities
            )
        )

        if missing_quantities:
            raise ValueError(
                "quantities missing assets: "
                f"{sorted(missing_quantities)}"
            )

        missing_prices = (
            set(ASSETS)
            - set(
                self.last_prices
            )
        )

        if missing_prices:
            raise ValueError(
                "last_prices missing assets: "
                f"{sorted(missing_prices)}"
            )

        for asset in ASSETS:
            quantity = float(
                self.quantities[
                    asset
                ]
            )

            price = float(
                self.last_prices[
                    asset
                ]
            )

            if not np.isfinite(
                quantity
            ):
                raise ValueError(
                    f"{asset} quantity must be finite"
                )

            if quantity < 0:
                raise ValueError(
                    f"{asset} quantity cannot be negative"
                )

            if not np.isfinite(
                price
            ):
                raise ValueError(
                    f"{asset} price must be finite"
                )

            if price < 0:
                raise ValueError(
                    f"{asset} price cannot be negative"
                )

    def asset_values(
        self,
        prices: pd.Series | None = None,
    ) -> pd.Series:
        """
        Calculate current market value of each risky asset.
        """
        price_series = (
            self.price_series()
            if prices is None
            else prices.reindex(
                ASSETS
            ).astype(
                float
            )
        )

        if price_series.isna().any():
            raise ValueError(
                "prices missing required assets"
            )

        if (
            price_series
            < 0
        ).any():
            raise ValueError(
                "prices cannot be negative"
            )

        quantities = pd.Series(
            {
                asset: float(
                    self.quantities[
                        asset
                    ]
                )
                for asset in ASSETS
            },
            dtype=float,
        )

        return (
            quantities
            * price_series
        )

    def equity(
        self,
        prices: pd.Series | None = None,
    ) -> float:
        """
        Calculate total portfolio equity.
        """
        risky_value = float(
            self.asset_values(
                prices
            ).sum()
        )

        return float(
            self.cash
            + risky_value
        )

    def risky_weights(
        self,
        prices: pd.Series | None = None,
    ) -> pd.Series:
        """
        Calculate current risky-asset portfolio weights.
        """
        values = self.asset_values(
            prices
        )

        total_equity = self.equity(
            prices
        )

        if total_equity <= 0:
            raise ValueError(
                "portfolio equity must be positive"
            )

        return (
            values
            / total_equity
        ).astype(
            float
        )

    def cash_weight(
        self,
        prices: pd.Series | None = None,
    ) -> float:
        """
        Calculate current cash weight.
        """
        total_equity = self.equity(
            prices
        )

        if total_equity <= 0:
            raise ValueError(
                "portfolio equity must be positive"
            )

        return float(
            self.cash
            / total_equity
        )

    def quantity_series(
        self,
    ) -> pd.Series:
        """
        Return asset quantities as a Series.
        """
        return pd.Series(
            {
                asset: float(
                    self.quantities[
                        asset
                    ]
                )
                for asset in ASSETS
            },
            dtype=float,
        )

    def price_series(
        self,
    ) -> pd.Series:
        """
        Return last observed prices as a Series.
        """
        return pd.Series(
            {
                asset: float(
                    self.last_prices[
                        asset
                    ]
                )
                for asset in ASSETS
            },
            dtype=float,
        )


def utc_now_iso() -> str:
    """
    Return current UTC timestamp in ISO-8601 format.
    """
    return datetime.now(
        UTC
    ).isoformat()


def create_initial_portfolio_state(
    initial_cash: float,
) -> PortfolioState:
    """
    Create a new paper portfolio starting fully in cash.
    """
    if not np.isfinite(
        initial_cash
    ):
        raise ValueError(
            "initial_cash must be finite"
        )

    if initial_cash <= 0:
        raise ValueError(
            "initial_cash must be positive"
        )

    timestamp = utc_now_iso()

    state = PortfolioState(
        cash=float(
            initial_cash
        ),
        quantities={
            asset: 0.0
            for asset in ASSETS
        },
        last_prices={
            asset: 0.0
            for asset in ASSETS
        },
        created_at=timestamp,
        updated_at=timestamp,
        last_forecast_timestamp=None,
        last_rebalance_timestamp=None,
    )

    state.validate()

    return state


def update_market_prices(
    state: PortfolioState,
    prices: pd.Series,
) -> PortfolioState:
    """
    Update the state's latest observed market prices.
    """
    aligned = (
        prices
        .reindex(
            ASSETS
        )
        .astype(
            float
        )
    )

    if aligned.isna().any():
        raise ValueError(
            "prices missing required assets"
        )

    if not np.isfinite(
        aligned.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "prices must contain only finite values"
        )

    if (
        aligned
        <= 0
    ).any():
        raise ValueError(
            "all market prices must be positive"
        )

    state.last_prices = {
        asset: float(
            aligned[
                asset
            ]
        )
        for asset in ASSETS
    }

    state.updated_at = (
        utc_now_iso()
    )

    state.validate()

    return state


def target_position_values(
    *,
    portfolio_equity: float,
    target_weights: pd.Series,
) -> pd.Series:
    """
    Convert portfolio target weights into target position values.
    """
    if not np.isfinite(
        portfolio_equity
    ):
        raise ValueError(
            "portfolio_equity must be finite"
        )

    if portfolio_equity <= 0:
        raise ValueError(
            "portfolio_equity must be positive"
        )

    aligned = (
        target_weights
        .reindex(
            ASSETS
        )
        .astype(
            float
        )
    )

    if aligned.isna().any():
        raise ValueError(
            "target_weights missing required assets"
        )

    if (
        aligned
        < 0
    ).any():
        raise ValueError(
            "target_weights cannot be negative"
        )

    if float(
        aligned.sum()
    ) > 1.0 + 1e-10:
        raise ValueError(
            "target risky weights cannot exceed 100%"
        )

    return (
        aligned
        * portfolio_equity
    )


def target_quantities(
    *,
    portfolio_equity: float,
    target_weights: pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """
    Convert target portfolio weights into crypto quantities.
    """
    target_values = (
        target_position_values(
            portfolio_equity=(
                portfolio_equity
            ),
            target_weights=(
                target_weights
            ),
        )
    )

    aligned_prices = (
        prices
        .reindex(
            ASSETS
        )
        .astype(
            float
        )
    )

    if aligned_prices.isna().any():
        raise ValueError(
            "prices missing required assets"
        )

    if (
        aligned_prices
        <= 0
    ).any():
        raise ValueError(
            "all prices must be positive"
        )

    return (
        target_values
        / aligned_prices
    )


def trade_quantities(
    *,
    state: PortfolioState,
    target_weights: pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """
    Calculate quantity changes required to reach target weights.

    Positive quantity:
        buy

    Negative quantity:
        sell
    """
    equity = state.equity(
        prices
    )

    desired = target_quantities(
        portfolio_equity=equity,
        target_weights=(
            target_weights
        ),
        prices=prices,
    )

    current = (
        state.quantity_series()
    )

    return (
        desired
        - current
    )


def save_portfolio_state(
    state: PortfolioState,
    path: str | Path,
) -> None:
    """
    Persist portfolio state atomically as JSON.
    """
    state.validate()

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        destination
        .with_suffix(
            destination.suffix
            + ".tmp"
        )
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(
                state
            ),
            file,
            indent=2,
        )

    temporary.replace(
        destination
    )


def load_portfolio_state(
    path: str | Path,
) -> PortfolioState:
    """
    Load persisted paper portfolio state.
    """
    source = Path(
        path
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Portfolio state not found: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    state = PortfolioState(
        cash=float(
            payload[
                "cash"
            ]
        ),
        quantities={
            asset: float(
                payload[
                    "quantities"
                ][
                    asset
                ]
            )
            for asset in ASSETS
        },
        last_prices={
            asset: float(
                payload[
                    "last_prices"
                ][
                    asset
                ]
            )
            for asset in ASSETS
        },
        created_at=str(
            payload[
                "created_at"
            ]
        ),
        updated_at=str(
            payload[
                "updated_at"
            ]
        ),
        last_forecast_timestamp=(
            payload.get(
                "last_forecast_timestamp"
            )
        ),
        last_rebalance_timestamp=(
            payload.get(
                "last_rebalance_timestamp"
            )
        ),
    )

    state.validate()

    return state