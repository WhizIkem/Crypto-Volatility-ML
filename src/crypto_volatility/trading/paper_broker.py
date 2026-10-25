from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_volatility.trading.portfolio_state import (
    ASSETS,
    PortfolioState,
    utc_now_iso,
)


@dataclass(frozen=True)
class PaperTrade:
    """
    Simulated paper trade.
    """

    asset: str
    side: str
    quantity: float
    price: float
    notional: float
    transaction_cost: float


@dataclass(frozen=True)
class PaperExecutionResult:
    """
    Result of a paper rebalance.
    """

    trades: list[PaperTrade]
    total_transaction_cost: float
    state: PortfolioState


def validate_prices(
    prices: pd.Series,
) -> pd.Series:
    """
    Validate and align execution prices.
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
            "all prices must be positive"
        )

    return aligned


def validate_target_quantities(
    target_quantities: pd.Series,
) -> pd.Series:
    """
    Validate target asset quantities.
    """
    aligned = (
        target_quantities
        .reindex(
            ASSETS
        )
        .astype(
            float
        )
    )

    if aligned.isna().any():
        raise ValueError(
            "target_quantities missing required assets"
        )

    if not np.isfinite(
        aligned.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "target_quantities must contain only finite values"
        )

    if (
        aligned
        < 0
    ).any():
        raise ValueError(
            "target_quantities cannot be negative"
        )

    return aligned


def build_trade_quantities(
    *,
    state: PortfolioState,
    target_quantities: pd.Series,
) -> pd.Series:
    """
    Calculate quantity changes required to reach target positions.

    Positive:
        buy

    Negative:
        sell
    """
    current = (
        state.quantity_series()
    )

    target = (
        validate_target_quantities(
            target_quantities
        )
    )

    return (
        target
        - current
    )


def simulate_paper_rebalance(
    *,
    state: PortfolioState,
    target_quantities: pd.Series,
    prices: pd.Series,
    transaction_cost_bps: float = 10.0,
    rebalance_timestamp: str | None = None,
) -> PaperExecutionResult:
    """
    Simulate a long-only portfolio rebalance.

    Sells are processed before buys so sale proceeds are available
    for subsequent purchases.

    Transaction costs are proportional to traded notional.

    No leverage or negative cash is allowed.
    """
    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative"
        )

    state.validate()

    execution_prices = (
        validate_prices(
            prices
        )
    )

    targets = (
        validate_target_quantities(
            target_quantities
        )
    )

    current_quantities = (
        state.quantity_series()
    )

    differences = (
        targets
        - current_quantities
    )

    cost_rate = (
        transaction_cost_bps
        / 10_000.0
    )

    trades: list[
        PaperTrade
    ] = []

    total_cost = 0.0

    # --------------------------------------------------------------
    # Process sells first.
    # --------------------------------------------------------------

    for asset in ASSETS:
        quantity_change = float(
            differences[
                asset
            ]
        )

        if quantity_change >= 0:
            continue

        sell_quantity = abs(
            quantity_change
        )

        price = float(
            execution_prices[
                asset
            ]
        )

        notional = (
            sell_quantity
            * price
        )

        transaction_cost = (
            notional
            * cost_rate
        )

        state.quantities[
            asset
        ] -= sell_quantity

        state.cash += (
            notional
            - transaction_cost
        )

        total_cost += (
            transaction_cost
        )

        trades.append(
            PaperTrade(
                asset=asset,
                side="SELL",
                quantity=float(
                    sell_quantity
                ),
                price=price,
                notional=float(
                    notional
                ),
                transaction_cost=float(
                    transaction_cost
                ),
            )
        )

    # --------------------------------------------------------------
    # Process buys second.
    # --------------------------------------------------------------

    for asset in ASSETS:
        quantity_change = float(
            differences[
                asset
            ]
        )

        if quantity_change <= 0:
            continue

        buy_quantity = (
            quantity_change
        )

        price = float(
            execution_prices[
                asset
            ]
        )

        notional = (
            buy_quantity
            * price
        )

        transaction_cost = (
            notional
            * cost_rate
        )

        required_cash = (
            notional
            + transaction_cost
        )

        if required_cash > (
            state.cash
            + 1e-8
        ):
            raise ValueError(
                f"Insufficient cash to buy {asset}: "
                f"required={required_cash:.2f}, "
                f"available={state.cash:.2f}"
            )

        state.cash -= (
            required_cash
        )

        state.quantities[
            asset
        ] += (
            buy_quantity
        )

        total_cost += (
            transaction_cost
        )

        trades.append(
            PaperTrade(
                asset=asset,
                side="BUY",
                quantity=float(
                    buy_quantity
                ),
                price=price,
                notional=float(
                    notional
                ),
                transaction_cost=float(
                    transaction_cost
                ),
            )
        )

    state.last_prices = {
        asset: float(
            execution_prices[
                asset
            ]
        )
        for asset in ASSETS
    }

    state.updated_at = (
        utc_now_iso()
    )

    state.last_rebalance_timestamp = (
        rebalance_timestamp
        if rebalance_timestamp is not None
        else state.updated_at
    )

    if state.cash < -1e-8:
        raise ValueError(
            "Paper execution created negative cash"
        )

    if abs(
        state.cash
    ) < 1e-8:
        state.cash = 0.0

    state.validate()

    return PaperExecutionResult(
        trades=trades,
        total_transaction_cost=float(
            total_cost
        ),
        state=state,
    )