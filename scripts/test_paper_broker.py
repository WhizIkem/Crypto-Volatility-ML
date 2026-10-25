from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.trading.paper_broker import (
    simulate_paper_rebalance,
)
from crypto_volatility.trading.portfolio_state import (
    load_portfolio_state,
    save_portfolio_state,
    target_quantities,
    update_market_prices,
)

STATE_PATH = Path(
    "logs/paper_portfolio_state.json"
)

TRANSACTION_COST_BPS = 10.0


def main() -> None:
    print(
        "\nPaper-broker smoke test"
    )

    state = (
        load_portfolio_state(
            STATE_PATH
        )
    )

    prices = pd.Series(
        {
            "BTC": 80_888.16,
            "ETH": 2_358.82,
            "SOL": 102.38,
            "BNB": 740.00,
        },
        dtype=float,
    )

    update_market_prices(
        state,
        prices,
    )

    print(
        "\nStarting equity:",
        f"{state.equity():,.2f}",
    )

    print(
        "Starting cash:",
        f"{state.cash:,.2f}",
    )

    target_weights = pd.Series(
        {
            "BTC": 0.185466,
            "ETH": 0.145625,
            "SOL": 0.103813,
            "BNB": 0.175483,
        },
        dtype=float,
    )

    targets = target_quantities(
        portfolio_equity=(
            state.equity()
        ),
        target_weights=(
            target_weights
        ),
        prices=prices,
    )

    result = simulate_paper_rebalance(
        state=state,
        target_quantities=(
            targets
        ),
        prices=prices,
        transaction_cost_bps=(
            TRANSACTION_COST_BPS
        ),
        rebalance_timestamp=(
            "2026-10-24T23:59:59.999000+00:00"
        ),
    )

    print(
        "\nExecuted trades:"
    )

    for trade in result.trades:
        print(
            f"{trade.side:4} "
            f"{trade.asset:3} "
            f"qty={trade.quantity:.8f} "
            f"price={trade.price:,.2f} "
            f"notional={trade.notional:,.2f} "
            f"cost={trade.transaction_cost:,.2f}"
        )

    print(
        "\nTotal transaction cost:",
        f"{result.total_transaction_cost:,.2f}",
    )

    print(
        "Ending cash:",
        f"{state.cash:,.2f}",
    )

    print(
        "Ending equity:",
        f"{state.equity():,.2f}",
    )

    print(
        "\nEnding quantities:"
    )

    print(
        state
        .quantity_series()
        .to_string()
    )

    print(
        "\nEnding risky weights:"
    )

    print(
        state
        .risky_weights()
        .to_string()
    )

    print(
        "\nEnding cash weight:",
        f"{state.cash_weight():.6f}",
    )

    save_portfolio_state(
        state,
        STATE_PATH,
    )

    print(
        "\nUpdated state saved to:",
        STATE_PATH,
    )

    print(
        "\nPaper-broker execution succeeded."
    )


if __name__ == "__main__":
    main()