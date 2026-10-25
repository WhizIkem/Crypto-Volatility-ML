from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_volatility.trading.portfolio_state import (
    create_initial_portfolio_state,
    load_portfolio_state,
    save_portfolio_state,
    target_quantities,
    update_market_prices,
)

STATE_PATH = Path(
    "logs/paper_portfolio_state.json"
)

INITIAL_CAPITAL = 1_000_000.0


def main() -> None:
    print(
        "\nPortfolio-state persistence smoke test"
    )

    if STATE_PATH.exists():
        print(
            "\nLoading existing paper portfolio..."
        )

        state = (
            load_portfolio_state(
                STATE_PATH
            )
        )
    else:
        print(
            "\nCreating new paper portfolio..."
        )

        state = (
            create_initial_portfolio_state(
                INITIAL_CAPITAL
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
        "\nPortfolio equity:",
        f"{state.equity():,.2f}",
    )

    print(
        "Cash:",
        f"{state.cash:,.2f}",
    )

    print(
        "\nCurrent risky weights:"
    )

    print(
        state
        .risky_weights()
        .to_string()
    )

    print(
        "\nCash weight:",
        f"{state.cash_weight():.6f}",
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

    desired_quantities = (
        target_quantities(
            portfolio_equity=(
                state.equity()
            ),
            target_weights=(
                target_weights
            ),
            prices=prices,
        )
    )

    print(
        "\nExample target quantities for "
        f"{state.equity():,.2f} portfolio:"
    )

    print(
        desired_quantities
        .to_string()
    )

    save_portfolio_state(
        state,
        STATE_PATH,
    )

    reloaded = (
        load_portfolio_state(
            STATE_PATH
        )
    )

    print(
        "\nState saved to:",
        STATE_PATH,
    )

    print(
        "Reloaded equity:",
        f"{reloaded.equity():,.2f}",
    )

    print(
        "\nPortfolio-state persistence succeeded."
    )


if __name__ == "__main__":
    main()