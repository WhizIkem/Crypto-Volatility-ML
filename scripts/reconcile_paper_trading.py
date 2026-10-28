from pathlib import Path

from crypto_volatility.trading.portfolio_state import (
    load_portfolio_state,
)
from crypto_volatility.trading.reconciliation import (
    assert_reconciliation_passed,
    load_portfolio_history,
    load_trade_history,
    reconcile_latest_paper_state,
)

STATE_PATH = Path(
    "logs/paper_portfolio_state.json"
)

PORTFOLIO_HISTORY_PATH = Path(
    "logs/paper_portfolio_history.csv"
)

TRADE_HISTORY_PATH = Path(
    "logs/paper_trades.csv"
)


def main() -> None:
    print(
        "\nPaper-trading reconciliation"
    )

    state = load_portfolio_state(
        STATE_PATH
    )

    portfolio_history = (
        load_portfolio_history(
            PORTFOLIO_HISTORY_PATH
        )
    )

    trade_history = (
        load_trade_history(
            TRADE_HISTORY_PATH
        )
    )

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                trade_history
            ),
        )
    )

    print(
        "\nForecast origin:",
        report.forecast_origin,
    )

    print(
        "\nEquity reconciliation:"
    )

    print(
        "Logged equity:",
        f"{report.equity_from_log:,.2f}",
    )

    print(
        "Reconstructed equity:",
        f"{report.equity_reconstructed:,.2f}",
    )

    print(
        "Persisted-state equity:",
        f"{report.state_equity:,.2f}",
    )

    print(
        "\nCash reconciliation:"
    )

    print(
        "Logged cash:",
        f"{report.cash_from_log:,.2f}",
    )

    print(
        "Persisted-state cash:",
        f"{report.state_cash:,.2f}",
    )

    print(
        "\nLatest-cycle transaction cost:"
    )

    print(
        "Portfolio-history cost:",
        f"{report.latest_portfolio_transaction_cost:,.2f}",
    )

    print(
        "Trade-history cost:",
        f"{report.latest_trade_transaction_cost:,.2f}",
    )

    print(
        "\nCumulative transaction cost:"
    )

    print(
        "Portfolio-history total:",
        f"{report.cumulative_portfolio_transaction_cost:,.2f}",
    )

    print(
        "Trade-history total:",
        f"{report.cumulative_trade_transaction_cost:,.2f}",
    )

    print(
        "\nChecks:"
    )

    print(
        "State matches history:",
        report.state_matches_history,
    )

    print(
        "Equity matches:",
        report.equity_matches,
    )

    print(
        "Latest-cycle costs match:",
        report.latest_transaction_cost_matches,
    )

    print(
        "Cumulative costs match:",
        report.cumulative_transaction_cost_matches,
    )

    assert_reconciliation_passed(
        report
    )

    print(
        "\nPaper-trading reconciliation passed."
    )


if __name__ == "__main__":
    main()