from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from crypto_volatility.models.production import (
    ASSETS,
    FEATURE_COLUMNS,
    load_production_models,
    predict_all_assets,
)
from crypto_volatility.trading.decision import (
    build_trading_decision,
)
from crypto_volatility.trading.live_data import (
    fetch_recent_closed_klines,
)
from crypto_volatility.trading.live_features import (
    build_live_feature_datasets,
    build_live_return_history,
    build_live_var_history,
    get_latest_valid_feature_history,
    get_latest_valid_feature_row,
)
from crypto_volatility.trading.paper_broker import (
    PaperExecutionResult,
    simulate_paper_rebalance,
)
from crypto_volatility.trading.paper_cycle import (
    check_cycle_status,
    mark_cycle_processed,
)
from crypto_volatility.trading.persistence import (
    PaperRuntimeTransaction,
)
from crypto_volatility.trading.portfolio_state import (
    PortfolioState,
    create_initial_portfolio_state,
    load_portfolio_state,
    save_portfolio_state,
    target_quantities,
    update_market_prices,
)
from crypto_volatility.trading.process_lock import (
    PaperTradingProcessLock,
)
from crypto_volatility.trading.runtime import (
    configure_runtime_logging,
    load_paper_trading_runtime_config,
)
from crypto_volatility.trading.startup_validation import (
    validate_paper_runtime_startup,
)
from crypto_volatility.utils.config import (
    load_config,
)

# ------------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------------

RUNTIME = load_paper_trading_runtime_config()

STATE_PATH = RUNTIME.state_path
PORTFOLIO_HISTORY_PATH = RUNTIME.portfolio_history_path
TRADE_HISTORY_PATH = RUNTIME.trade_history_path

INITIAL_CAPITAL = RUNTIME.initial_capital
TRANSACTION_COST_BPS = RUNTIME.transaction_cost_bps
TURNOVER_THRESHOLD = RUNTIME.turnover_threshold


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOGGER = configure_runtime_logging(
    log_path=RUNTIME.log_path,
)


# ------------------------------------------------------------------
# Crash-recoverable persistence
# ------------------------------------------------------------------

RUNTIME_TRANSACTION = PaperRuntimeTransaction(
    state_path=STATE_PATH,
    portfolio_history_path=PORTFOLIO_HISTORY_PATH,
    trade_history_path=TRADE_HISTORY_PATH,
)


# ------------------------------------------------------------------
# Single-process execution lock
# ------------------------------------------------------------------

PROCESS_LOCK = PaperTradingProcessLock(
    RUNTIME.lock_path
)


def load_or_create_state() -> PortfolioState:
    """
    Load the persisted paper portfolio.

    If no state exists, initialize a new portfolio fully in cash.
    """
    if STATE_PATH.exists():
        print(
            "Loading existing paper portfolio..."
        )

        LOGGER.info(
            "Loading existing paper portfolio"
        )

        return load_portfolio_state(
            STATE_PATH
        )

    print(
        "Creating new paper portfolio..."
    )

    LOGGER.info(
        "Creating new paper portfolio with initial capital %.2f",
        INITIAL_CAPITAL,
    )

    state = create_initial_portfolio_state(
        INITIAL_CAPITAL
    )

    save_portfolio_state(
        state,
        STATE_PATH,
    )

    return state


def fetch_live_candles() -> dict[
    str,
    pd.DataFrame,
]:
    """
    Fetch recent closed Binance candles for all configured assets.
    """
    config = load_config()

    data_config = config[
        "data"
    ]

    interval = data_config[
        "interval"
    ]

    datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
        print(
            f"Fetching {asset} ({symbol})..."
        )

        LOGGER.info(
            "Fetching closed Binance candles for %s (%s)",
            asset,
            symbol,
        )

        datasets[
            asset
        ] = fetch_recent_closed_klines(
            symbol=symbol,
            interval=interval,
            limit=120,
        )

    return datasets


def build_latest_prices(
    live_candles: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.Series:
    """
    Extract latest closed-candle prices.

    All assets must share the same latest closed candle.
    """
    prices: dict[
        str,
        float,
    ] = {}

    timestamps = []

    for asset in ASSETS:
        latest = (
            live_candles[
                asset
            ].iloc[
                -1
            ]
        )

        prices[
            asset
        ] = float(
            latest[
                "close"
            ]
        )

        timestamps.append(
            latest[
                "open_time"
            ]
        )

    if len(
        set(
            timestamps
        )
    ) != 1:
        raise ValueError(
            "Latest execution prices are not timestamp aligned"
        )

    return pd.Series(
        prices,
        dtype=float,
    )


def append_portfolio_history(
    *,
    state: PortfolioState,
    forecast_origin: pd.Timestamp,
    forecasts: pd.Series,
    current_weights: pd.Series,
    target_weights: pd.Series,
    turnover: float,
    should_rebalance: bool,
    equity_before: float,
    execution_result: PaperExecutionResult | None,
) -> None:
    """
    Append one successfully processed daily decision to the
    portfolio journal.
    """
    PORTFOLIO_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    equity_after = (
        state.equity()
    )

    total_transaction_cost = (
        0.0
        if execution_result is None
        else execution_result.total_transaction_cost
    )

    row = {
        "run_timestamp_utc": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "forecast_origin": (
            forecast_origin.isoformat()
        ),
        "equity_before": (
            equity_before
        ),
        "equity_after": (
            equity_after
        ),
        "cash": (
            state.cash
        ),
        "cash_weight": (
            state.cash_weight()
        ),
        "turnover": (
            turnover
        ),
        "should_rebalance": (
            should_rebalance
        ),
        "executed": (
            execution_result
            is not None
        ),
        "transaction_cost": (
            total_transaction_cost
        ),
    }

    for asset in ASSETS:
        row[
            f"{asset}_forecast_log_vol"
        ] = float(
            forecasts[
                asset
            ]
        )

        row[
            f"{asset}_current_weight"
        ] = float(
            current_weights[
                asset
            ]
        )

        row[
            f"{asset}_target_weight"
        ] = float(
            target_weights[
                asset
            ]
        )

        row[
            f"{asset}_quantity"
        ] = float(
            state.quantities[
                asset
            ]
        )

        row[
            f"{asset}_price"
        ] = float(
            state.last_prices[
                asset
            ]
        )

    pd.DataFrame(
        [
            row,
        ]
    ).to_csv(
        PORTFOLIO_HISTORY_PATH,
        mode="a",
        header=(
            not PORTFOLIO_HISTORY_PATH.exists()
        ),
        index=False,
    )


def append_trade_history(
    *,
    forecast_origin: pd.Timestamp,
    execution_result: PaperExecutionResult,
) -> None:
    """
    Append executed paper trades to the trade journal.
    """
    if not execution_result.trades:
        return

    TRADE_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_timestamp = (
        datetime.now(
            UTC
        ).isoformat()
    )

    rows = []

    for trade in execution_result.trades:
        rows.append(
            {
                "run_timestamp_utc": (
                    run_timestamp
                ),
                "forecast_origin": (
                    forecast_origin.isoformat()
                ),
                "asset": (
                    trade.asset
                ),
                "side": (
                    trade.side
                ),
                "quantity": (
                    trade.quantity
                ),
                "price": (
                    trade.price
                ),
                "notional": (
                    trade.notional
                ),
                "transaction_cost": (
                    trade.transaction_cost
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        TRADE_HISTORY_PATH,
        mode="a",
        header=(
            not TRADE_HISTORY_PATH.exists()
        ),
        index=False,
    )


def recover_runtime_transaction() -> None:
    """
    Recover unfinished transactional persistence before startup
    validation or state loading.
    """
    recovered = (
        RUNTIME_TRANSACTION
        .recover_if_needed()
    )

    if not recovered:
        return

    LOGGER.warning(
        "Recovered unfinished paper-trading runtime transaction"
    )

    print(
        "\nRecovered an unfinished "
        "paper-trading transaction."
    )

    print(
        "Runtime files were restored to "
        "their previous consistent state."
    )


def validate_runtime_startup() -> None:
    """
    Validate persisted paper-trading files before proceeding.
    """
    report = (
        validate_paper_runtime_startup(
            state_path=STATE_PATH,
            portfolio_history_path=(
                PORTFOLIO_HISTORY_PATH
            ),
            trade_history_path=(
                TRADE_HISTORY_PATH
            ),
            initial_capital=(
                INITIAL_CAPITAL
            ),
        )
    )

    if report.clean_start:
        LOGGER.info(
            "Paper-trading startup validation passed: clean runtime"
        )

        print(
            "\nStartup integrity check passed."
        )

        print(
            "No existing paper portfolio was found."
        )

        return

    LOGGER.info(
        "Paper-trading startup validation passed | "
        "latest_origin=%s | cumulative_cost=%.2f",
        report.latest_forecast_origin,
        report.cumulative_transaction_cost,
    )

    print(
        "\nStartup integrity check passed."
    )

    if report.latest_forecast_origin is not None:
        print(
            "Latest processed forecast:",
            report.latest_forecast_origin,
        )

        print(
            "Cumulative transaction cost:",
            f"{report.cumulative_transaction_cost:,.2f}",
        )


def main() -> None:
    print(
        "\nDaily crypto-volatility paper trader"
    )

    # --------------------------------------------------------------
    # Crash recovery
    # --------------------------------------------------------------

    recover_runtime_transaction()

    # --------------------------------------------------------------
    # Startup integrity validation
    # --------------------------------------------------------------

    validate_runtime_startup()

    # --------------------------------------------------------------
    # Portfolio state
    # --------------------------------------------------------------

    print(
        "\nLoading paper portfolio..."
    )

    state = load_or_create_state()

    # --------------------------------------------------------------
    # Production models
    # --------------------------------------------------------------

    print(
        "\nLoading production models..."
    )

    bundle = (
        load_production_models()
    )

    print(
        "Production models loaded."
    )

    LOGGER.info(
        "Production models loaded"
    )

    # --------------------------------------------------------------
    # Live Binance data
    # --------------------------------------------------------------

    print(
        "\nFetching latest Binance data..."
    )

    live_candles = (
        fetch_live_candles()
    )

    latest_prices = (
        build_latest_prices(
            live_candles
        )
    )

    update_market_prices(
        state,
        latest_prices,
    )

    # --------------------------------------------------------------
    # Live features
    # --------------------------------------------------------------

    print(
        "\nBuilding live features..."
    )

    feature_datasets = (
        build_live_feature_datasets(
            live_candles
        )
    )

    latest_rows = {
        asset: (
            get_latest_valid_feature_row(
                feature_datasets[
                    asset
                ]
            )
        )
        for asset in ASSETS
    }

    latest_timestamps = {
        asset: row[
            "timestamp"
        ]
        for asset, row in (
            latest_rows.items()
        )
    }

    if len(
        set(
            latest_timestamps.values()
        )
    ) != 1:
        raise ValueError(
            "Latest live feature timestamps are not aligned"
        )

    forecast_origin = pd.Timestamp(
        next(
            iter(
                latest_timestamps.values()
            )
        )
    )

    print(
        "\nForecast origin:",
        forecast_origin,
    )

    LOGGER.info(
        "Forecast origin: %s",
        forecast_origin.isoformat(),
    )

    # --------------------------------------------------------------
    # Duplicate-candle protection
    # --------------------------------------------------------------

    cycle_status = (
        check_cycle_status(
            state=state,
            forecast_origin=(
                forecast_origin
            ),
        )
    )

    if cycle_status.already_processed:
        print(
            "\nThis forecast origin has already been processed."
        )

        print(
            "No duplicate paper trade will be executed."
        )

        print(
            "\nCurrent equity:",
            f"{state.equity():,.2f}",
        )

        LOGGER.info(
            "Forecast origin %s already processed; "
            "duplicate execution prevented",
            forecast_origin.isoformat(),
        )

        return

    # --------------------------------------------------------------
    # Production model inputs
    # --------------------------------------------------------------

    var_history = (
        build_live_var_history(
            feature_datasets,
            rows=30,
        )
    )

    return_history = (
        build_live_return_history(
            feature_datasets,
            rows=30,
        )
    )

    eth_feature_history = (
        get_latest_valid_feature_history(
            feature_datasets[
                "ETH"
            ],
            rows=7,
        )
    )

    bnb_latest_features = (
        latest_rows[
            "BNB"
        ][
            FEATURE_COLUMNS
        ]
        .copy()
    )

    # --------------------------------------------------------------
    # Forecasts
    # --------------------------------------------------------------

    print(
        "\nGenerating volatility forecasts..."
    )

    forecasts = (
        predict_all_assets(
            bundle,
            volatility_history=(
                var_history
            ),
            eth_feature_history=(
                eth_feature_history
            ),
            bnb_latest_features=(
                bnb_latest_features
            ),
        )
    )

    # --------------------------------------------------------------
    # Current portfolio
    # --------------------------------------------------------------

    current_weights = (
        state.risky_weights(
            latest_prices
        )
    )

    equity_before = (
        state.equity(
            latest_prices
        )
    )

    # --------------------------------------------------------------
    # Portfolio decision
    # --------------------------------------------------------------

    decision = (
        build_trading_decision(
            forecasts=forecasts,
            historical_returns=(
                return_history
            ),
            current_weights=(
                current_weights
            ),
            turnover_threshold=(
                TURNOVER_THRESHOLD
            ),
        )
    )

    print(
        "\nPortfolio before decision:"
    )

    print(
        "Equity:",
        f"{equity_before:,.2f}",
    )

    print(
        "Cash:",
        f"{state.cash:,.2f}",
    )

    print(
        "\nCurrent risky weights:"
    )

    print(
        current_weights.to_string()
    )

    print(
        "\nForecast log volatility:"
    )

    print(
        decision
        .forecast_log_volatility
        .to_string()
    )

    print(
        "\nTarget risky weights:"
    )

    print(
        decision
        .target_weights
        .to_string()
    )

    print(
        "\nProposed trade weights:"
    )

    print(
        decision
        .trade_weights
        .to_string()
    )

    print(
        "\nRisk allocation:"
    )

    print(
        "Target risky exposure:",
        f"{decision.risky_exposure:.6f}",
    )

    print(
        "Target cash weight:",
        f"{decision.cash_weight:.6f}",
    )

    print(
        "Forecast portfolio volatility:",
        f"{decision.forecast_portfolio_volatility:.6f}",
    )

    print(
        "\nExecution decision:"
    )

    print(
        "Turnover:",
        f"{decision.turnover:.6f}",
    )

    print(
        "Threshold:",
        f"{TURNOVER_THRESHOLD:.6f}",
    )

    print(
        "Should rebalance:",
        decision.should_rebalance,
    )

    LOGGER.info(
        "Portfolio decision | origin=%s | "
        "equity=%.2f | turnover=%.6f | "
        "threshold=%.6f | rebalance=%s",
        forecast_origin.isoformat(),
        equity_before,
        decision.turnover,
        TURNOVER_THRESHOLD,
        decision.should_rebalance,
    )

    # --------------------------------------------------------------
    # Optional execution
    # --------------------------------------------------------------

    execution_result: (
        PaperExecutionResult
        | None
    ) = None

    if decision.should_rebalance:
        targets = (
            target_quantities(
                portfolio_equity=(
                    equity_before
                ),
                target_weights=(
                    decision.target_weights
                ),
                prices=(
                    latest_prices
                ),
            )
        )

        execution_result = (
            simulate_paper_rebalance(
                state=state,
                target_quantities=(
                    targets
                ),
                prices=(
                    latest_prices
                ),
                transaction_cost_bps=(
                    TRANSACTION_COST_BPS
                ),
                rebalance_timestamp=(
                    forecast_origin.isoformat()
                ),
            )
        )

        print(
            "\nExecuted paper trades:"
        )

        for trade in (
            execution_result.trades
        ):
            print(
                f"{trade.side:4} "
                f"{trade.asset:3} "
                f"qty={trade.quantity:.8f} "
                f"price={trade.price:,.2f} "
                f"notional={trade.notional:,.2f} "
                f"cost={trade.transaction_cost:,.2f}"
            )

            LOGGER.info(
                "Paper trade | origin=%s | "
                "asset=%s | side=%s | "
                "quantity=%.8f | price=%.8f | "
                "notional=%.2f | cost=%.2f",
                forecast_origin.isoformat(),
                trade.asset,
                trade.side,
                trade.quantity,
                trade.price,
                trade.notional,
                trade.transaction_cost,
            )

        print(
            "\nTotal transaction cost:",
            f"{execution_result.total_transaction_cost:,.2f}",
        )

        LOGGER.info(
            "Paper rebalance executed | "
            "origin=%s | trades=%d | cost=%.2f",
            forecast_origin.isoformat(),
            len(
                execution_result.trades
            ),
            execution_result.total_transaction_cost,
        )

    else:
        print(
            "\nNo trades executed."
        )

        print(
            "Portfolio change is below the "
            f"{TURNOVER_THRESHOLD:.0%} turnover threshold."
        )

        LOGGER.info(
            "No rebalance executed | "
            "origin=%s | turnover=%.6f",
            forecast_origin.isoformat(),
            decision.turnover,
        )

    # --------------------------------------------------------------
    # Successful-cycle marker in memory
    # --------------------------------------------------------------

    mark_cycle_processed(
        state=state,
        forecast_origin=(
            forecast_origin
        ),
    )

    # --------------------------------------------------------------
    # Transactional persistence
    # --------------------------------------------------------------

    with RUNTIME_TRANSACTION:
        save_portfolio_state(
            state,
            STATE_PATH,
        )

        append_portfolio_history(
            state=state,
            forecast_origin=(
                forecast_origin
            ),
            forecasts=(
                decision.forecast_log_volatility
            ),
            current_weights=(
                current_weights
            ),
            target_weights=(
                decision.target_weights
            ),
            turnover=(
                decision.turnover
            ),
            should_rebalance=(
                decision.should_rebalance
            ),
            equity_before=(
                equity_before
            ),
            execution_result=(
                execution_result
            ),
        )

        if execution_result is not None:
            append_trade_history(
                forecast_origin=(
                    forecast_origin
                ),
                execution_result=(
                    execution_result
                ),
            )

    LOGGER.info(
        "Paper runtime transaction committed | origin=%s",
        forecast_origin.isoformat(),
    )

    # --------------------------------------------------------------
    # Final reporting
    # --------------------------------------------------------------

    print(
        "\nPortfolio after decision:"
    )

    print(
        "Equity:",
        f"{state.equity():,.2f}",
    )

    print(
        "Cash:",
        f"{state.cash:,.2f}",
    )

    print(
        "\nRisky weights:"
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

    print(
        "\nState saved to:",
        STATE_PATH,
    )

    print(
        "Portfolio history:",
        PORTFOLIO_HISTORY_PATH,
    )

    if execution_result is not None:
        print(
            "Trade history:",
            TRADE_HISTORY_PATH,
        )

    print(
        "\nPaper-trading cycle completed successfully."
    )


if __name__ == "__main__":
    try:
        LOGGER.info(
            "Starting paper-trading cycle"
        )

        with PROCESS_LOCK:
            LOGGER.info(
                "Paper-trading process lock acquired"
            )

            main()

        LOGGER.info(
            "Paper-trading process lock released"
        )

        LOGGER.info(
            "Paper-trading cycle finished successfully"
        )

    except Exception:
        LOGGER.exception(
            "Paper-trading cycle failed"
        )

        raise