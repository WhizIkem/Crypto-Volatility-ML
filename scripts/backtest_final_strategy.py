from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.portfolio.allocation import (
    build_portfolio_allocation,
)
from crypto_volatility.portfolio.backtest import (
    build_equity_curve,
    calculate_max_drawdown,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
)
from crypto_volatility.utils.config import load_config


ANNUALIZATION_FACTOR = 365

# Portfolio settings frozen before the execution-rule experiment.
TARGET_ANNUAL_VOLATILITY = 0.20
TARGET_DAILY_VOLATILITY = (
    TARGET_ANNUAL_VOLATILITY
    / np.sqrt(ANNUALIZATION_FACTOR)
)

MAX_ASSET_WEIGHT = 0.40
CORRELATION_LOOKBACK = 30
MAX_RISKY_EXPOSURE = 1.0
TRANSACTION_COST_BPS = 10.0
CASH_RETURN = 0.0

# Selected using development-period OOS forecasts only.
TURNOVER_THRESHOLD = 0.05

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]


def load_final_predictions() -> pd.DataFrame:
    """
    Load frozen champion-model forecasts from the final test.

    Expected columns:

        timestamp
        BTC_forecast
        ETH_forecast
        SOL_forecast
        BNB_forecast
    """
    path = Path(
        "reports/metrics/final_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Final prediction file not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "timestamp",
        ],
    )

    required_columns = {
        "timestamp",
        "BTC_forecast",
        "ETH_forecast",
        "SOL_forecast",
        "BNB_forecast",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Final prediction file missing columns: "
            f"{sorted(missing)}"
        )

    if df[
        "timestamp"
    ].duplicated().any():
        raise ValueError(
            "Final prediction file contains duplicate timestamps"
        )

    forecast_columns = [
        f"{asset}_forecast"
        for asset in ASSETS
    ]

    if df[
        forecast_columns
    ].isna().any().any():
        raise ValueError(
            "Final prediction file contains missing forecasts"
        )

    return (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


def load_market_returns(
    config: dict,
) -> pd.DataFrame:
    """
    Load aligned historical log returns for all assets.

    Returns through day t are used for correlation estimation.

    The return on day t+1 is used for realized portfolio P&L.
    """
    data_config = config[
        "data"
    ]

    processed_dir = Path(
        data_config[
            "processed_dir"
        ]
    )

    interval = data_config[
        "interval"
    ]

    frames = []

    for asset, symbol in (
        data_config[
            "symbols"
        ].items()
    ):
        path = processed_dir / (
            f"{symbol.lower()}_{interval}_model.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        df = pd.read_parquet(
            path
        )

        required_columns = {
            "timestamp",
            "log_return",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{asset} missing return columns: "
                f"{sorted(missing)}"
            )

        frame = (
            df[
                [
                    "timestamp",
                    "log_return",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_return": asset,
                }
            )
        )

        frames.append(
            frame
        )

    returns = frames[
        0
    ]

    for frame in frames[
        1:
    ]:
        returns = returns.merge(
            frame,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

    return (
        returns
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


def log_returns_to_simple_returns(
    log_returns: pd.Series,
) -> pd.Series:
    """
    Convert continuously compounded returns to simple returns.
    """
    values = log_returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "log_returns must contain only finite values"
        )

    return pd.Series(
        np.expm1(
            values
        ),
        index=log_returns.index,
        dtype=float,
    )


def drift_portfolio_weights(
    full_weights: pd.Series,
    asset_returns: pd.Series,
    cash_return: float = 0.0,
) -> pd.Series:
    """
    Update portfolio weights after realized returns.

    When no rebalance occurs, asset weights naturally drift
    according to each asset's realized return.

    Parameters
    ----------
    full_weights:
        Current BTC/ETH/SOL/BNB/CASH weights before returns.

    asset_returns:
        Realized simple returns for BTC/ETH/SOL/BNB.

    cash_return:
        Realized cash return for the period.

    Returns
    -------
    pd.Series
        End-of-period portfolio weights.
    """
    risky_weights = (
        full_weights[
            ASSETS
        ]
        .astype(
            float
        )
    )

    cash_weight = float(
        full_weights[
            "CASH"
        ]
    )

    if not risky_weights.index.equals(
        asset_returns.index
    ):
        raise ValueError(
            "Risky weight and return indices must match"
        )

    if not np.isfinite(
        risky_weights.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Portfolio weights must be finite"
        )

    if not np.isfinite(
        asset_returns.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Asset returns must be finite"
        )

    risky_end_values = (
        risky_weights
        * (
            1.0
            + asset_returns
        )
    )

    cash_end_value = (
        cash_weight
        * (
            1.0
            + cash_return
        )
    )

    total_end_value = float(
        risky_end_values.sum()
        + cash_end_value
    )

    if total_end_value <= 0:
        raise ValueError(
            "Portfolio value must remain positive"
        )

    drifted = pd.Series(
        {
            **{
                asset: float(
                    risky_end_values[
                        asset
                    ]
                    / total_end_value
                )
                for asset in ASSETS
            },
            "CASH": float(
                cash_end_value
                / total_end_value
            ),
        },
        dtype=float,
    )

    if not np.isclose(
        drifted.sum(),
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Drifted weights do not sum to one"
        )

    return drifted


def calculate_performance_metrics(
    daily: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate portfolio performance metrics.
    """
    returns = daily[
        "net_return"
    ]

    equity = daily[
        "equity"
    ]

    n_days = len(
        returns
    )

    if n_days < 2:
        raise ValueError(
            "Backtest requires at least two observations"
        )

    total_return = float(
        equity.iloc[
            -1
        ]
        - 1.0
    )

    annualized_return = float(
        equity.iloc[
            -1
        ]
        ** (
            ANNUALIZATION_FACTOR
            / n_days
        )
        - 1.0
    )

    daily_volatility = float(
        returns.std(
            ddof=1
        )
    )

    annualized_volatility = float(
        daily_volatility
        * np.sqrt(
            ANNUALIZATION_FACTOR
        )
    )

    mean_daily_return = float(
        returns.mean()
    )

    if daily_volatility > 0:
        sharpe_ratio = float(
            mean_daily_return
            / daily_volatility
            * np.sqrt(
                ANNUALIZATION_FACTOR
            )
        )
    else:
        sharpe_ratio = float(
            "nan"
        )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": calculate_max_drawdown(
            equity
        ),
        "average_daily_turnover": float(
            daily[
                "turnover"
            ].mean()
        ),
        "total_turnover": float(
            daily[
                "turnover"
            ].sum()
        ),
        "total_transaction_cost": float(
            daily[
                "transaction_cost"
            ].sum()
        ),
        "average_risky_exposure": float(
            daily[
                "risky_exposure"
            ].mean()
        ),
        "average_cash_weight": float(
            daily[
                "cash_weight"
            ].mean()
        ),
        "rebalance_count": int(
            daily[
                "rebalanced"
            ].sum()
        ),
    }


def run_backtest(
    forecasts: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    """
    Run the final-test portfolio strategy using the execution rule
    selected only from development-period OOS forecasts.

    Timing
    ------
    At the end of day t:

        1. champion models provide volatility forecasts for t+1;
        2. return information through t estimates correlation;
        3. desired portfolio weights are calculated;
        4. trade only if one-way turnover is at least 5%.

    During day t+1:

        5. executed weights earn t+1 returns;
        6. transaction costs are deducted;
        7. weights drift naturally according to realized returns.

    No final-test results are used to choose the 5% threshold.
    """
    return_lookup = (
        market_returns
        .set_index(
            "timestamp"
        )
    )

    market_timestamps = (
        market_returns[
            "timestamp"
        ]
        .tolist()
    )

    timestamp_to_position = {
        timestamp: position
        for position, timestamp in enumerate(
            market_timestamps
        )
    }

    # Start entirely in cash.
    current_full_weights = pd.Series(
        {
            **{
                asset: 0.0
                for asset in ASSETS
            },
            "CASH": 1.0,
        },
        dtype=float,
    )

    daily_rows = []
    weight_rows = []

    for _, forecast_row in (
        forecasts.iterrows()
    ):
        forecast_date = forecast_row[
            "timestamp"
        ]

        if (
            forecast_date
            not in timestamp_to_position
        ):
            raise ValueError(
                "Forecast timestamp missing from market returns: "
                f"{forecast_date}"
            )

        position = (
            timestamp_to_position[
                forecast_date
            ]
        )

        next_position = (
            position
            + 1
        )

        if next_position >= len(
            market_returns
        ):
            continue

        next_date = market_returns[
            "timestamp"
        ].iloc[
            next_position
        ]

        historical_returns = (
            market_returns
            .iloc[
                : position + 1
            ][
                ASSETS
            ]
            .copy()
        )

        if len(
            historical_returns
        ) < CORRELATION_LOOKBACK:
            continue

        log_volatility_forecasts = pd.Series(
            {
                asset: float(
                    forecast_row[
                        f"{asset}_forecast"
                    ]
                )
                for asset in ASSETS
            },
            dtype=float,
        )

        proposed_allocation = (
            build_portfolio_allocation(
                log_volatility_forecasts=(
                    log_volatility_forecasts
                ),
                historical_returns=(
                    historical_returns
                ),
                max_asset_weight=(
                    MAX_ASSET_WEIGHT
                ),
                target_portfolio_volatility=(
                    TARGET_DAILY_VOLATILITY
                ),
                correlation_lookback=(
                    CORRELATION_LOOKBACK
                ),
                max_risky_exposure=(
                    MAX_RISKY_EXPOSURE
                ),
            )
        )

        proposed_full_weights = pd.Series(
            {
                **(
                    proposed_allocation
                    .risky_weights
                    .to_dict()
                ),
                "CASH": (
                    proposed_allocation
                    .cash_weight
                ),
            },
            dtype=float,
        )

        proposed_full_weights = (
            proposed_full_weights[
                current_full_weights.index
            ]
        )

        proposed_turnover = (
            calculate_turnover(
                current_weights=(
                    proposed_full_weights
                ),
                previous_weights=(
                    current_full_weights
                ),
            )
        )

        rebalance = (
            proposed_turnover
            >= TURNOVER_THRESHOLD
        )

        if rebalance:
            executed_full_weights = (
                proposed_full_weights
                .copy()
            )

            turnover = (
                proposed_turnover
            )
        else:
            executed_full_weights = (
                current_full_weights
                .copy()
            )

            turnover = 0.0

        transaction_cost = (
            calculate_transaction_cost(
                turnover=turnover,
                transaction_cost_bps=(
                    TRANSACTION_COST_BPS
                ),
            )
        )

        next_log_returns = (
            return_lookup.loc[
                next_date,
                ASSETS,
            ]
        )

        next_simple_returns = (
            log_returns_to_simple_returns(
                next_log_returns
            )
        )

        executed_risky_weights = (
            executed_full_weights[
                ASSETS
            ]
        )

        executed_cash_weight = float(
            executed_full_weights[
                "CASH"
            ]
        )

        gross_return = (
            calculate_portfolio_return(
                weights=(
                    executed_risky_weights
                ),
                asset_returns=(
                    next_simple_returns
                ),
                cash_weight=(
                    executed_cash_weight
                ),
                cash_return=CASH_RETURN,
            )
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        daily_rows.append(
            {
                "forecast_date": (
                    forecast_date
                ),
                "return_date": (
                    next_date
                ),
                "gross_return": (
                    gross_return
                ),
                "turnover": (
                    turnover
                ),
                "transaction_cost": (
                    transaction_cost
                ),
                "net_return": (
                    net_return
                ),
                "rebalanced": bool(
                    rebalance
                ),
                "proposed_turnover": (
                    proposed_turnover
                ),
                "forecast_portfolio_volatility": (
                    proposed_allocation
                    .forecast_portfolio_volatility
                ),
                "risky_exposure": float(
                    executed_risky_weights.sum()
                ),
                "cash_weight": (
                    executed_cash_weight
                ),
            }
        )

        weight_rows.append(
            {
                "forecast_date": (
                    forecast_date
                ),
                "return_date": (
                    next_date
                ),
                "rebalanced": bool(
                    rebalance
                ),
                **{
                    f"{asset}_weight": float(
                        executed_risky_weights[
                            asset
                        ]
                    )
                    for asset in ASSETS
                },
                "cash_weight": (
                    executed_cash_weight
                ),
            }
        )

        # After returns are realized, weights drift naturally.
        current_full_weights = (
            drift_portfolio_weights(
                full_weights=(
                    executed_full_weights
                ),
                asset_returns=(
                    next_simple_returns
                ),
                cash_return=(
                    CASH_RETURN
                ),
            )
        )

    if not daily_rows:
        raise ValueError(
            "Backtest produced no observations"
        )

    daily = pd.DataFrame(
        daily_rows
    )

    weights = pd.DataFrame(
        weight_rows
    )

    daily[
        "equity"
    ] = build_equity_curve(
        daily[
            "net_return"
        ]
    ).to_numpy()

    metrics = (
        calculate_performance_metrics(
            daily
        )
    )

    return (
        daily,
        weights,
        metrics,
    )


def main() -> None:
    config = load_config()

    forecasts = (
        load_final_predictions()
    )

    market_returns = (
        load_market_returns(
            config
        )
    )

    (
        daily,
        weights,
        metrics,
    ) = run_backtest(
        forecasts=forecasts,
        market_returns=market_returns,
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = (
        output_dir
        / "champion_strategy_daily_5pct_threshold.csv"
    )

    weights_path = (
        output_dir
        / "champion_strategy_weights_5pct_threshold.csv"
    )

    metrics_path = (
        output_dir
        / "champion_strategy_metrics_5pct_threshold.csv"
    )

    daily.to_csv(
        daily_path,
        index=False,
    )

    weights.to_csv(
        weights_path,
        index=False,
    )

    pd.DataFrame(
        [
            metrics
        ]
    ).to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\nDevelopment-selected champion strategy backtest"
    )

    print(
        "Annual volatility target:",
        f"{TARGET_ANNUAL_VOLATILITY:.0%}",
    )

    print(
        "Daily volatility target:",
        f"{TARGET_DAILY_VOLATILITY:.6f}",
    )

    print(
        "Max asset weight:",
        f"{MAX_ASSET_WEIGHT:.0%}",
    )

    print(
        "Correlation lookback:",
        CORRELATION_LOOKBACK,
    )

    print(
        "Transaction cost:",
        f"{TRANSACTION_COST_BPS:.1f} bps",
    )

    print(
        "Turnover threshold:",
        f"{TURNOVER_THRESHOLD:.0%}",
    )

    print(
        "Backtest observations:",
        len(
            daily
        ),
    )

    print(
        "Rebalances:",
        int(
            daily[
                "rebalanced"
            ].sum()
        ),
    )

    print(
        "\nPerformance metrics:\n"
    )

    for name, value in (
        metrics.items()
    ):
        if isinstance(
            value,
            int,
        ):
            print(
                f"{name}: {value}"
            )
        else:
            print(
                f"{name}: {value:.6f}"
            )

    print(
        "\nSaved daily results to:",
        daily_path,
    )

    print(
        "Saved weights to:",
        weights_path,
    )

    print(
        "Saved metrics to:",
        metrics_path,
    )


if __name__ == "__main__":
    main()