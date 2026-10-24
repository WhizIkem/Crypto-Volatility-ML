from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.portfolio.backtest import (
    build_equity_curve,
    calculate_max_drawdown,
    calculate_transaction_cost,
)
from crypto_volatility.utils.config import load_config

ANNUALIZATION_FACTOR = 365
TRANSACTION_COST_BPS = 10.0

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]


def load_test_dates() -> pd.DataFrame:
    """
    Load the exact return dates used by the champion strategy.

    This ensures all benchmark strategies are evaluated over
    exactly the same period.
    """
    path = Path(
        "reports/metrics/champion_strategy_daily.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Champion strategy results not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "return_date",
        ],
    )

    return df[
        [
            "return_date",
        ]
    ].copy()


def load_market_returns(
    config: dict,
) -> pd.DataFrame:
    """
    Load aligned log returns for all assets.
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
                    "timestamp": "return_date",
                    "log_return": asset,
                }
            )
        )

        frames.append(
            frame
        )

    returns = frames[0]

    for frame in frames[1:]:
        returns = returns.merge(
            frame,
            on="return_date",
            how="inner",
            validate="one_to_one",
        )

    for asset in ASSETS:
        returns[
            asset
        ] = np.expm1(
            returns[
                asset
            ]
        )

    return (
        returns
        .sort_values(
            "return_date"
        )
        .reset_index(
            drop=True
        )
    )


def calculate_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    transaction_cost: pd.Series,
) -> dict[str, float]:
    """
    Calculate benchmark performance statistics.
    """
    equity = build_equity_curve(
        returns
    )

    n_days = len(
        returns
    )

    total_return = (
        equity.iloc[-1]
        - 1.0
    )

    annualized_return = (
        equity.iloc[-1]
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

    annualized_volatility = (
        daily_volatility
        * np.sqrt(
            ANNUALIZATION_FACTOR
        )
    )

    if daily_volatility > 0:
        sharpe_ratio = (
            returns.mean()
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
        "total_return": float(
            total_return
        ),
        "annualized_return": float(
            annualized_return
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe_ratio": float(
            sharpe_ratio
        ),
        "max_drawdown": (
            calculate_max_drawdown(
                equity
            )
        ),
        "average_daily_turnover": float(
            turnover.mean()
        ),
        "total_transaction_cost": float(
            transaction_cost.sum()
        ),
    }


def btc_buy_and_hold(
    returns: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    BTC-only buy-and-hold benchmark.

    Initial purchase incurs transaction cost once.
    """
    turnover = pd.Series(
        0.0,
        index=returns.index,
    )

    turnover.iloc[0] = 1.0

    costs = turnover.apply(
        lambda value: calculate_transaction_cost(
            turnover=value,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
    )

    net_returns = (
        returns[
            "BTC"
        ]
        - costs
    )

    return (
        net_returns,
        turnover,
        costs,
    )


def equal_weight_rebalanced(
    returns: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Daily-rebalanced 25% portfolio.

    First allocation starts from cash and therefore incurs
    full initial turnover.
    """
    gross_returns = (
        returns[
            ASSETS
        ]
        .mean(
            axis=1
        )
    )

    turnover = pd.Series(
        0.0,
        index=returns.index,
    )

    turnover.iloc[0] = 1.0

    costs = turnover.apply(
        lambda value: calculate_transaction_cost(
            turnover=value,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
    )

    net_returns = (
        gross_returns
        - costs
    )

    return (
        net_returns,
        turnover,
        costs,
    )


def equal_weight_buy_and_hold(
    returns: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Buy all four assets at 25% each and allow weights to drift.
    """
    asset_values = pd.Series(
        0.25,
        index=ASSETS,
        dtype=float,
    )

    portfolio_returns = []

    turnover_values = []

    transaction_cost_values = []

    for index, row in (
        returns[
            ASSETS
        ]
        .iterrows()
    ):
        if index == returns.index[
            0
        ]:
            turnover = 1.0
        else:
            turnover = 0.0

        cost = calculate_transaction_cost(
            turnover=turnover,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )

        total_value = float(
            asset_values.sum()
        )

        weights = (
            asset_values
            / total_value
        )

        gross_return = float(
            weights
            @ row
        )

        net_return = (
            gross_return
            - cost
        )

        portfolio_returns.append(
            net_return
        )

        turnover_values.append(
            turnover
        )

        transaction_cost_values.append(
            cost
        )

        asset_values = (
            asset_values
            * (
                1.0
                + row
            )
        )

    return (
        pd.Series(
            portfolio_returns,
            index=returns.index,
            dtype=float,
        ),
        pd.Series(
            turnover_values,
            index=returns.index,
            dtype=float,
        ),
        pd.Series(
            transaction_cost_values,
            index=returns.index,
            dtype=float,
        ),
    )


def main() -> None:
    config = load_config()

    test_dates = load_test_dates()

    market_returns = (
        load_market_returns(
            config
        )
    )

    returns = test_dates.merge(
        market_returns,
        on="return_date",
        how="inner",
        validate="one_to_one",
    )

    if len(
        returns
    ) != len(
        test_dates
    ):
        raise ValueError(
            "Benchmark dates do not match champion backtest dates"
        )

    strategies = {
        "BTC Buy & Hold": btc_buy_and_hold,
        "Equal Weight Buy & Hold": (
            equal_weight_buy_and_hold
        ),
        "Equal Weight Daily Rebalanced": (
            equal_weight_rebalanced
        ),
    }

    metric_rows = []

    for strategy_name, strategy_function in (
        strategies.items()
    ):
        (
            net_returns,
            turnover,
            costs,
        ) = strategy_function(
            returns
        )

        metrics = calculate_metrics(
            returns=net_returns,
            turnover=turnover,
            transaction_cost=costs,
        )

        metric_rows.append(
            {
                "strategy": strategy_name,
                **metrics,
            }
        )

    champion_metrics_path = Path(
        "reports/metrics/champion_strategy_metrics.csv"
    )

    if not champion_metrics_path.exists():
        raise FileNotFoundError(
            "Champion strategy metrics not found"
        )

    champion_metrics = (
        pd.read_csv(
            champion_metrics_path
        )
        .iloc[
            0
        ]
        .to_dict()
    )

    metric_rows.append(
        {
            "strategy": "Champion Volatility Strategy",
            **{
                key: float(
                    value
                )
                for key, value in (
                    champion_metrics.items()
                )
            },
        }
    )

    comparison = pd.DataFrame(
        metric_rows
    )

    output_path = Path(
        "reports/metrics/backtest_benchmark_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    display_columns = [
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_daily_turnover",
        "total_transaction_cost",
    ]

    print(
        "\nBacktest benchmark comparison:\n"
    )

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved comparison to:",
        output_path,
    )


if __name__ == "__main__":
    main()