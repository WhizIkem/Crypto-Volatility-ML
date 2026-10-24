from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.portfolio.backtest import (
    build_equity_curve,
    calculate_calmar_ratio,
    calculate_expected_shortfall,
    calculate_historical_var,
    calculate_hit_rate,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_sortino_ratio,
    calculate_transaction_cost,
    calculate_turnover,
)
from crypto_volatility.utils.config import load_config

ANNUALIZATION_FACTOR = 365
TRANSACTION_COST_BPS = 10.0
VAR_CONFIDENCE_LEVEL = 0.95

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]


def load_champion_strategy() -> pd.DataFrame:
    """
    Load the final development-selected portfolio strategy.

    This is the 5% turnover-threshold strategy.
    """
    path = Path(
        "reports/metrics/"
        "champion_strategy_daily_5pct_threshold.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Champion strategy file not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "forecast_date",
            "return_date",
        ],
    )

    required = {
        "return_date",
        "net_return",
        "turnover",
        "transaction_cost",
        "risky_exposure",
        "cash_weight",
        "equity",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Champion strategy file missing columns: "
            f"{sorted(missing)}"
        )

    return (
        df
        .sort_values(
            "return_date"
        )
        .reset_index(
            drop=True
        )
    )


def load_market_returns(
    config: dict,
) -> pd.DataFrame:
    """
    Load aligned simple returns for all four assets.
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

        required = {
            "timestamp",
            "log_return",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{asset} missing columns: "
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
                    "timestamp": "return_date",
                    "log_return": asset,
                }
            )
        )

        frame[
            asset
        ] = np.expm1(
            frame[
                asset
            ].to_numpy(
                dtype=float
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
            on="return_date",
            how="inner",
            validate="one_to_one",
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


def calculate_full_metrics(
    *,
    returns: pd.Series,
    turnover: pd.Series,
    transaction_cost: pd.Series,
    average_risky_exposure: float | None = None,
    average_cash_weight: float | None = None,
    rebalance_count: int | None = None,
) -> dict[str, float | int]:
    """
    Calculate the full final reporting metric set.
    """
    if len(
        returns
    ) < 2:
        raise ValueError(
            "At least two return observations are required"
        )

    equity = build_equity_curve(
        returns
    )

    n_days = len(
        returns
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

    max_drawdown = (
        calculate_max_drawdown(
            equity
        )
    )

    sortino_ratio = (
        calculate_sortino_ratio(
            returns,
            annualization_factor=(
                ANNUALIZATION_FACTOR
            ),
        )
    )

    calmar_ratio = (
        calculate_calmar_ratio(
            annualized_return=(
                annualized_return
            ),
            max_drawdown=(
                max_drawdown
            ),
        )
    )

    hit_rate = (
        calculate_hit_rate(
            returns
        )
    )

    historical_var_95 = (
        calculate_historical_var(
            returns,
            confidence_level=(
                VAR_CONFIDENCE_LEVEL
            ),
        )
    )

    expected_shortfall_95 = (
        calculate_expected_shortfall(
            returns,
            confidence_level=(
                VAR_CONFIDENCE_LEVEL
            ),
        )
    )

    max_drawdown_duration = (
        calculate_max_drawdown_duration(
            equity
        )
    )

    metrics: dict[
        str,
        float | int,
    ] = {
        "n_observations": int(
            n_days
        ),
        "total_return": (
            total_return
        ),
        "annualized_return": (
            annualized_return
        ),
        "annualized_volatility": (
            annualized_volatility
        ),
        "sharpe_ratio": (
            sharpe_ratio
        ),
        "sortino_ratio": (
            sortino_ratio
        ),
        "calmar_ratio": (
            calmar_ratio
        ),
        "max_drawdown": (
            max_drawdown
        ),
        "max_drawdown_duration_days": int(
            max_drawdown_duration
        ),
        "hit_rate": (
            hit_rate
        ),
        "historical_var_95": (
            historical_var_95
        ),
        "expected_shortfall_95": (
            expected_shortfall_95
        ),
        "average_daily_turnover": float(
            turnover.mean()
        ),
        "total_turnover": float(
            turnover.sum()
        ),
        "total_transaction_cost": float(
            transaction_cost.sum()
        ),
    }

    if (
        average_risky_exposure
        is not None
    ):
        metrics[
            "average_risky_exposure"
        ] = float(
            average_risky_exposure
        )

    if (
        average_cash_weight
        is not None
    ):
        metrics[
            "average_cash_weight"
        ] = float(
            average_cash_weight
        )

    if (
        rebalance_count
        is not None
    ):
        metrics[
            "rebalance_count"
        ] = int(
            rebalance_count
        )

    return metrics


def build_btc_buy_and_hold(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    BTC buy-and-hold benchmark.

    The initial move from cash into BTC incurs transaction cost.
    No later rebalancing occurs.
    """
    turnover = pd.Series(
        0.0,
        index=returns.index,
        dtype=float,
    )

    turnover.iloc[
        0
    ] = 1.0

    transaction_cost = (
        turnover
        * (
            TRANSACTION_COST_BPS
            / 10_000.0
        )
    )

    net_return = (
        returns[
            "BTC"
        ]
        - transaction_cost
    )

    return pd.DataFrame(
        {
            "return_date": (
                returns[
                    "return_date"
                ]
            ),
            "net_return": (
                net_return
            ),
            "turnover": (
                turnover
            ),
            "transaction_cost": (
                transaction_cost
            ),
        }
    )


def build_equal_weight_buy_and_hold(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Start at 25% in each crypto asset and never rebalance.

    Portfolio weights drift naturally with asset performance.
    """
    asset_values = pd.Series(
        0.25,
        index=ASSETS,
        dtype=float,
    )

    rows = []

    for row_number, row in (
        returns.iterrows()
    ):
        total_value = float(
            asset_values.sum()
        )

        current_weights = (
            asset_values
            / total_value
        )

        simple_returns = (
            row[
                ASSETS
            ]
            .astype(
                float
            )
        )

        gross_return = float(
            current_weights
            @ simple_returns
        )

        if row_number == 0:
            turnover = 1.0
        else:
            turnover = 0.0

        transaction_cost = (
            calculate_transaction_cost(
                turnover=turnover,
                transaction_cost_bps=(
                    TRANSACTION_COST_BPS
                ),
            )
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        rows.append(
            {
                "return_date": (
                    row[
                        "return_date"
                    ]
                ),
                "net_return": (
                    net_return
                ),
                "turnover": (
                    turnover
                ),
                "transaction_cost": (
                    transaction_cost
                ),
            }
        )

        asset_values = (
            asset_values
            * (
                1.0
                + simple_returns
            )
        )

    return pd.DataFrame(
        rows
    )


def build_equal_weight_daily_rebalanced(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Maintain a 25% target weight in all four crypto assets.

    Unlike the earlier exploratory benchmark, this version
    correctly calculates turnover caused by daily weight drift.
    """
    target_weights = pd.Series(
        0.25,
        index=ASSETS,
        dtype=float,
    )

    current_weights = pd.Series(
        0.0,
        index=ASSETS,
        dtype=float,
    )

    cash_weight = 1.0

    rows = []

    for _, row in returns.iterrows():
        current_full_weights = pd.Series(
            {
                **current_weights.to_dict(),
                "CASH": cash_weight,
            },
            dtype=float,
        )

        target_full_weights = pd.Series(
            {
                **target_weights.to_dict(),
                "CASH": 0.0,
            },
            dtype=float,
        )

        turnover = (
            calculate_turnover(
                current_weights=(
                    target_full_weights
                ),
                previous_weights=(
                    current_full_weights
                ),
            )
        )

        transaction_cost = (
            calculate_transaction_cost(
                turnover=turnover,
                transaction_cost_bps=(
                    TRANSACTION_COST_BPS
                ),
            )
        )

        simple_returns = (
            row[
                ASSETS
            ]
            .astype(
                float
            )
        )

        gross_return = float(
            target_weights
            @ simple_returns
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        rows.append(
            {
                "return_date": (
                    row[
                        "return_date"
                    ]
                ),
                "net_return": (
                    net_return
                ),
                "turnover": (
                    turnover
                ),
                "transaction_cost": (
                    transaction_cost
                ),
            }
        )

        end_values = (
            target_weights
            * (
                1.0
                + simple_returns
            )
        )

        total_end_value = float(
            end_values.sum()
        )

        current_weights = (
            end_values
            / total_end_value
        )

        cash_weight = 0.0

    return pd.DataFrame(
        rows
    )


def benchmark_metrics(
    name: str,
    benchmark: pd.DataFrame,
) -> dict[str, float | int | str]:
    metrics = (
        calculate_full_metrics(
            returns=(
                benchmark[
                    "net_return"
                ]
            ),
            turnover=(
                benchmark[
                    "turnover"
                ]
            ),
            transaction_cost=(
                benchmark[
                    "transaction_cost"
                ]
            ),
        )
    )

    return {
        "strategy": name,
        **metrics,
    }


def main() -> None:
    config = load_config()

    champion = (
        load_champion_strategy()
    )

    market_returns = (
        load_market_returns(
            config
        )
    )

    test_dates = (
        champion[
            [
                "return_date",
            ]
        ]
        .copy()
    )

    final_returns = (
        test_dates
        .merge(
            market_returns,
            on="return_date",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(
        final_returns
    ) != len(
        champion
    ):
        raise ValueError(
            "Benchmark dates do not match champion strategy dates"
        )

    champion_metrics = (
        calculate_full_metrics(
            returns=(
                champion[
                    "net_return"
                ]
            ),
            turnover=(
                champion[
                    "turnover"
                ]
            ),
            transaction_cost=(
                champion[
                    "transaction_cost"
                ]
            ),
            average_risky_exposure=float(
                champion[
                    "risky_exposure"
                ].mean()
            ),
            average_cash_weight=float(
                champion[
                    "cash_weight"
                ].mean()
            ),
            rebalance_count=int(
                champion[
                    "rebalanced"
                ].sum()
            ),
        )
    )

    btc = (
        build_btc_buy_and_hold(
            final_returns
        )
    )

    equal_weight_bh = (
        build_equal_weight_buy_and_hold(
            final_returns
        )
    )

    equal_weight_daily = (
        build_equal_weight_daily_rebalanced(
            final_returns
        )
    )

    rows = [
        {
            "strategy": (
                "Champion Volatility Strategy"
            ),
            **champion_metrics,
        },
        benchmark_metrics(
            "BTC Buy & Hold",
            btc,
        ),
        benchmark_metrics(
            "Equal Weight Buy & Hold",
            equal_weight_bh,
        ),
        benchmark_metrics(
            "Equal Weight Daily Rebalanced",
            equal_weight_daily,
        ),
    ]

    comparison = pd.DataFrame(
        rows
    )

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        output_dir
        / "final_backtest_report.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    champion_path = (
        output_dir
        / "final_champion_metrics.csv"
    )

    pd.DataFrame(
        [
            {
                "strategy": (
                    "Champion Volatility Strategy"
                ),
                **champion_metrics,
            }
        ]
    ).to_csv(
        champion_path,
        index=False,
    )

    display_columns = [
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration_days",
        "hit_rate",
        "historical_var_95",
        "expected_shortfall_95",
        "average_daily_turnover",
        "total_transaction_cost",
    ]

    print(
        "\nFinal backtest report:\n"
    )

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nChampion strategy additional metrics:\n"
    )

    print(
        "Average risky exposure:",
        f"{champion_metrics['average_risky_exposure']:.6f}",
    )

    print(
        "Average cash weight:",
        f"{champion_metrics['average_cash_weight']:.6f}",
    )

    print(
        "Rebalance count:",
        champion_metrics[
            "rebalance_count"
        ],
    )

    print(
        "\nSaved final report to:",
        comparison_path,
    )

    print(
        "Saved champion metrics to:",
        champion_path,
    )


if __name__ == "__main__":
    main()