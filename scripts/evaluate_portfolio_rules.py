from __future__ import annotations

from dataclasses import dataclass
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

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]

REBALANCE_INTERVALS = [
    1,
    3,
    7,
]

TURNOVER_THRESHOLDS = [
    0.00,
    0.02,
    0.05,
]


@dataclass(frozen=True)
class PortfolioRule:
    rebalance_every_n_days: int
    turnover_threshold: float

    @property
    def name(self) -> str:
        threshold_pct = (
            100
            * self.turnover_threshold
        )

        return (
            f"rebalance_{self.rebalance_every_n_days}d"
            f"_threshold_{threshold_pct:.0f}pct"
        )


def load_development_predictions() -> pd.DataFrame:
    path = Path(
        "reports/metrics/development_champion_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Development predictions not found: "
            f"{path}"
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
            "Development prediction file is missing: "
            f"{sorted(missing)}"
        )

    if df[
        "timestamp"
    ].duplicated().any():
        raise ValueError(
            "Development predictions contain duplicate timestamps"
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
                f"{asset} dataset missing columns: "
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

    returns = frames[0]

    for frame in frames[1:]:
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


def should_rebalance(
    *,
    observation_number: int,
    rule: PortfolioRule,
    proposed_weights: pd.Series,
    current_weights: pd.Series,
) -> bool:
    calendar_allows_rebalance = (
        observation_number
        % rule.rebalance_every_n_days
        == 0
    )

    if not calendar_allows_rebalance:
        return False

    proposed_turnover = calculate_turnover(
        current_weights=proposed_weights,
        previous_weights=current_weights,
    )

    return (
        proposed_turnover
        >= rule.turnover_threshold
    )


def calculate_metrics(
    daily: pd.DataFrame,
) -> dict[str, float]:
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
            "Not enough observations for performance metrics"
        )

    total_return = float(
        equity.iloc[-1]
        - 1.0
    )

    annualized_return = float(
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

    annualized_volatility = float(
        daily_volatility
        * np.sqrt(
            ANNUALIZATION_FACTOR
        )
    )

    if daily_volatility > 0:
        sharpe_ratio = float(
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


def drift_portfolio_weights(
    full_weights: pd.Series,
    asset_returns: pd.Series,
    cash_return: float = 0.0,
) -> pd.Series:
    """
    Update portfolio weights after one period of realized returns.

    If no rebalance occurs, risky-asset weights naturally drift as
    assets earn different returns.

    Cash is also carried forward using the supplied cash return.
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

    return drifted

def run_rule_backtest(
    forecasts: pd.DataFrame,
    market_returns: pd.DataFrame,
    rule: PortfolioRule,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
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

    rows = []

    for observation_number, forecast_row in (
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
                "Forecast timestamp missing from returns: "
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
                historical_returns=historical_returns,
                max_asset_weight=MAX_ASSET_WEIGHT,
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

        rebalance = should_rebalance(
            observation_number=(
                observation_number
            ),
            rule=rule,
            proposed_weights=(
                proposed_full_weights
            ),
            current_weights=(
                current_full_weights
            ),
        )

        if rebalance:
            executed_weights = (
                proposed_full_weights
                .copy()
            )

            turnover = calculate_turnover(
                current_weights=(
                    executed_weights
                ),
                previous_weights=(
                    current_full_weights
                ),
            )
        else:
            executed_weights = (
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

        risky_weights = (
            executed_weights[
                ASSETS
            ]
        )

        cash_weight = float(
            executed_weights[
                "CASH"
            ]
        )

        gross_return = (
            calculate_portfolio_return(
                weights=risky_weights,
                asset_returns=(
                    next_simple_returns
                ),
                cash_weight=cash_weight,
                cash_return=CASH_RETURN,
            )
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        rows.append(
            {
                "forecast_date": forecast_date,
                "return_date": next_date,
                "net_return": net_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "rebalanced": bool(
                    rebalance
                ),
                "risky_exposure": float(
                    risky_weights.sum()
                ),
                "cash_weight": cash_weight,
            }
        )

        current_full_weights = (
            drift_portfolio_weights(
                full_weights=executed_weights,
                asset_returns=next_simple_returns,
                cash_return=CASH_RETURN,
            )
        )

    if not rows:
        raise ValueError(
            f"No observations generated for {rule.name}"
        )

    daily = pd.DataFrame(
        rows
    )

    daily[
        "equity"
    ] = build_equity_curve(
        daily[
            "net_return"
        ]
    ).to_numpy()

    metrics = calculate_metrics(
        daily
    )

    return (
        daily,
        metrics,
    )


def build_rules() -> list[
    PortfolioRule
]:
    return [
        PortfolioRule(
            rebalance_every_n_days=(
                rebalance_interval
            ),
            turnover_threshold=(
                threshold
            ),
        )
        for rebalance_interval in (
            REBALANCE_INTERVALS
        )
        for threshold in (
            TURNOVER_THRESHOLDS
        )
    ]


def rank_rules(
    results: pd.DataFrame,
) -> pd.DataFrame:
    ranked = results.copy()

    ranked[
        "sharpe_rank"
    ] = ranked[
        "sharpe_ratio"
    ].rank(
        ascending=False,
        method="min",
    )

    ranked[
        "drawdown_rank"
    ] = ranked[
        "max_drawdown"
    ].rank(
        ascending=False,
        method="min",
    )

    ranked[
        "cost_rank"
    ] = ranked[
        "total_transaction_cost"
    ].rank(
        ascending=True,
        method="min",
    )

    ranked[
        "simplicity_rank"
    ] = ranked[
        "rebalance_every_n_days"
    ].rank(
        ascending=False,
        method="min",
    )

    ranked[
        "selection_score"
    ] = (
        ranked[
            "sharpe_rank"
        ]
        + ranked[
            "drawdown_rank"
        ]
        + ranked[
            "cost_rank"
        ]
        + (
            0.25
            * ranked[
                "simplicity_rank"
            ]
        )
    )

    return (
        ranked
        .sort_values(
            [
                "selection_score",
                "sharpe_ratio",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    config = load_config()

    forecasts = (
        load_development_predictions()
    )

    market_returns = (
        load_market_returns(
            config
        )
    )

    rules = build_rules()

    output_dir = Path(
        "reports/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_rows = []

    for rule in rules:
        print(
            "Evaluating:",
            rule.name,
        )

        daily, metrics = (
            run_rule_backtest(
                forecasts=forecasts,
                market_returns=(
                    market_returns
                ),
                rule=rule,
            )
        )

        daily_path = (
            output_dir
            / (
                "portfolio_rule_"
                f"{rule.name}_daily.csv"
            )
        )

        daily.to_csv(
            daily_path,
            index=False,
        )

        metric_rows.append(
            {
                "rule": rule.name,
                "rebalance_every_n_days": (
                    rule
                    .rebalance_every_n_days
                ),
                "turnover_threshold": (
                    rule
                    .turnover_threshold
                ),
                **metrics,
            }
        )

    comparison = pd.DataFrame(
        metric_rows
    )

    ranked = rank_rules(
        comparison
    )

    comparison_path = (
        output_dir
        / "portfolio_rule_comparison.csv"
    )

    ranked.to_csv(
        comparison_path,
        index=False,
    )

    columns = [
        "rule",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_daily_turnover",
        "total_transaction_cost",
        "rebalance_count",
        "selection_score",
    ]

    print(
        "\nDevelopment-period portfolio-rule comparison:\n"
    )

    print(
        ranked[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nBest ranked development rule:"
    )

    print(
        ranked.iloc[
            0
        ][
            "rule"
        ]
    )

    print(
        "\nSaved comparison to:",
        comparison_path,
    )


if __name__ == "__main__":
    main()