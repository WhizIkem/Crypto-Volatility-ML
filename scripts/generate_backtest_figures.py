from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_volatility.portfolio.backtest import (
    build_equity_curve,
    calculate_drawdown_series,
)
from crypto_volatility.utils.config import load_config

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
]

TRANSACTION_COST_BPS = 10.0


def load_champion() -> pd.DataFrame:
    path = Path(
        "reports/metrics/"
        "champion_strategy_daily_5pct_threshold.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Champion results not found: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=[
            "forecast_date",
            "return_date",
        ],
    )


def load_weights() -> pd.DataFrame:
    path = Path(
        "reports/metrics/"
        "champion_strategy_weights_5pct_threshold.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Champion weights not found: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=[
            "forecast_date",
            "return_date",
        ],
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


def build_benchmark_returns(
    champion: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    dates = champion[
        [
            "return_date",
        ]
    ]

    returns = dates.merge(
        market_returns,
        on="return_date",
        how="inner",
        validate="one_to_one",
    )

    if len(
        returns
    ) != len(
        champion
    ):
        raise ValueError(
            "Benchmark and champion dates do not match"
        )

    cost_rate = (
        TRANSACTION_COST_BPS
        / 10_000.0
    )

    btc_returns = (
        returns[
            "BTC"
        ]
        .copy()
    )

    btc_returns.iloc[
        0
    ] -= cost_rate

    equal_weight_values = pd.Series(
        0.25,
        index=ASSETS,
        dtype=float,
    )

    equal_weight_bh_returns = []

    for row_number, row in (
        returns.iterrows()
    ):
        weights = (
            equal_weight_values
            / equal_weight_values.sum()
        )

        asset_returns = (
            row[
                ASSETS
            ]
            .astype(
                float
            )
        )

        portfolio_return = float(
            weights
            @ asset_returns
        )

        if row_number == 0:
            portfolio_return -= (
                cost_rate
            )

        equal_weight_bh_returns.append(
            portfolio_return
        )

        equal_weight_values = (
            equal_weight_values
            * (
                1.0
                + asset_returns
            )
        )

    return pd.DataFrame(
        {
            "return_date": returns[
                "return_date"
            ],
            "Champion": champion[
                "net_return"
            ].to_numpy(),
            "BTC Buy & Hold": (
                btc_returns.to_numpy()
            ),
            "Equal Weight Buy & Hold": (
                np.asarray(
                    equal_weight_bh_returns
                )
            ),
        }
    )


def save_equity_curve_figure(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    for strategy in [
        "Champion",
        "BTC Buy & Hold",
        "Equal Weight Buy & Hold",
    ]:
        equity = build_equity_curve(
            pd.Series(
                comparison[
                    strategy
                ].to_numpy(),
                index=comparison[
                    "return_date"
                ],
            )
        )

        axis.plot(
            equity.index,
            equity.values,
            label=strategy,
        )

    axis.axhline(
        1.0,
        linewidth=1.0,
    )

    axis.set_title(
        "Out-of-Sample Equity Curves"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Growth of $1"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "backtest_equity_curves.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_drawdown_figure(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    for strategy in [
        "Champion",
        "BTC Buy & Hold",
        "Equal Weight Buy & Hold",
    ]:
        equity = build_equity_curve(
            pd.Series(
                comparison[
                    strategy
                ].to_numpy(),
                index=comparison[
                    "return_date"
                ],
            )
        )

        drawdown = (
            calculate_drawdown_series(
                equity
            )
        )

        axis.plot(
            drawdown.index,
            drawdown.values,
            label=strategy,
        )

    axis.set_title(
        "Out-of-Sample Drawdowns"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Drawdown"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "backtest_drawdowns.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_exposure_figure(
    champion: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    axis.plot(
        champion[
            "return_date"
        ],
        champion[
            "risky_exposure"
        ],
        label="Crypto exposure",
    )

    axis.plot(
        champion[
            "return_date"
        ],
        champion[
            "cash_weight"
        ],
        label="Cash weight",
    )

    axis.set_title(
        "Dynamic Crypto and Cash Exposure"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Portfolio weight"
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "backtest_exposure.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_turnover_figure(
    champion: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    axis.plot(
        champion[
            "return_date"
        ],
        champion[
            "turnover"
        ],
    )

    axis.axhline(
        0.05,
        linestyle="--",
        label="5% execution threshold",
    )

    axis.set_title(
        "Daily Portfolio Turnover"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "One-way turnover"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "backtest_turnover.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_asset_weight_figure(
    weights: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    for asset in ASSETS:
        axis.plot(
            weights[
                "return_date"
            ],
            weights[
                f"{asset}_weight"
            ],
            label=asset,
        )

    axis.set_title(
        "Champion Strategy Asset Weights"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Portfolio weight"
    )

    axis.set_ylim(
        0.0,
        0.45,
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "backtest_asset_weights.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    config = load_config()

    champion = (
        load_champion()
    )

    weights = (
        load_weights()
    )

    market_returns = (
        load_market_returns(
            config
        )
    )

    comparison = (
        build_benchmark_returns(
            champion,
            market_returns,
        )
    )

    output_dir = Path(
        "reports/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_equity_curve_figure(
        comparison,
        output_dir,
    )

    save_drawdown_figure(
        comparison,
        output_dir,
    )

    save_exposure_figure(
        champion,
        output_dir,
    )

    save_turnover_figure(
        champion,
        output_dir,
    )

    save_asset_weight_figure(
        weights,
        output_dir,
    )

    print(
        "\nBacktest figures generated:"
    )

    for filename in [
        "backtest_equity_curves.png",
        "backtest_drawdowns.png",
        "backtest_exposure.png",
        "backtest_turnover.png",
        "backtest_asset_weights.png",
    ]:
        print(
            " -",
            output_dir
            / filename,
        )


if __name__ == "__main__":
    main()