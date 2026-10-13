from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from crypto_volatility.analysis.diagnostics import (
    calculate_regime_persistence,
    calculate_transition_matrix,
    classify_volatility_regimes,
    run_adf_test,
    run_arch_lm_test,
    run_kpss_test,
)
from crypto_volatility.analysis.volatility import (
    add_garman_klass_volatility,
    add_log_garman_klass_volatility,
    add_log_returns,
    add_rolling_volatility,
)
from crypto_volatility.utils.config import load_config

TRADING_DAYS = 365
ROLLING_VOL_WINDOW = 30


def load_datasets(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Load cleaned interim Binance datasets defined in the project config.
    """
    data_config = config["data"]

    interim_dir = Path(
        data_config["interim_dir"]
    )

    interval = data_config["interval"]
    symbols = data_config["symbols"]

    datasets = {}

    for asset, symbol in symbols.items():
        path = interim_dir / (
            f"{symbol.lower()}_{interval}_clean.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Clean dataset not found: {path}"
            )

        datasets[asset] = pd.read_parquet(
            path
        )

    return datasets


def prepare_datasets(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Apply the finalized EDA calculations to each asset.
    """
    prepared = {}

    for asset, df in datasets.items():
        result = add_log_returns(
            df
        )

        result = add_rolling_volatility(
            result,
            window=ROLLING_VOL_WINDOW,
            annualization_factor=TRADING_DAYS,
        )

        result = add_garman_klass_volatility(
            result,
            annualization_factor=TRADING_DAYS,
        )

        result = add_log_garman_klass_volatility(
            result
        )

        prepared[asset] = result

    return prepared


def build_dataset_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build basic dataset integrity and date-range summary.
    """
    rows = []

    for asset, df in datasets.items():
        rows.append(
            {
                "asset": asset,
                "rows": len(df),
                "start_date": (
                    df["timestamp"].min()
                ),
                "end_date": (
                    df["timestamp"].max()
                ),
                "missing_values": int(
                    df.isna().sum().sum()
                ),
                "duplicate_timestamps": int(
                    df["timestamp"]
                    .duplicated()
                    .sum()
                ),
                "chronologically_sorted": bool(
                    df["timestamp"]
                    .is_monotonic_increasing
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_return_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build descriptive statistics for daily log returns.
    """
    rows = []

    for asset, df in datasets.items():
        returns = (
            df["log_return"]
            .dropna()
        )

        rows.append(
            {
                "asset": asset,
                "count": int(
                    returns.count()
                ),
                "mean": (
                    returns.mean()
                ),
                "std": (
                    returns.std()
                ),
                "min": (
                    returns.min()
                ),
                "median": (
                    returns.median()
                ),
                "max": (
                    returns.max()
                ),
                "skewness": (
                    returns.skew()
                ),
                "kurtosis": (
                    returns.kurt()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_normality_results(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Run Jarque-Bera normality tests on log returns.
    """
    rows = []

    for asset, df in datasets.items():
        returns = (
            df["log_return"]
            .dropna()
        )

        result = stats.jarque_bera(
            returns
        )

        rows.append(
            {
                "asset": asset,
                "jb_statistic": float(
                    result.statistic
                ),
                "p_value": float(
                    result.pvalue
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_arch_results(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Run ARCH-LM tests on daily log returns.
    """
    rows = []

    for asset, df in datasets.items():
        result = run_arch_lm_test(
            df["log_return"],
            nlags=10,
        )

        rows.append(
            {
                "asset": asset,
                **result,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_stationarity_results(
    datasets: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run finalized ADF and KPSS tests for prices, returns,
    raw GK volatility, and log GK volatility.
    """
    adf_rows = []
    kpss_rows = []

    specifications = [
        (
            "close_price",
            "close",
            "ct",
        ),
        (
            "log_return",
            "log_return",
            "c",
        ),
        (
            "gk_volatility_annualized",
            "gk_volatility_annualized",
            "c",
        ),
        (
            "log_gk_volatility",
            "log_gk_volatility",
            "c",
        ),
    ]

    for asset, df in datasets.items():
        for (
            variable,
            column,
            regression,
        ) in specifications:
            adf_result = run_adf_test(
                df[column],
                regression=regression,
            )

            kpss_result = run_kpss_test(
                df[column],
                regression=regression,
            )

            adf_rows.append(
                {
                    "asset": asset,
                    "variable": variable,
                    "regression": regression,
                    **adf_result,
                }
            )

            kpss_rows.append(
                {
                    "asset": asset,
                    "variable": variable,
                    "regression": regression,
                    **kpss_result,
                }
            )

    adf_table = pd.DataFrame(
        adf_rows
    )

    kpss_table = pd.DataFrame(
        kpss_rows
    )

    adf_table[
        "reject_unit_root_5pct"
    ] = (
        adf_table["p_value"]
        < 0.05
    )

    kpss_table[
        "reject_stationarity_5pct"
    ] = (
        kpss_table["p_value"]
        < 0.05
    )

    return (
        adf_table,
        kpss_table,
    )


def build_correlation_tables(
    datasets: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build return and volatility correlation matrices.
    """
    returns = pd.DataFrame(
        {
            asset: (
                df
                .set_index("timestamp")
                ["log_return"]
            )
            for asset, df in datasets.items()
        }
    )

    volatility = pd.DataFrame(
        {
            asset: (
                df
                .set_index("timestamp")
                ["gk_volatility_annualized"]
            )
            for asset, df in datasets.items()
        }
    )

    return (
        returns.corr(),
        volatility.corr(),
    )


def build_regime_results(
    datasets: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """
    Build exploratory volatility regime summaries and
    transition matrices.
    """
    summary_rows = []
    transition_matrices = {}

    for asset, df in datasets.items():
        volatility = (
            df[
                "gk_volatility_annualized"
            ]
        )

        clean_volatility = (
            volatility.dropna()
        )

        low_threshold = (
            clean_volatility.quantile(
                1 / 3
            )
        )

        high_threshold = (
            clean_volatility.quantile(
                2 / 3
            )
        )

        regimes = (
            classify_volatility_regimes(
                volatility
            )
        )

        persistence = (
            calculate_regime_persistence(
                regimes
            )
        )

        transition_matrix = (
            calculate_transition_matrix(
                regimes
            )
        )

        transition_matrices[
            asset
        ] = transition_matrix

        summary_rows.append(
            {
                "asset": asset,
                "low_upper_bound": (
                    low_threshold
                ),
                "high_lower_bound": (
                    high_threshold
                ),
                "same_regime_next_day_rate": (
                    persistence
                ),
            }
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        transition_matrices,
    )


def save_table(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save a DataFrame as CSV.
    """
    dataframe.to_csv(
        path,
        index=False,
    )


def save_normalized_price_figure(
    datasets: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    """
    Save normalized price performance for all assets.
    """
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for asset, df in datasets.items():
        normalized_price = (
            df["close"]
            / df["close"].iloc[0]
            * 100
        )

        ax.plot(
            df["timestamp"],
            normalized_price,
            label=asset,
        )

    ax.set_title(
        "Normalized Cryptocurrency Price Performance"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Indexed Price (Start = 100)"
    )

    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_gk_volatility_figure(
    datasets: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    """
    Save annualized Garman-Klass volatility for all assets.
    """
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    for ax, (asset, df) in zip(
        axes,
        datasets.items(),
        strict=True,
    ):
        ax.plot(
            df["timestamp"],
            df[
                "gk_volatility_annualized"
            ],
        )

        ax.set_title(
            f"{asset} Annualized "
            "Garman-Klass Volatility"
        )

        ax.set_ylabel(
            "Annualized Volatility"
        )

        ax.grid(
            True
        )

    axes[-1].set_xlabel(
        "Date"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    """
    Save a correlation matrix as a heatmap.
    """
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        correlation,
        vmin=-1,
        vmax=1,
    )

    x_labels = list(
        correlation.columns
    )

    y_labels = list(
        correlation.index
    )

    ax.set_xticks(
        range(
            len(x_labels)
        )
    )

    ax.set_xticklabels(
        x_labels
    )

    ax.set_yticks(
        range(
            len(y_labels)
        )
    )

    ax.set_yticklabels(
        y_labels
    )

    for i in range(
        len(correlation.index)
    ):
        for j in range(
            len(correlation.columns)
        ):
            ax.text(
                j,
                i,
                (
                    f"{correlation.iloc[i, j]:.2f}"
                ),
                ha="center",
                va="center",
            )

    ax.set_title(
        title
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_regime_figure(
    datasets: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    """
    Save exploratory BTC volatility regime thresholds.
    """
    btc = datasets["BTC"]

    volatility = (
        btc[
            "gk_volatility_annualized"
        ]
    )

    low_threshold = (
        volatility.quantile(
            1 / 3
        )
    )

    high_threshold = (
        volatility.quantile(
            2 / 3
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        btc["timestamp"],
        volatility,
        label="BTC volatility",
    )

    ax.axhline(
        low_threshold,
        linestyle="--",
        label=(
            "Low/Medium threshold"
        ),
    )

    ax.axhline(
        high_threshold,
        linestyle="--",
        label=(
            "Medium/High threshold"
        ),
    )

    ax.set_title(
        "BTC Garman-Klass Volatility Regimes"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Annualized Volatility"
    )

    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main() -> None:
    """
    Run the reproducible Phase 3 EDA pipeline.
    """
    config = load_config()

    tables_dir = Path(
        "reports/tables"
    )

    figures_dir = Path(
        "reports/figures"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading cleaned Binance datasets..."
    )

    datasets = load_datasets(
        config
    )

    print(
        "Applying finalized EDA calculations..."
    )

    datasets = prepare_datasets(
        datasets
    )

    dataset_summary = (
        build_dataset_summary(
            datasets
        )
    )

    return_summary = (
        build_return_summary(
            datasets
        )
    )

    normality_results = (
        build_normality_results(
            datasets
        )
    )

    arch_results = (
        build_arch_results(
            datasets
        )
    )

    (
        adf_results,
        kpss_results,
    ) = build_stationarity_results(
        datasets
    )

    (
        return_correlation,
        volatility_correlation,
    ) = build_correlation_tables(
        datasets
    )

    (
        regime_summary,
        transition_matrices,
    ) = build_regime_results(
        datasets
    )

    save_table(
        dataset_summary,
        tables_dir
        / "dataset_summary.csv",
    )

    save_table(
        return_summary,
        tables_dir
        / "return_summary.csv",
    )

    save_table(
        normality_results,
        tables_dir
        / "normality_results.csv",
    )

    save_table(
        arch_results,
        tables_dir
        / "arch_results.csv",
    )

    save_table(
        adf_results,
        tables_dir
        / "adf_results.csv",
    )

    save_table(
        kpss_results,
        tables_dir
        / "kpss_results.csv",
    )

    save_table(
        return_correlation.reset_index(
            names="asset"
        ),
        tables_dir
        / "return_correlation.csv",
    )

    save_table(
        volatility_correlation.reset_index(
            names="asset"
        ),
        tables_dir
        / "volatility_correlation.csv",
    )

    save_table(
        regime_summary,
        tables_dir
        / "regime_summary.csv",
    )

    for (
        asset,
        transition_matrix,
    ) in transition_matrices.items():
        save_table(
            transition_matrix.reset_index(),
            tables_dir
            / (
                f"{asset.lower()}"
                "_regime_transition_matrix.csv"
            ),
        )

    print(
        "Saving selected EDA figures..."
    )

    save_normalized_price_figure(
        datasets,
        figures_dir
        / "normalized_price_performance.png",
    )

    save_gk_volatility_figure(
        datasets,
        figures_dir
        / "gk_volatility.png",
    )

    save_correlation_heatmap(
        return_correlation,
        "Daily Log Return Correlation",
        figures_dir
        / "return_correlation.png",
    )

    save_correlation_heatmap(
        volatility_correlation,
        "Garman-Klass Volatility Correlation",
        figures_dir
        / "volatility_correlation.png",
    )

    save_regime_figure(
        datasets,
        figures_dir
        / "btc_volatility_regimes.png",
    )

    print(
        "EDA outputs saved to reports/"
    )


if __name__ == "__main__":
    main()