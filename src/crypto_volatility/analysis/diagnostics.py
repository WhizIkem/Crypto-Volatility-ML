from __future__ import annotations

import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf_test(
    series: pd.Series,
    regression: str = "c",
) -> dict[str, float | int]:
    """
    Run the Augmented Dickey-Fuller stationarity test.

    Parameters
    ----------
    series:
        Time series to test.

    regression:
        Deterministic terms included in the test.

        Common options are:
        - "c": constant
        - "ct": constant and linear trend

    Returns
    -------
    dict
        ADF statistic, p-value, lag count, and number of observations.

    Raises
    ------
    ValueError
        If the input series contains too few usable observations.
    """
    clean_series = series.dropna()

    if len(clean_series) < 10:
        raise ValueError(
            "ADF test requires at least 10 non-missing observations"
        )

    result = adfuller(
        clean_series,
        regression=regression,
        autolag="AIC",
        result_object=False,
    )

    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "lags_used": int(result[2]),
        "n_observations": int(result[3]),
    }


def run_kpss_test(
    series: pd.Series,
    regression: str = "c",
) -> dict[str, float | int]:
    """
    Run the KPSS stationarity test.

    Parameters
    ----------
    series:
        Time series to test.

    regression:
        Deterministic terms included in the test.

        Common options are:
        - "c": level stationarity
        - "ct": trend stationarity

    Returns
    -------
    dict
        KPSS statistic, p-value, and lag count.

    Raises
    ------
    ValueError
        If the input series contains too few usable observations.
    """
    clean_series = series.dropna()

    if len(clean_series) < 10:
        raise ValueError(
            "KPSS test requires at least 10 non-missing observations"
        )

    statistic, p_value, lags, _ = kpss(
        clean_series,
        regression=regression,
        nlags="auto",
        result_object=False,
    )

    return {
        "kpss_statistic": float(statistic),
        "p_value": float(p_value),
        "lags_used": int(lags),
    }


def run_arch_lm_test(
    series: pd.Series,
    nlags: int = 10,
) -> dict[str, float]:
    """
    Run Engle's ARCH-LM test.

    Parameters
    ----------
    series:
        Return series to test for conditional heteroskedasticity.

    nlags:
        Number of lags used in the ARCH test.

    Returns
    -------
    dict
        LM statistic, LM p-value, F statistic, and F-test p-value.

    Raises
    ------
    ValueError
        If the lag count is invalid or insufficient data are available.
    """
    clean_series = series.dropna()

    if nlags < 1:
        raise ValueError(
            "nlags must be at least 1"
        )

    if len(clean_series) <= nlags + 1:
        raise ValueError(
            "Not enough observations for the requested ARCH lags"
        )

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(
        clean_series,
        nlags=nlags,
        result_object=False,
    )

    return {
        "lm_statistic": float(lm_stat),
        "lm_p_value": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_pvalue),
    }


def classify_volatility_regimes(
    series: pd.Series,
) -> pd.Series:
    """
    Classify volatility into low, medium, and high regimes.

    Regimes are based on empirical terciles:

    - Low: <= 33rd percentile
    - Medium: between 33rd and 67th percentiles
    - High: > 67th percentile

    Notes
    -----
    This full-sample classification is intended for exploratory analysis only.

    Production backtests and live systems must estimate thresholds using
    historical training data only to avoid look-ahead bias.

    Parameters
    ----------
    series:
        Volatility series.

    Returns
    -------
    pd.Series
        Ordered categorical regime labels.
    """
    clean_series = series.dropna()

    if clean_series.empty:
        raise ValueError(
            "Volatility series cannot be empty"
        )

    low_threshold = clean_series.quantile(
        1 / 3
    )

    high_threshold = clean_series.quantile(
        2 / 3
    )

    return pd.cut(
        series,
        bins=[
            float("-inf"),
            low_threshold,
            high_threshold,
            float("inf"),
        ],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        include_lowest=True,
    )


def calculate_regime_persistence(
    regimes: pd.Series,
) -> float:
    """
    Calculate the proportion of consecutive observations
    that remain in the same volatility regime.

    Parameters
    ----------
    regimes:
        Series containing regime labels.

    Returns
    -------
    float
        Fraction of valid regime transitions that remain
        in the same state.
    """
    clean_regimes = regimes.dropna()

    if len(clean_regimes) < 2:
        raise ValueError(
            "At least two regime observations are required"
        )

    previous_regime = (
        clean_regimes.shift(1)
    )

    same_regime = (
        clean_regimes
        == previous_regime
    )

    return float(
        same_regime.iloc[1:].mean()
    )


def calculate_transition_matrix(
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Calculate a row-normalized volatility regime transition matrix.

    Rows represent the previous regime and columns represent
    the current regime.

    Parameters
    ----------
    regimes:
        Series containing regime labels.

    Returns
    -------
    pd.DataFrame
        Row-normalized transition probabilities.
    """
    clean_regimes = regimes.dropna()

    if len(clean_regimes) < 2:
        raise ValueError(
            "At least two regime observations are required"
        )

    return pd.crosstab(
        clean_regimes.shift(1),
        clean_regimes,
        normalize="index",
    )