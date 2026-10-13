from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(
    df: pd.DataFrame,
    price_column: str = "close",
    output_column: str = "log_return",
) -> pd.DataFrame:
    """
    Add close-to-close logarithmic returns to market data.

    Parameters
    ----------
    df:
        Input market-data DataFrame.

    price_column:
        Column containing the price series.

    output_column:
        Name of the generated log-return column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with log returns added.

    Raises
    ------
    ValueError
        If the price column is missing or contains non-positive values.
    """
    if price_column not in df.columns:
        raise ValueError(
            f"Missing price column: {price_column}"
        )

    if (df[price_column] <= 0).any():
        raise ValueError(
            f"{price_column} must contain only positive values"
        )

    result = df.copy()

    result[output_column] = np.log(
        result[price_column]
        / result[price_column].shift(1)
    )

    return result


def add_rolling_volatility(
    df: pd.DataFrame,
    return_column: str = "log_return",
    window: int = 30,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """
    Add rolling and annualized rolling volatility.

    Parameters
    ----------
    df:
        Input DataFrame containing log returns.

    return_column:
        Column containing the return series.

    window:
        Rolling-window length in observations.

    annualization_factor:
        Number of trading periods used for annualization.

        Cryptocurrency markets operate every day, so daily data
        use 365 periods per year.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with rolling volatility columns.

    Raises
    ------
    ValueError
        If the return column is missing or the window is invalid.
    """
    if return_column not in df.columns:
        raise ValueError(
            f"Missing return column: {return_column}"
        )

    if window <= 1:
        raise ValueError(
            "window must be greater than 1"
        )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be positive"
        )

    result = df.copy()

    rolling_column = f"rolling_vol_{window}d"
    annualized_column = (
        f"rolling_vol_{window}d_annualized"
    )

    result[rolling_column] = (
        result[return_column]
        .rolling(window)
        .std()
    )

    result[annualized_column] = (
        result[rolling_column]
        * np.sqrt(annualization_factor)
    )

    return result


def add_garman_klass_volatility(
    df: pd.DataFrame,
    annualization_factor: int = 365,
) -> pd.DataFrame:
    """
    Calculate Garman-Klass variance and volatility.

    The estimator uses daily open, high, low, and close prices.

    Parameters
    ----------
    df:
        Input OHLC market-data DataFrame.

    annualization_factor:
        Number of trading periods used for annualization.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame containing:

        - gk_variance
        - gk_volatility
        - gk_volatility_annualized

    Raises
    ------
    ValueError
        If required OHLC columns are missing, contain non-positive
        values, or the annualization factor is invalid.
    """
    required_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required OHLC columns: "
            f"{sorted(missing_columns)}"
        )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be positive"
        )

    if (df[list(required_columns)] <= 0).any().any():
        raise ValueError(
            "OHLC prices must contain only positive values"
        )

    result = df.copy()

    log_hl = np.log(
        result["high"] / result["low"]
    )

    log_co = np.log(
        result["close"] / result["open"]
    )

    result["gk_variance"] = (
        0.5 * log_hl.pow(2)
        - (2 * np.log(2) - 1)
        * log_co.pow(2)
    )

    result["gk_volatility"] = np.sqrt(
        result["gk_variance"].clip(lower=0)
    )

    result["gk_volatility_annualized"] = (
        result["gk_volatility"]
        * np.sqrt(annualization_factor)
    )

    return result


def add_log_garman_klass_volatility(
    df: pd.DataFrame,
    volatility_column: str = "gk_volatility",
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Add log-transformed Garman-Klass volatility.

    Parameters
    ----------
    df:
        Input DataFrame containing Garman-Klass volatility.

    volatility_column:
        Column containing non-negative volatility values.

    epsilon:
        Small positive floor applied before taking logarithms.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame containing log GK volatility.

    Raises
    ------
    ValueError
        If the volatility column is missing, contains negative
        values, or epsilon is not positive.
    """
    if volatility_column not in df.columns:
        raise ValueError(
            f"Missing volatility column: {volatility_column}"
        )

    if (df[volatility_column] < 0).any():
        raise ValueError(
            f"{volatility_column} cannot contain negative values"
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive"
        )

    result = df.copy()

    result["log_gk_volatility"] = np.log(
        result[volatility_column].clip(
            lower=epsilon
        )
    )

    return result