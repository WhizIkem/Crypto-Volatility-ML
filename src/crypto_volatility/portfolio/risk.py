from __future__ import annotations

import numpy as np
import pandas as pd


def apply_max_weight_cap(
    weights: pd.Series,
    max_weight: float,
) -> pd.Series:
    """
    Cap individual asset weights and redistribute excess weight
    across assets that remain below the cap.

    Parameters
    ----------
    weights:
        Long-only portfolio weights.

    max_weight:
        Maximum allowed weight for any individual asset.

    Returns
    -------
    pd.Series
        Capped and normalized portfolio weights.

    Notes
    -----
    The portfolio remains fully invested, so weights sum to 1.

    A fully invested solution is possible only when:

        number_of_assets * max_weight >= 1
    """
    if weights.empty:
        raise ValueError(
            "weights cannot be empty"
        )

    if not 0 < max_weight <= 1:
        raise ValueError(
            "max_weight must be in (0, 1]"
        )

    values = weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "weights must contain only finite values"
        )

    if (
        values < 0
    ).any():
        raise ValueError(
            "weights cannot contain negative values"
        )

    total = float(
        weights.sum()
    )

    if total <= 0:
        raise ValueError(
            "weights must have a positive sum"
        )

    n_assets = len(
        weights
    )

    if (
        n_assets
        * max_weight
        < 1.0 - 1e-12
    ):
        raise ValueError(
            "max_weight is too small for a fully invested portfolio"
        )

    capped = (
        weights
        / total
    ).astype(
        float
    )

    tolerance = 1e-12

    while True:
        above_cap = (
            capped
            > max_weight
            + tolerance
        )

        if not above_cap.any():
            break

        excess = float(
            (
                capped[
                    above_cap
                ]
                - max_weight
            ).sum()
        )

        capped.loc[
            above_cap
        ] = max_weight

        below_cap = (
            capped
            < max_weight
            - tolerance
        )

        if not below_cap.any():
            break

        available_capacity = (
            max_weight
            - capped[
                below_cap
            ]
        )

        total_capacity = float(
            available_capacity.sum()
        )

        if total_capacity <= 0:
            break

        redistribution = (
            available_capacity
            / total_capacity
            * excess
        )

        capped.loc[
            below_cap
        ] = (
            capped.loc[
                below_cap
            ]
            + redistribution
        )

    capped = capped.clip(
        lower=0.0,
        upper=max_weight,
    )

    normalization_error = (
        1.0
        - float(
            capped.sum()
        )
    )

    if abs(
        normalization_error
    ) > 1e-10:
        available = (
            capped
            < max_weight
            - tolerance
        )

        if available.any():
            capacity = (
                max_weight
                - capped[
                    available
                ]
            )

            capped.loc[
                available
            ] = (
                capped.loc[
                    available
                ]
                + (
                    capacity
                    / capacity.sum()
                    * normalization_error
                )
            )

    if not np.isclose(
        capped.sum(),
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Could not construct fully invested capped weights"
        )

    if (
        capped
        > max_weight
        + 1e-9
    ).any():
        raise ValueError(
            "Maximum weight constraint was violated"
        )

    return pd.Series(
        capped,
        index=weights.index,
        name="portfolio_weight",
        dtype=float,
    )

def calculate_volatility_target_exposure(
    forecast_portfolio_volatility: float,
    target_volatility: float,
    max_exposure: float = 1.0,
) -> float:
    """
    Calculate total risky-asset exposure from a volatility target.

    Exposure is:

        target_volatility / forecast_portfolio_volatility

    subject to a maximum exposure cap.

    A value below 1 means some capital remains in cash.

    Parameters
    ----------
    forecast_portfolio_volatility:
        Expected portfolio volatility for the next period.

    target_volatility:
        Desired portfolio volatility in the same units.

    max_exposure:
        Maximum allowed total risky-asset exposure.
        Use 1.0 for a long-only, unleveraged portfolio.

    Returns
    -------
    float
        Total exposure allocated to risky assets.
    """
    if not np.isfinite(
        forecast_portfolio_volatility
    ):
        raise ValueError(
            "forecast_portfolio_volatility must be finite"
        )

    if not np.isfinite(
        target_volatility
    ):
        raise ValueError(
            "target_volatility must be finite"
        )

    if not np.isfinite(
        max_exposure
    ):
        raise ValueError(
            "max_exposure must be finite"
        )

    if forecast_portfolio_volatility <= 0:
        raise ValueError(
            "forecast_portfolio_volatility "
            "must be greater than zero"
        )

    if target_volatility <= 0:
        raise ValueError(
            "target_volatility must be greater than zero"
        )

    if not 0 < max_exposure <= 1:
        raise ValueError(
            "max_exposure must be in (0, 1]"
        )

    raw_exposure = (
        target_volatility
        / forecast_portfolio_volatility
    )

    exposure = min(
        raw_exposure,
        max_exposure,
    )

    return float(
        exposure
    )


def apply_total_exposure(
    risky_weights: pd.Series,
    exposure: float,
) -> pd.Series:
    """
    Scale fully invested risky-asset weights by total exposure.

    The remaining capital is implicitly held in cash.

    Example
    -------
    Fully invested weights:

        BTC  0.30
        ETH  0.25
        SOL  0.25
        BNB  0.20

    With exposure = 0.60:

        BTC  0.18
        ETH  0.15
        SOL  0.15
        BNB  0.12

    Total crypto exposure = 0.60
    Cash allocation = 0.40
    """
    if risky_weights.empty:
        raise ValueError(
            "risky_weights cannot be empty"
        )

    values = risky_weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "risky_weights must contain only finite values"
        )

    if (
        values < 0
    ).any():
        raise ValueError(
            "risky_weights cannot contain negative values"
        )

    total = float(
        risky_weights.sum()
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "risky_weights must sum to 1 before exposure scaling"
        )

    if not np.isfinite(
        exposure
    ):
        raise ValueError(
            "exposure must be finite"
        )

    if not 0 <= exposure <= 1:
        raise ValueError(
            "exposure must be in [0, 1]"
        )

    scaled = (
        risky_weights
        * exposure
    )

    return pd.Series(
        scaled,
        index=risky_weights.index,
        name="portfolio_weight",
        dtype=float,
    )


def calculate_cash_weight(
    risky_weights: pd.Series,
) -> float:
    """
    Calculate the cash allocation implied by risky-asset weights.
    """
    if risky_weights.empty:
        raise ValueError(
            "risky_weights cannot be empty"
        )

    values = risky_weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "risky_weights must contain only finite values"
        )

    risky_exposure = float(
        risky_weights.sum()
    )

    if not 0 <= risky_exposure <= 1 + 1e-9:
        raise ValueError(
            "Total risky exposure must be between 0 and 1"
        )

    cash_weight = (
        1.0
        - risky_exposure
    )

    return float(
        max(
            cash_weight,
            0.0,
        )
    )

def build_forecast_covariance(
    forecast_volatility: pd.Series,
    historical_returns: pd.DataFrame,
    lookback: int = 30,
) -> pd.DataFrame:
    """
    Build a forecast covariance matrix.

    Marginal asset volatility comes from the forecasting models.

    Cross-asset dependence is estimated using the correlation matrix
    of recently observed historical returns.

    Parameters
    ----------
    forecast_volatility:
        Positive next-period volatility forecasts indexed by asset.

    historical_returns:
        Historical return observations with one column per asset.

        The caller is responsible for providing only information
        available at the forecast date.

    lookback:
        Number of most recent return observations used to estimate
        correlation.

    Returns
    -------
    pd.DataFrame
        Forecast covariance matrix.
    """
    if forecast_volatility.empty:
        raise ValueError(
            "forecast_volatility cannot be empty"
        )

    if historical_returns.empty:
        raise ValueError(
            "historical_returns cannot be empty"
        )

    if lookback < 2:
        raise ValueError(
            "lookback must be at least 2"
        )

    volatility_values = (
        forecast_volatility
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        volatility_values
    ).all():
        raise ValueError(
            "forecast_volatility must contain only finite values"
        )

    if (
        volatility_values <= 0
    ).any():
        raise ValueError(
            "forecast_volatility must contain only positive values"
        )

    assets = list(
        forecast_volatility.index
    )

    missing_columns = (
        set(
            assets
        )
        - set(
            historical_returns.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing historical return columns: "
            f"{sorted(missing_columns)}"
        )

    recent_returns = (
        historical_returns[
            assets
        ]
        .tail(
            lookback
        )
        .copy()
    )

    if len(
        recent_returns
    ) < 2:
        raise ValueError(
            "Not enough historical returns to estimate correlation"
        )

    if recent_returns.isna().any().any():
        raise ValueError(
            "Historical return window cannot contain missing values"
        )

    return_values = (
        recent_returns
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        return_values
    ).all():
        raise ValueError(
            "Historical returns must contain only finite values"
        )

    correlation = (
        recent_returns
        .corr()
    )

    if correlation.isna().any().any():
        raise ValueError(
            "Could not estimate a valid correlation matrix"
        )

    volatility_matrix = np.diag(
        volatility_values
    )

    covariance_values = (
        volatility_matrix
        @ correlation.to_numpy(
            dtype=float
        )
        @ volatility_matrix
    )

    return pd.DataFrame(
        covariance_values,
        index=assets,
        columns=assets,
        dtype=float,
    )


def calculate_portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Calculate portfolio volatility from weights and covariance.

    Portfolio variance is:

        w' Σ w

    Portfolio volatility is the square root of that variance.
    """
    if weights.empty:
        raise ValueError(
            "weights cannot be empty"
        )

    weight_values = weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        weight_values
    ).all():
        raise ValueError(
            "weights must contain only finite values"
        )

    if (
        weight_values < 0
    ).any():
        raise ValueError(
            "weights cannot contain negative values"
        )

    assets = list(
        weights.index
    )

    missing_rows = (
        set(
            assets
        )
        - set(
            covariance.index
        )
    )

    missing_columns = (
        set(
            assets
        )
        - set(
            covariance.columns
        )
    )

    if (
        missing_rows
        or missing_columns
    ):
        raise ValueError(
            "Covariance matrix does not contain all portfolio assets"
        )

    aligned_covariance = covariance.loc[
        assets,
        assets,
    ]

    covariance_values = (
        aligned_covariance
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        covariance_values
    ).all():
        raise ValueError(
            "covariance must contain only finite values"
        )

    portfolio_variance = float(
        weight_values
        @ covariance_values
        @ weight_values
    )

    if portfolio_variance < -1e-12:
        raise ValueError(
            "Portfolio variance cannot be negative"
        )

    portfolio_variance = max(
        portfolio_variance,
        0.0,
    )

    return float(
        np.sqrt(
            portfolio_variance
        )
    )