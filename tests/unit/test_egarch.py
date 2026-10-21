import numpy as np
import pandas as pd
import pytest

from crypto_volatility.models.egarch import (
    fit_egarch_model,
    forecast_egarch_log_volatility,
    forecast_egarch_variance,
    forecast_egarch_volatility,
)


def make_return_series(
    rows: int = 500,
) -> pd.Series:
    rng = np.random.default_rng(
        42
    )

    returns = rng.normal(
        loc=0.0005,
        scale=0.03,
        size=rows,
    )

    return pd.Series(
        returns,
        name="log_return",
    )


def test_fit_egarch_model():
    returns = make_return_series()

    model = fit_egarch_model(
        returns
    )

    assert model is not None
    assert model.convergence_flag == 0


def test_egarch_variance_forecast_is_positive():
    returns = make_return_series()

    model = fit_egarch_model(
        returns
    )

    variance = forecast_egarch_variance(
        model
    )

    assert np.isfinite(
        variance
    )

    assert variance > 0


def test_egarch_volatility_matches_variance():
    returns = make_return_series()

    model = fit_egarch_model(
        returns
    )

    variance = forecast_egarch_variance(
        model
    )

    volatility = forecast_egarch_volatility(
        model
    )

    assert volatility == pytest.approx(
        np.sqrt(variance)
    )


def test_egarch_log_volatility_matches_volatility():
    returns = make_return_series()

    model = fit_egarch_model(
        returns
    )

    volatility = forecast_egarch_volatility(
        model
    )

    log_volatility = (
        forecast_egarch_log_volatility(
            model
        )
    )

    assert log_volatility == pytest.approx(
        np.log(volatility)
    )


def test_egarch_rejects_short_series():
    returns = make_return_series(
        rows=50
    )

    with pytest.raises(
        ValueError,
        match="at least 100",
    ):
        fit_egarch_model(
            returns
        )


def test_egarch_rejects_infinite_returns():
    returns = make_return_series()

    returns.iloc[10] = np.inf

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        fit_egarch_model(
            returns
        )