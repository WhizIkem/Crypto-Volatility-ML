import numpy as np
import pandas as pd
import pytest

from crypto_volatility.analysis.diagnostics import (
    calculate_regime_persistence,
    calculate_transition_matrix,
    classify_volatility_regimes,
    run_adf_test,
    run_arch_lm_test,
    run_kpss_test,
)


def test_run_adf_test_returns_expected_keys():
    rng = np.random.default_rng(42)

    series = pd.Series(
        rng.normal(
            size=200,
        )
    )

    result = run_adf_test(series)

    assert set(result) == {
        "adf_statistic",
        "p_value",
        "lags_used",
        "n_observations",
    }


def test_run_kpss_test_returns_expected_keys():
    rng = np.random.default_rng(42)

    series = pd.Series(
        rng.normal(
            size=200,
        )
    )

    result = run_kpss_test(series)

    assert set(result) == {
        "kpss_statistic",
        "p_value",
        "lags_used",
    }


def test_stationarity_tests_reject_short_series():
    series = pd.Series(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least 10",
    ):
        run_adf_test(series)

    with pytest.raises(
        ValueError,
        match="at least 10",
    ):
        run_kpss_test(series)


def test_run_arch_lm_test_returns_expected_keys():
    rng = np.random.default_rng(42)

    series = pd.Series(
        rng.normal(
            size=200,
        )
    )

    result = run_arch_lm_test(
        series,
        nlags=10,
    )

    assert set(result) == {
        "lm_statistic",
        "lm_p_value",
        "f_statistic",
        "f_p_value",
    }


def test_arch_lm_rejects_invalid_lags():
    series = pd.Series(
        np.arange(
            20,
            dtype=float,
        )
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        run_arch_lm_test(
            series,
            nlags=0,
        )


def test_classify_volatility_regimes():
    series = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    regimes = classify_volatility_regimes(
        series
    )

    assert list(regimes.astype(str)) == [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
        "High",
    ]


def test_calculate_regime_persistence():
    regimes = pd.Series(
        [
            "Low",
            "Low",
            "Medium",
            "Medium",
            "High",
        ]
    )

    persistence = (
        calculate_regime_persistence(
            regimes
        )
    )

    assert persistence == pytest.approx(
        0.5
    )


def test_calculate_transition_matrix():
    regimes = pd.Series(
        [
            "Low",
            "Low",
            "Medium",
            "Medium",
        ]
    )

    matrix = calculate_transition_matrix(
        regimes
    )

    assert matrix.loc[
        "Low",
        "Low",
    ] == pytest.approx(
        0.5
    )

    assert matrix.loc[
        "Low",
        "Medium",
    ] == pytest.approx(
        0.5
    )

    assert matrix.loc[
        "Medium",
        "Medium",
    ] == pytest.approx(
        1.0
    )