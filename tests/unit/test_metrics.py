import numpy as np
import pandas as pd
import pytest

from crypto_volatility.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


def test_mean_absolute_error():
    y_true = pd.Series(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    y_pred = pd.Series(
        [
            1.0,
            3.0,
            2.0,
        ]
    )

    result = mean_absolute_error(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(
        2 / 3
    )


def test_root_mean_squared_error():
    y_true = pd.Series(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    y_pred = pd.Series(
        [
            1.0,
            3.0,
            2.0,
        ]
    )

    expected = np.sqrt(
        2 / 3
    )

    result = root_mean_squared_error(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(
        expected
    )


def test_metrics_ignore_aligned_missing_values():
    y_true = pd.Series(
        [
            1.0,
            np.nan,
            3.0,
        ]
    )

    y_pred = pd.Series(
        [
            1.0,
            2.0,
            2.0,
        ]
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    assert mae == pytest.approx(
        0.5
    )


def test_metrics_reject_empty_alignment():
    y_true = pd.Series(
        [
            np.nan,
        ]
    )

    y_pred = pd.Series(
        [
            np.nan,
        ]
    )

    with pytest.raises(
        ValueError,
        match="No aligned observations",
    ):
        mean_absolute_error(
            y_true,
            y_pred,
        )