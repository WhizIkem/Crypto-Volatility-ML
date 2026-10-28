import pandas as pd
import pytest

from crypto_volatility.trading.paper_cycle import (
    check_cycle_status,
    ensure_new_cycle,
    mark_cycle_processed,
    normalize_forecast_origin,
)
from crypto_volatility.trading.portfolio_state import (
    create_initial_portfolio_state,
)


def test_normalize_naive_timestamp():
    timestamp = (
        normalize_forecast_origin(
            "2026-10-24"
        )
    )

    assert timestamp == pd.Timestamp(
        "2026-10-24T00:00:00+00:00"
    )


def test_normalize_timezone_aware_timestamp():
    timestamp = (
        normalize_forecast_origin(
            "2026-10-24T01:00:00+01:00"
        )
    )

    assert timestamp == pd.Timestamp(
        "2026-10-24T00:00:00+00:00"
    )


def test_new_state_has_unprocessed_cycle():
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    status = check_cycle_status(
        state=state,
        forecast_origin=(
            "2026-10-24T00:00:00+00:00"
        ),
    )

    assert (
        status.already_processed
        is False
    )


def test_mark_cycle_processed():
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    forecast_origin = pd.Timestamp(
        "2026-10-24T00:00:00+00:00"
    )

    mark_cycle_processed(
        state=state,
        forecast_origin=forecast_origin,
    )

    assert (
        state.last_forecast_timestamp
        == "2026-10-24T00:00:00+00:00"
    )


def test_processed_cycle_is_detected():
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    forecast_origin = pd.Timestamp(
        "2026-10-24T00:00:00+00:00"
    )

    mark_cycle_processed(
        state=state,
        forecast_origin=forecast_origin,
    )

    status = check_cycle_status(
        state=state,
        forecast_origin=forecast_origin,
    )

    assert (
        status.already_processed
        is True
    )


def test_ensure_new_cycle_rejects_duplicate():
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    forecast_origin = pd.Timestamp(
        "2026-10-24T00:00:00+00:00"
    )

    mark_cycle_processed(
        state=state,
        forecast_origin=forecast_origin,
    )

    with pytest.raises(
        ValueError,
        match="already been processed",
    ):
        ensure_new_cycle(
            state=state,
            forecast_origin=forecast_origin,
        )


def test_different_future_cycle_is_allowed():
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    mark_cycle_processed(
        state=state,
        forecast_origin=(
            "2026-10-24T00:00:00+00:00"
        ),
    )

    next_origin = ensure_new_cycle(
        state=state,
        forecast_origin=(
            "2026-10-25T00:00:00+00:00"
        ),
    )

    assert next_origin == pd.Timestamp(
        "2026-10-25T00:00:00+00:00"
    )


def test_failed_cycle_does_not_mark_state():
    """
    Merely checking a cycle must never mark it as processed.
    """
    state = (
        create_initial_portfolio_state(
            1_000_000.0
        )
    )

    forecast_origin = (
        ensure_new_cycle(
            state=state,
            forecast_origin=(
                "2026-10-24T00:00:00+00:00"
            ),
        )
    )

    # Simulate a downstream failure before mark_cycle_processed().
    try:
        raise RuntimeError(
            "simulated model/execution failure"
        )
    except RuntimeError:
        pass

    assert (
        state.last_forecast_timestamp
        is None
    )

    status = check_cycle_status(
        state=state,
        forecast_origin=forecast_origin,
    )

    assert (
        status.already_processed
        is False
    )