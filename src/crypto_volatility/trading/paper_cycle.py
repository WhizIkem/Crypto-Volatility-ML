from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from crypto_volatility.trading.portfolio_state import (
    PortfolioState,
)


@dataclass(frozen=True)
class PaperCycleStatus:
    """
    Processing status for a single forecast origin.
    """

    forecast_origin: pd.Timestamp
    already_processed: bool


def normalize_forecast_origin(
    forecast_origin: pd.Timestamp | str,
) -> pd.Timestamp:
    """
    Normalize a forecast timestamp to timezone-aware UTC.
    """
    timestamp = pd.Timestamp(
        forecast_origin
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp


def check_cycle_status(
    *,
    state: PortfolioState,
    forecast_origin: pd.Timestamp | str,
) -> PaperCycleStatus:
    """
    Check whether a forecast origin has already been processed.
    """
    normalized = (
        normalize_forecast_origin(
            forecast_origin
        )
    )

    last_processed = (
        state.last_forecast_timestamp
    )

    if last_processed is None:
        return PaperCycleStatus(
            forecast_origin=normalized,
            already_processed=False,
        )

    last_timestamp = (
        normalize_forecast_origin(
            last_processed
        )
    )

    return PaperCycleStatus(
        forecast_origin=normalized,
        already_processed=(
            last_timestamp
            == normalized
        ),
    )


def mark_cycle_processed(
    *,
    state: PortfolioState,
    forecast_origin: pd.Timestamp | str,
) -> None:
    """
    Mark a successfully completed forecast cycle as processed.

    This function should only be called after forecasting and any
    required execution have completed successfully.
    """
    normalized = (
        normalize_forecast_origin(
            forecast_origin
        )
    )

    state.last_forecast_timestamp = (
        normalized.isoformat()
    )

    state.updated_at = (
        datetime.now(
            UTC
        ).isoformat()
    )


def ensure_new_cycle(
    *,
    state: PortfolioState,
    forecast_origin: pd.Timestamp | str,
) -> pd.Timestamp:
    """
    Validate that the forecast origin has not already been processed.

    Returns the normalized timestamp when the cycle is new.
    """
    status = check_cycle_status(
        state=state,
        forecast_origin=forecast_origin,
    )

    if status.already_processed:
        raise ValueError(
            "Forecast origin has already been processed: "
            f"{status.forecast_origin.isoformat()}"
        )

    return status.forecast_origin