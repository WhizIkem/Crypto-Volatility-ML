from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ("BTC", "ETH", "SOL", "BNB")

DEFAULT_MAX_SCHEDULER_AGE_HOURS = 36

SUCCESS_PATTERN = re.compile(
    r"\[(?P<timestamp>[^\]]+)\]\s+"
    r"Scheduled paper-trading cycle "
    r"finished successfully"
)


@dataclass(frozen=True)
class RuntimeHealth:
    healthy: bool
    reconciliation_healthy: bool
    files_present: bool
    state_matches_latest_history: bool
    transaction_costs_match: bool

    duplicate_forecast_origins: int

    state_forecast_origin: str | None
    history_forecast_origin: str | None

    last_successful_scheduled_run: str | None
    scheduler_success_found: bool
    scheduler_success_recent: bool

    issues: tuple[str, ...]


def _read_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _state_equity(state: dict) -> float:
    return float(
        state["cash"]
    ) + sum(
        float(
            state["quantities"][asset]
        )
        * float(
            state["last_prices"][asset]
        )
        for asset in ASSETS
    )


def _last_successful_run(
    scheduler_log_path: Path,
) -> datetime | None:
    if not scheduler_log_path.exists():
        return None

    last_timestamp: datetime | None = None

    with scheduler_log_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            match = SUCCESS_PATTERN.search(
                line
            )

            if not match:
                continue

            raw_timestamp = match.group(
                "timestamp"
            )

            try:
                timestamp = (
                    datetime.fromisoformat(
                        raw_timestamp.replace(
                            "Z",
                            "+00:00",
                        )
                    )
                )
            except ValueError:
                continue

            if timestamp.tzinfo is None:
                timestamp = (
                    timestamp.replace(
                        tzinfo=UTC
                    )
                )

            timestamp = (
                timestamp.astimezone(UTC)
            )

            if (
                last_timestamp is None
                or timestamp
                > last_timestamp
            ):
                last_timestamp = timestamp

    return last_timestamp


def check_runtime_health(
    *,
    portfolio_history_path: str | Path,
    trade_history_path: str | Path,
    state_path: str | Path,
    scheduler_log_path: str | Path,
    max_scheduler_age_hours: int = (
        DEFAULT_MAX_SCHEDULER_AGE_HOURS
    ),
    now_utc: datetime | None = None,
) -> RuntimeHealth:
    portfolio_history_path = Path(
        portfolio_history_path
    )

    trade_history_path = Path(
        trade_history_path
    )

    state_path = Path(
        state_path
    )

    scheduler_log_path = Path(
        scheduler_log_path
    )

    required_paths = [
        portfolio_history_path,
        trade_history_path,
        state_path,
        scheduler_log_path,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        return RuntimeHealth(
            healthy=False,
            reconciliation_healthy=False,
            files_present=False,
            state_matches_latest_history=False,
            transaction_costs_match=False,
            duplicate_forecast_origins=0,
            state_forecast_origin=None,
            history_forecast_origin=None,
            last_successful_scheduled_run=None,
            scheduler_success_found=False,
            scheduler_success_recent=False,
            issues=tuple(
                f"Missing runtime file: {path}"
                for path in missing_paths
            ),
        )

    issues: list[str] = []

    history = pd.read_csv(
        portfolio_history_path
    )

    trades = pd.read_csv(
        trade_history_path
    )

    state = _read_json(
        state_path
    )

    if history.empty:
        issues.append(
            "Portfolio history is empty."
        )

        return RuntimeHealth(
            healthy=False,
            reconciliation_healthy=False,
            files_present=True,
            state_matches_latest_history=False,
            transaction_costs_match=False,
            duplicate_forecast_origins=0,
            state_forecast_origin=(
                state.get(
                    "last_forecast_timestamp"
                )
            ),
            history_forecast_origin=None,
            last_successful_scheduled_run=None,
            scheduler_success_found=False,
            scheduler_success_recent=False,
            issues=tuple(issues),
        )

    history["forecast_origin"] = (
        pd.to_datetime(
            history["forecast_origin"],
            utc=True,
        )
    )

    duplicate_forecast_origins = int(
        history[
            "forecast_origin"
        ].duplicated().sum()
    )

    if duplicate_forecast_origins:
        issues.append(
            "Duplicate forecast origins "
            "detected in portfolio history: "
            f"{duplicate_forecast_origins}"
        )

    history = history.sort_values(
        "forecast_origin"
    ).reset_index(drop=True)

    latest_history = (
        history.iloc[-1]
    )

    history_forecast_origin = (
        latest_history[
            "forecast_origin"
        ].isoformat()
    )

    state_forecast_origin = (
        state.get(
            "last_forecast_timestamp"
        )
    )

    state_equity = _state_equity(
        state
    )

    equity_matches = bool(
        np.isclose(
            state_equity,
            float(
                latest_history[
                    "equity_after"
                ]
            ),
            rtol=1e-9,
            atol=1e-6,
        )
    )

    cash_matches = bool(
        np.isclose(
            float(state["cash"]),
            float(
                latest_history[
                    "cash"
                ]
            ),
            rtol=1e-9,
            atol=1e-6,
        )
    )

    timestamp_matches = (
        state_forecast_origin
        == history_forecast_origin
    )

    state_matches_latest_history = (
        equity_matches
        and cash_matches
        and timestamp_matches
    )

    if not equity_matches:
        issues.append(
            "Persisted-state equity does "
            "not match latest "
            "portfolio-history equity."
        )

    if not cash_matches:
        issues.append(
            "Persisted-state cash does "
            "not match latest "
            "portfolio-history cash."
        )

    if not timestamp_matches:
        issues.append(
            "Persisted-state forecast "
            "origin does not match latest "
            "portfolio-history origin."
        )

    portfolio_cost = float(
        history[
            "transaction_cost"
        ].sum()
    )

    if trades.empty:
        trade_cost = 0.0

    else:
        trade_cost = float(
            trades[
                "transaction_cost"
            ].sum()
        )

    transaction_costs_match = bool(
        np.isclose(
            portfolio_cost,
            trade_cost,
            rtol=1e-9,
            atol=1e-6,
        )
    )

    if not transaction_costs_match:
        issues.append(
            "Cumulative transaction costs "
            "do not match between portfolio "
            "and trade histories."
        )

    reconciliation_healthy = (
        state_matches_latest_history
        and transaction_costs_match
        and duplicate_forecast_origins == 0
    )

    last_success = _last_successful_run(
        scheduler_log_path
    )

    scheduler_success_found = (
        last_success is not None
    )

    if now_utc is None:
        now_utc = datetime.now(UTC)

    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(
            tzinfo=UTC
        )

    else:
        now_utc = now_utc.astimezone(
            UTC
        )

    scheduler_success_recent = False

    if last_success is None:
        issues.append(
            "No successful scheduled "
            "runtime entry was found in "
            "the scheduler log."
        )

    else:
        maximum_age = timedelta(
            hours=max_scheduler_age_hours
        )

        scheduler_age = (
            now_utc - last_success
        )

        scheduler_success_recent = (
            timedelta(0)
            <= scheduler_age
            <= maximum_age
        )

        if not scheduler_success_recent:
            issues.append(
                "Last successful scheduled "
                "runtime is stale: "
                f"{last_success.isoformat()}."
            )

    healthy = (
        reconciliation_healthy
        and scheduler_success_found
        and scheduler_success_recent
    )

    return RuntimeHealth(
        healthy=healthy,
        reconciliation_healthy=(
            reconciliation_healthy
        ),
        files_present=True,
        state_matches_latest_history=(
            state_matches_latest_history
        ),
        transaction_costs_match=(
            transaction_costs_match
        ),
        duplicate_forecast_origins=(
            duplicate_forecast_origins
        ),
        state_forecast_origin=(
            state_forecast_origin
        ),
        history_forecast_origin=(
            history_forecast_origin
        ),
        last_successful_scheduled_run=(
            last_success.isoformat()
            if last_success is not None
            else None
        ),
        scheduler_success_found=(
            scheduler_success_found
        ),
        scheduler_success_recent=(
            scheduler_success_recent
        ),
        issues=tuple(issues),
    )