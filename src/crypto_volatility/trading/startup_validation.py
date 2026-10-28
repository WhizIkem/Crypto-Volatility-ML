from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_volatility.trading.paper_cycle import (
    normalize_forecast_origin,
)
from crypto_volatility.trading.portfolio_state import (
    ASSETS,
    load_portfolio_state,
)
from crypto_volatility.trading.reconciliation import (
    assert_reconciliation_passed,
    load_portfolio_history,
    load_trade_history,
    reconcile_latest_paper_state,
)

DEFAULT_TOLERANCE = 1e-6


class StartupIntegrityError(RuntimeError):
    """
    Raised when persisted paper-trading runtime data are unsafe
    or internally inconsistent.
    """


@dataclass(frozen=True)
class StartupValidationReport:
    """
    Result of startup integrity validation.
    """

    clean_start: bool
    state_exists: bool
    portfolio_history_exists: bool
    trade_history_exists: bool
    latest_forecast_origin: str | None
    cumulative_transaction_cost: float


def _validate_portfolio_origins(
    portfolio_history: pd.DataFrame,
) -> pd.Series:
    """
    Validate portfolio-history forecast origins.

    Portfolio history must contain exactly one row per processed
    forecast origin and must be strictly chronological.
    """
    try:
        origins = pd.to_datetime(
            portfolio_history[
                "forecast_origin"
            ],
            utc=True,
            errors="raise",
        )

    except Exception as exc:
        raise StartupIntegrityError(
            "Portfolio history contains invalid forecast timestamps"
        ) from exc

    if origins.duplicated().any():
        duplicates = (
            origins[
                origins.duplicated(
                    keep=False
                )
            ]
            .astype(
                str
            )
            .tolist()
        )

        raise StartupIntegrityError(
            "Portfolio history contains duplicate forecast origins: "
            f"{duplicates}"
        )

    if not origins.is_monotonic_increasing:
        raise StartupIntegrityError(
            "Portfolio history forecast origins are not chronological"
        )

    return origins


def _validate_trade_history(
    *,
    trade_history: pd.DataFrame,
    portfolio_origins: pd.Series,
    tolerance: float,
) -> None:
    """
    Validate individual paper-trade journal rows.
    """
    if trade_history.empty:
        return

    try:
        trade_origins = pd.to_datetime(
            trade_history[
                "forecast_origin"
            ],
            utc=True,
            errors="raise",
        )

    except Exception as exc:
        raise StartupIntegrityError(
            "Trade history contains invalid forecast timestamps"
        ) from exc

    valid_origins = set(
        portfolio_origins.tolist()
    )

    invalid_origins = [
        timestamp
        for timestamp in trade_origins.tolist()
        if timestamp not in valid_origins
    ]

    if invalid_origins:
        raise StartupIntegrityError(
            "Trade history contains forecast origins that are "
            "missing from portfolio history"
        )

    invalid_assets = (
        set(
            trade_history[
                "asset"
            ].astype(
                str
            )
        )
        - set(
            ASSETS
        )
    )

    if invalid_assets:
        raise StartupIntegrityError(
            "Trade history contains unsupported assets: "
            f"{sorted(invalid_assets)}"
        )

    valid_sides = {
        "BUY",
        "SELL",
    }

    invalid_sides = (
        set(
            trade_history[
                "side"
            ].astype(
                str
            )
            .str.upper()
        )
        - valid_sides
    )

    if invalid_sides:
        raise StartupIntegrityError(
            "Trade history contains invalid sides: "
            f"{sorted(invalid_sides)}"
        )

    numeric_columns = [
        "quantity",
        "price",
        "notional",
        "transaction_cost",
    ]

    for column in numeric_columns:
        try:
            values = (
                trade_history[
                    column
                ]
                .astype(
                    float
                )
                .to_numpy()
            )

        except Exception as exc:
            raise StartupIntegrityError(
                f"Trade history column {column} "
                "contains non-numeric values"
            ) from exc

        if not np.isfinite(
            values
        ).all():
            raise StartupIntegrityError(
                f"Trade history column {column} "
                "contains non-finite values"
            )

    quantities = (
        trade_history[
            "quantity"
        ].astype(
            float
        )
    )

    prices = (
        trade_history[
            "price"
        ].astype(
            float
        )
    )

    notionals = (
        trade_history[
            "notional"
        ].astype(
            float
        )
    )

    costs = (
        trade_history[
            "transaction_cost"
        ].astype(
            float
        )
    )

    if (
        quantities
        <= 0
    ).any():
        raise StartupIntegrityError(
            "Trade history contains non-positive quantities"
        )

    if (
        prices
        <= 0
    ).any():
        raise StartupIntegrityError(
            "Trade history contains non-positive prices"
        )

    if (
        notionals
        <= 0
    ).any():
        raise StartupIntegrityError(
            "Trade history contains non-positive notionals"
        )

    if (
        costs
        < 0
    ).any():
        raise StartupIntegrityError(
            "Trade history contains negative transaction costs"
        )

    reconstructed_notional = (
        quantities
        * prices
    )

    if not np.allclose(
        notionals.to_numpy(
            dtype=float
        ),
        reconstructed_notional.to_numpy(
            dtype=float
        ),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise StartupIntegrityError(
            "Trade-history notionals do not match quantity × price"
        )


def _validate_unprocessed_initial_state(
    *,
    state_path: Path,
    trade_history_path: Path,
    initial_capital: float,
    tolerance: float,
) -> StartupValidationReport:
    """
    Validate the special case where state exists but no portfolio
    cycle has been journaled yet.

    This can occur if the paper account was initialized before the
    first successful trading cycle.
    """
    try:
        state = load_portfolio_state(
            state_path
        )

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise StartupIntegrityError(
            "Paper portfolio state is corrupted or invalid"
        ) from exc

    if state.last_forecast_timestamp is not None:
        raise StartupIntegrityError(
            "Portfolio state records a processed forecast but "
            "portfolio history is missing"
        )

    if trade_history_path.exists():
        trade_history = load_trade_history(
            trade_history_path
        )

        if not trade_history.empty:
            raise StartupIntegrityError(
                "Trade history exists without portfolio history"
            )

    if any(
        not np.isclose(
            float(
                state.quantities[
                    asset
                ]
            ),
            0.0,
            atol=tolerance,
            rtol=tolerance,
        )
        for asset in ASSETS
    ):
        raise StartupIntegrityError(
            "Unjournaled portfolio state contains asset positions"
        )

    if not np.isclose(
        float(
            state.cash
        ),
        float(
            initial_capital
        ),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise StartupIntegrityError(
            "Unjournaled portfolio state cash does not match "
            "configured initial capital"
        )

    return StartupValidationReport(
        clean_start=False,
        state_exists=True,
        portfolio_history_exists=False,
        trade_history_exists=(
            trade_history_path.exists()
        ),
        latest_forecast_origin=None,
        cumulative_transaction_cost=0.0,
    )


def validate_paper_runtime_startup(
    *,
    state_path: str | Path,
    portfolio_history_path: str | Path,
    trade_history_path: str | Path,
    initial_capital: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> StartupValidationReport:
    """
    Validate persisted paper-trading runtime data before trading.

    Validation covers:

    - corrupted portfolio-state JSON;
    - impossible persisted portfolio values;
    - missing state/history relationships;
    - duplicate or unordered processed forecast origins;
    - state/latest-history timestamp consistency;
    - invalid trade rows;
    - orphaned trade records;
    - state/history/equity reconciliation;
    - latest and cumulative transaction-cost reconciliation.

    A completely empty runtime directory is considered a valid
    clean start.
    """
    state_source = Path(
        state_path
    )

    portfolio_source = Path(
        portfolio_history_path
    )

    trade_source = Path(
        trade_history_path
    )

    if not np.isfinite(
        initial_capital
    ):
        raise ValueError(
            "initial_capital must be finite"
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive"
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative"
        )

    state_exists = (
        state_source.exists()
    )

    portfolio_history_exists = (
        portfolio_source.exists()
    )

    trade_history_exists = (
        trade_source.exists()
    )

    # --------------------------------------------------------------
    # Completely clean runtime.
    # --------------------------------------------------------------

    if not (
        state_exists
        or portfolio_history_exists
        or trade_history_exists
    ):
        return StartupValidationReport(
            clean_start=True,
            state_exists=False,
            portfolio_history_exists=False,
            trade_history_exists=False,
            latest_forecast_origin=None,
            cumulative_transaction_cost=0.0,
        )

    # --------------------------------------------------------------
    # Journals must never exist without portfolio state.
    # --------------------------------------------------------------

    if not state_exists:
        raise StartupIntegrityError(
            "Paper-trading journals exist but portfolio state is missing"
        )

    # --------------------------------------------------------------
    # State-only initialization is valid only before the first
    # successful processed cycle.
    # --------------------------------------------------------------

    if not portfolio_history_exists:
        return _validate_unprocessed_initial_state(
            state_path=state_source,
            trade_history_path=(
                trade_source
            ),
            initial_capital=(
                initial_capital
            ),
            tolerance=tolerance,
        )

    # --------------------------------------------------------------
    # Load and validate persisted state.
    # --------------------------------------------------------------

    try:
        state = load_portfolio_state(
            state_source
        )

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise StartupIntegrityError(
            "Paper portfolio state is corrupted or invalid"
        ) from exc

    # --------------------------------------------------------------
    # Load journals.
    # --------------------------------------------------------------

    try:
        portfolio_history = (
            load_portfolio_history(
                portfolio_source
            )
        )

    except Exception as exc:
        raise StartupIntegrityError(
            "Portfolio history is corrupted or invalid"
        ) from exc

    try:
        trade_history = (
            load_trade_history(
                trade_source
            )
        )

    except Exception as exc:
        raise StartupIntegrityError(
            "Trade history is corrupted or invalid"
        ) from exc

    # --------------------------------------------------------------
    # Portfolio-history chronology.
    # --------------------------------------------------------------

    portfolio_origins = (
        _validate_portfolio_origins(
            portfolio_history
        )
    )

    latest_origin = (
        portfolio_origins.iloc[
            -1
        ]
    )

    # --------------------------------------------------------------
    # State timestamp must agree with the latest journal row.
    # --------------------------------------------------------------

    if state.last_forecast_timestamp is None:
        raise StartupIntegrityError(
            "Portfolio history exists but state has no "
            "last_forecast_timestamp"
        )

    state_origin = (
        normalize_forecast_origin(
            state.last_forecast_timestamp
        )
    )

    if state_origin != latest_origin:
        raise StartupIntegrityError(
            "Persisted state's last forecast timestamp does not "
            "match the latest portfolio-history forecast origin"
        )

    # --------------------------------------------------------------
    # Validate trade journal.
    # --------------------------------------------------------------

    _validate_trade_history(
        trade_history=trade_history,
        portfolio_origins=(
            portfolio_origins
        ),
        tolerance=tolerance,
    )

    # --------------------------------------------------------------
    # Reconcile state and accounting journals.
    # --------------------------------------------------------------

    report = (
        reconcile_latest_paper_state(
            state=state,
            portfolio_history=(
                portfolio_history
            ),
            trade_history=(
                trade_history
            ),
            tolerance=tolerance,
        )
    )

    try:
        assert_reconciliation_passed(
            report
        )

    except ValueError as exc:
        raise StartupIntegrityError(
            "Paper-trading state and journals failed reconciliation"
        ) from exc

    return StartupValidationReport(
        clean_start=False,
        state_exists=True,
        portfolio_history_exists=True,
        trade_history_exists=(
            trade_history_exists
        ),
        latest_forecast_origin=(
            latest_origin.isoformat()
        ),
        cumulative_transaction_cost=float(
            report.cumulative_trade_transaction_cost
        ),
    )