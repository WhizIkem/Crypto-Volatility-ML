from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path


@dataclass(frozen=True)
class PaperTradingRuntimeConfig:
    """
    Runtime configuration for the paper-trading system.
    """

    initial_capital: float
    transaction_cost_bps: float
    turnover_threshold: float

    state_path: Path
    portfolio_history_path: Path
    trade_history_path: Path
    log_path: Path
    lock_path: Path


def _read_float_env(
    name: str,
    default: float,
) -> float:
    """
    Read a floating-point environment variable.
    """
    raw_value = os.getenv(
        name
    )

    if raw_value is None:
        return float(
            default
        )

    try:
        return float(
            raw_value
        )

    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid number"
        ) from exc


def load_paper_trading_runtime_config(
) -> PaperTradingRuntimeConfig:
    """
    Load paper-trading runtime configuration.

    Environment variables may override safe defaults.
    """
    initial_capital = _read_float_env(
        "PAPER_INITIAL_CAPITAL",
        1_000_000.0,
    )

    transaction_cost_bps = _read_float_env(
        "PAPER_TRANSACTION_COST_BPS",
        10.0,
    )

    turnover_threshold = _read_float_env(
        "PAPER_TURNOVER_THRESHOLD",
        0.05,
    )

    if initial_capital <= 0:
        raise ValueError(
            "PAPER_INITIAL_CAPITAL must be positive"
        )

    if transaction_cost_bps < 0:
        raise ValueError(
            "PAPER_TRANSACTION_COST_BPS cannot be negative"
        )

    if not (
        0.0
        <= turnover_threshold
        <= 1.0
    ):
        raise ValueError(
            "PAPER_TURNOVER_THRESHOLD must be between 0 and 1"
        )

    return PaperTradingRuntimeConfig(
        initial_capital=initial_capital,
        transaction_cost_bps=(
            transaction_cost_bps
        ),
        turnover_threshold=(
            turnover_threshold
        ),
        state_path=Path(
            os.getenv(
                "PAPER_STATE_PATH",
                "logs/paper_portfolio_state.json",
            )
        ),
        portfolio_history_path=Path(
            os.getenv(
                "PAPER_PORTFOLIO_HISTORY_PATH",
                "logs/paper_portfolio_history.csv",
            )
        ),
        trade_history_path=Path(
            os.getenv(
                "PAPER_TRADE_HISTORY_PATH",
                "logs/paper_trades.csv",
            )
        ),
        log_path=Path(
            os.getenv(
                "PAPER_LOG_PATH",
                "logs/paper_trading.log",
            )
        ),
        lock_path=Path(
            os.getenv(
                "PAPER_LOCK_PATH",
                "logs/paper_trading.lock",
            )
        ),
    )


def configure_runtime_logging(
    *,
    log_path: str | Path,
    logger_name: str = "crypto_volatility.paper_trading",
) -> logging.Logger:
    """
    Configure console and rotating-file logging.

    Repeated calls do not duplicate handlers.
    """
    destination = Path(
        log_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        logger_name
    )

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setLevel(
        logging.INFO
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = (
        RotatingFileHandler(
            destination,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
    )

    file_handler.setLevel(
        logging.INFO
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger