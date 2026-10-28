import logging

import pytest

from crypto_volatility.trading.runtime import (
    configure_runtime_logging,
    load_paper_trading_runtime_config,
)


def test_runtime_config_defaults(
    monkeypatch,
):
    names = [
        "PAPER_INITIAL_CAPITAL",
        "PAPER_TRANSACTION_COST_BPS",
        "PAPER_TURNOVER_THRESHOLD",
        "PAPER_STATE_PATH",
        "PAPER_PORTFOLIO_HISTORY_PATH",
        "PAPER_TRADE_HISTORY_PATH",
        "PAPER_LOG_PATH",
        "PAPER_LOCK_PATH",
    ]

    for name in names:
        monkeypatch.delenv(
            name,
            raising=False,
        )

    config = (
        load_paper_trading_runtime_config()
    )

    assert (
        config.initial_capital
        == pytest.approx(
            1_000_000.0
        )
    )

    assert (
        config.transaction_cost_bps
        == pytest.approx(
            10.0
        )
    )

    assert (
        config.turnover_threshold
        == pytest.approx(
            0.05
        )
    )

    assert str(
        config.state_path
    ) == (
        "logs/paper_portfolio_state.json"
    )

    assert str(
        config.portfolio_history_path
    ) == (
        "logs/paper_portfolio_history.csv"
    )

    assert str(
        config.trade_history_path
    ) == (
        "logs/paper_trades.csv"
    )

    assert str(
        config.log_path
    ) == (
        "logs/paper_trading.log"
    )

    assert str(
        config.lock_path
    ) == (
        "logs/paper_trading.lock"
    )


def test_runtime_config_environment_override(
    monkeypatch,
):
    monkeypatch.setenv(
        "PAPER_INITIAL_CAPITAL",
        "2500000",
    )

    monkeypatch.setenv(
        "PAPER_TRANSACTION_COST_BPS",
        "8",
    )

    monkeypatch.setenv(
        "PAPER_TURNOVER_THRESHOLD",
        "0.03",
    )

    monkeypatch.setenv(
        "PAPER_STATE_PATH",
        "runtime/state.json",
    )

    monkeypatch.setenv(
        "PAPER_PORTFOLIO_HISTORY_PATH",
        "runtime/portfolio.csv",
    )

    monkeypatch.setenv(
        "PAPER_TRADE_HISTORY_PATH",
        "runtime/trades.csv",
    )

    monkeypatch.setenv(
        "PAPER_LOG_PATH",
        "runtime/trading.log",
    )

    monkeypatch.setenv(
        "PAPER_LOCK_PATH",
        "runtime/trading.lock",
    )

    config = (
        load_paper_trading_runtime_config()
    )

    assert (
        config.initial_capital
        == pytest.approx(
            2_500_000.0
        )
    )

    assert (
        config.transaction_cost_bps
        == pytest.approx(
            8.0
        )
    )

    assert (
        config.turnover_threshold
        == pytest.approx(
            0.03
        )
    )

    assert str(
        config.state_path
    ) == "runtime/state.json"

    assert str(
        config.portfolio_history_path
    ) == "runtime/portfolio.csv"

    assert str(
        config.trade_history_path
    ) == "runtime/trades.csv"

    assert str(
        config.log_path
    ) == "runtime/trading.log"

    assert str(
        config.lock_path
    ) == "runtime/trading.lock"


def test_invalid_initial_capital(
    monkeypatch,
):
    monkeypatch.setenv(
        "PAPER_INITIAL_CAPITAL",
        "0",
    )

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        load_paper_trading_runtime_config()


def test_invalid_transaction_cost(
    monkeypatch,
):
    monkeypatch.setenv(
        "PAPER_TRANSACTION_COST_BPS",
        "-1",
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        load_paper_trading_runtime_config()


def test_invalid_turnover_threshold(
    monkeypatch,
):
    monkeypatch.setenv(
        "PAPER_TURNOVER_THRESHOLD",
        "1.5",
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        load_paper_trading_runtime_config()


def test_non_numeric_environment_value(
    monkeypatch,
):
    monkeypatch.setenv(
        "PAPER_INITIAL_CAPITAL",
        "not-a-number",
    )

    with pytest.raises(
        ValueError,
        match="valid number",
    ):
        load_paper_trading_runtime_config()


def test_runtime_logging_creates_file(
    tmp_path,
):
    log_path = (
        tmp_path
        / "paper_trading.log"
    )

    logger_name = (
        "crypto_volatility.test_runtime"
    )

    logger = (
        configure_runtime_logging(
            log_path=log_path,
            logger_name=logger_name,
        )
    )

    logger.info(
        "test runtime message"
    )

    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()

    contents = (
        log_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "test runtime message"
        in contents
    )

    for handler in list(
        logger.handlers
    ):
        logger.removeHandler(
            handler
        )

        handler.close()

    logging.getLogger(
        logger_name
    ).handlers.clear()