from pathlib import Path

import pytest

from crypto_volatility.trading.persistence import (
    PaperRuntimeTransaction,
)


def build_transaction(
    tmp_path: Path,
) -> tuple[
    PaperRuntimeTransaction,
    Path,
    Path,
    Path,
]:
    state_path = (
        tmp_path
        / "paper_portfolio_state.json"
    )

    portfolio_history_path = (
        tmp_path
        / "paper_portfolio_history.csv"
    )

    trade_history_path = (
        tmp_path
        / "paper_trades.csv"
    )

    transaction = (
        PaperRuntimeTransaction(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_history_path
            ),
            trade_history_path=(
                trade_history_path
            ),
        )
    )

    return (
        transaction,
        state_path,
        portfolio_history_path,
        trade_history_path,
    )


def test_successful_transaction_keeps_changes(
    tmp_path,
):
    (
        transaction,
        state_path,
        portfolio_history_path,
        trade_history_path,
    ) = build_transaction(
        tmp_path
    )

    state_path.write_text(
        "old state",
        encoding="utf-8",
    )

    portfolio_history_path.write_text(
        "old portfolio history",
        encoding="utf-8",
    )

    trade_history_path.write_text(
        "old trades",
        encoding="utf-8",
    )

    with transaction:
        state_path.write_text(
            "new state",
            encoding="utf-8",
        )

        portfolio_history_path.write_text(
            "new portfolio history",
            encoding="utf-8",
        )

        trade_history_path.write_text(
            "new trades",
            encoding="utf-8",
        )

    assert (
        state_path.read_text(
            encoding="utf-8"
        )
        == "new state"
    )

    assert (
        portfolio_history_path.read_text(
            encoding="utf-8"
        )
        == "new portfolio history"
    )

    assert (
        trade_history_path.read_text(
            encoding="utf-8"
        )
        == "new trades"
    )

    assert not (
        transaction.manifest_path.exists()
    )


def test_failed_transaction_restores_existing_files(
    tmp_path,
):
    (
        transaction,
        state_path,
        portfolio_history_path,
        trade_history_path,
    ) = build_transaction(
        tmp_path
    )

    state_path.write_text(
        "old state",
        encoding="utf-8",
    )

    portfolio_history_path.write_text(
        "old portfolio",
        encoding="utf-8",
    )

    trade_history_path.write_text(
        "old trades",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="simulated failure",
    ):
        with transaction:
            state_path.write_text(
                "new state",
                encoding="utf-8",
            )

            portfolio_history_path.write_text(
                "new portfolio",
                encoding="utf-8",
            )

            raise RuntimeError(
                "simulated failure"
            )

    assert (
        state_path.read_text(
            encoding="utf-8"
        )
        == "old state"
    )

    assert (
        portfolio_history_path.read_text(
            encoding="utf-8"
        )
        == "old portfolio"
    )

    assert (
        trade_history_path.read_text(
            encoding="utf-8"
        )
        == "old trades"
    )


def test_failed_transaction_removes_new_files(
    tmp_path,
):
    (
        transaction,
        state_path,
        portfolio_history_path,
        trade_history_path,
    ) = build_transaction(
        tmp_path
    )

    state_path.write_text(
        "existing state",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError
    ):
        with transaction:
            portfolio_history_path.write_text(
                "new portfolio history",
                encoding="utf-8",
            )

            trade_history_path.write_text(
                "new trades",
                encoding="utf-8",
            )

            raise RuntimeError(
                "failure"
            )

    assert (
        state_path.read_text(
            encoding="utf-8"
        )
        == "existing state"
    )

    assert not (
        portfolio_history_path.exists()
    )

    assert not (
        trade_history_path.exists()
    )


def test_unfinished_transaction_can_be_recovered(
    tmp_path,
):
    (
        transaction,
        state_path,
        portfolio_history_path,
        trade_history_path,
    ) = build_transaction(
        tmp_path
    )

    state_path.write_text(
        "old state",
        encoding="utf-8",
    )

    portfolio_history_path.write_text(
        "old portfolio",
        encoding="utf-8",
    )

    trade_history_path.write_text(
        "old trades",
        encoding="utf-8",
    )

    # Simulate process starting a transaction.
    transaction.begin()

    state_path.write_text(
        "partially written state",
        encoding="utf-8",
    )

    portfolio_history_path.write_text(
        "partially written portfolio",
        encoding="utf-8",
    )

    # Simulate process crash:
    # no commit() and no rollback().

    recovery = (
        PaperRuntimeTransaction(
            state_path=state_path,
            portfolio_history_path=(
                portfolio_history_path
            ),
            trade_history_path=(
                trade_history_path
            ),
        )
    )

    recovered = (
        recovery.recover_if_needed()
    )

    assert recovered

    assert (
        state_path.read_text(
            encoding="utf-8"
        )
        == "old state"
    )

    assert (
        portfolio_history_path.read_text(
            encoding="utf-8"
        )
        == "old portfolio"
    )

    assert (
        trade_history_path.read_text(
            encoding="utf-8"
        )
        == "old trades"
    )


def test_no_recovery_when_no_pending_transaction(
    tmp_path,
):
    (
        transaction,
        _,
        _,
        _,
    ) = build_transaction(
        tmp_path
    )

    assert (
        transaction.recover_if_needed()
        is False
    )


def test_cannot_begin_second_transaction(
    tmp_path,
):
    (
        transaction,
        _,
        _,
        _,
    ) = build_transaction(
        tmp_path
    )

    transaction.begin()

    try:
        with pytest.raises(
            RuntimeError,
            match="already active",
        ):
            transaction.begin()

    finally:
        transaction.rollback()