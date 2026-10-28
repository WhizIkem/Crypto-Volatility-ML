import pytest

from crypto_volatility.trading.process_lock import (
    PaperTradingProcessLock,
)


def test_lock_can_be_acquired_and_released(
    tmp_path,
):
    path = (
        tmp_path
        / "paper_trading.lock"
    )

    lock = (
        PaperTradingProcessLock(
            path
        )
    )

    lock.acquire()

    assert lock.acquired
    assert path.exists()

    lock.release()

    assert not lock.acquired


def test_context_manager_releases_lock(
    tmp_path,
):
    path = (
        tmp_path
        / "paper_trading.lock"
    )

    lock = (
        PaperTradingProcessLock(
            path
        )
    )

    with lock:
        assert lock.acquired

    assert not lock.acquired


def test_second_lock_is_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "paper_trading.lock"
    )

    first = (
        PaperTradingProcessLock(
            path
        )
    )

    second = (
        PaperTradingProcessLock(
            path
        )
    )

    first.acquire()

    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Another paper-trading process "
                "is already running"
            ),
        ):
            second.acquire()

    finally:
        first.release()


def test_lock_available_after_release(
    tmp_path,
):
    path = (
        tmp_path
        / "paper_trading.lock"
    )

    first = (
        PaperTradingProcessLock(
            path
        )
    )

    second = (
        PaperTradingProcessLock(
            path
        )
    )

    first.acquire()

    first.release()

    second.acquire()

    try:
        assert second.acquired

    finally:
        second.release()


def test_exception_releases_context_manager_lock(
    tmp_path,
):
    path = (
        tmp_path
        / "paper_trading.lock"
    )

    first = (
        PaperTradingProcessLock(
            path
        )
    )

    with pytest.raises(
        RuntimeError,
        match="simulated failure",
    ):
        with first:
            raise RuntimeError(
                "simulated failure"
            )

    assert not first.acquired

    second = (
        PaperTradingProcessLock(
            path
        )
    )

    second.acquire()

    second.release()


def test_double_acquire_is_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "paper_trading.lock"
    )

    lock = (
        PaperTradingProcessLock(
            path
        )
    )

    lock.acquire()

    try:
        with pytest.raises(
            RuntimeError,
            match="already acquired",
        ):
            lock.acquire()

    finally:
        lock.release()