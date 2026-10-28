from __future__ import annotations

import fcntl
import os
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO


class PaperTradingProcessLock:
    """
    Prevent multiple paper-trading processes from controlling the
    portfolio at the same time.

    Uses an operating-system advisory file lock.

    The lock is automatically released by the OS if the process
    exits or crashes.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(
            path
        )

        self.file: TextIO | None = None

        self.acquired = False

    def acquire(
        self,
    ) -> None:
        """
        Acquire the trading-process lock without waiting.

        Raises RuntimeError when another process already owns it.
        """
        if self.acquired:
            raise RuntimeError(
                "Paper-trading process lock is already acquired"
            )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file = self.path.open(
            "a+",
            encoding="utf-8",
        )

        try:
            fcntl.flock(
                file.fileno(),
                fcntl.LOCK_EX
                | fcntl.LOCK_NB,
            )

        except BlockingIOError as exc:
            file.close()

            raise RuntimeError(
                "Another paper-trading process is already running"
            ) from exc

        file.seek(
            0
        )

        file.truncate()

        file.write(
            f"pid={os.getpid()}\n"
        )

        file.write(
            "acquired_at="
            f"{datetime.now(UTC).isoformat()}\n"
        )

        file.flush()

        self.file = file

        self.acquired = True

    def release(
        self,
    ) -> None:
        """
        Release the process lock.
        """
        if not self.acquired:
            return

        if self.file is None:
            self.acquired = False
            return

        try:
            fcntl.flock(
                self.file.fileno(),
                fcntl.LOCK_UN,
            )

        finally:
            self.file.close()

            self.file = None

            self.acquired = False

    def __enter__(
        self,
    ) -> PaperTradingProcessLock:
        self.acquire()

        return self

    def __exit__(
        self,
        exc_type: type[
            BaseException
        ] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.release()

        return False