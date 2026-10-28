from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


@dataclass(frozen=True)
class RuntimeFile:
    """
    Runtime file participating in a paper-trading transaction.
    """

    name: str
    path: Path


class PaperRuntimeTransaction:
    """
    Crash-recoverable transaction across paper-trading runtime files.

    Before any runtime file is changed, existing versions are copied
    into a transaction directory and a manifest is written.

    If execution fails, the previous files are restored.

    If the process crashes before commit, the manifest remains and
    recovery can restore the previous consistent state on the next run.
    """

    def __init__(
        self,
        *,
        state_path: str | Path,
        portfolio_history_path: str | Path,
        trade_history_path: str | Path,
    ) -> None:
        self.files = [
            RuntimeFile(
                name="state",
                path=Path(
                    state_path
                ),
            ),
            RuntimeFile(
                name="portfolio_history",
                path=Path(
                    portfolio_history_path
                ),
            ),
            RuntimeFile(
                name="trade_history",
                path=Path(
                    trade_history_path
                ),
            ),
        ]

        self.transaction_dir = (
            Path(
                state_path
            ).parent
            / ".paper_runtime_transaction"
        )

        self.manifest_path = (
            self.transaction_dir
            / "manifest.json"
        )

        self.active = False

    def _backup_path(
        self,
        runtime_file: RuntimeFile,
    ) -> Path:
        return (
            self.transaction_dir
            / f"{runtime_file.name}.backup"
        )

    def begin(
        self,
    ) -> None:
        """
        Begin a new runtime transaction.
        """
        if self.active:
            raise RuntimeError(
                "Paper runtime transaction is already active"
            )

        if self.manifest_path.exists():
            raise RuntimeError(
                "Unfinished paper runtime transaction exists. "
                "Recover it before starting another transaction."
            )

        self.transaction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_files = []

        # ----------------------------------------------------------
        # Create all backups before writing the manifest.
        #
        # Target files must not be modified until begin() returns.
        # ----------------------------------------------------------

        for runtime_file in self.files:
            source = runtime_file.path

            existed = source.exists()

            backup_path = self._backup_path(
                runtime_file
            )

            if backup_path.exists():
                backup_path.unlink()

            if existed:
                shutil.copy2(
                    source,
                    backup_path,
                )

            manifest_files.append(
                {
                    "name": runtime_file.name,
                    "path": str(
                        source
                    ),
                    "existed": existed,
                    "backup_path": str(
                        backup_path
                    ),
                }
            )

        manifest = {
            "status": "pending",
            "files": manifest_files,
        }

        temporary_manifest = (
            self.manifest_path
            .with_suffix(
                ".json.tmp"
            )
        )

        with temporary_manifest.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )

        temporary_manifest.replace(
            self.manifest_path
        )

        self.active = True

    def commit(
        self,
    ) -> None:
        """
        Commit the runtime transaction.

        The manifest is removed first. Once the manifest is gone,
        the transaction is considered committed.
        """
        if not self.active:
            raise RuntimeError(
                "No active paper runtime transaction"
            )

        if self.manifest_path.exists():
            self.manifest_path.unlink()

        self.active = False

        self._cleanup_transaction_directory()

    def rollback(
        self,
    ) -> None:
        """
        Restore all runtime files to their pre-transaction state.
        """
        if not self.manifest_path.exists():
            self.active = False
            return

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(
                file
            )

        for item in manifest[
            "files"
        ]:
            target = Path(
                item[
                    "path"
                ]
            )

            backup = Path(
                item[
                    "backup_path"
                ]
            )

            existed = bool(
                item[
                    "existed"
                ]
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if existed:
                if not backup.exists():
                    raise RuntimeError(
                        "Paper runtime recovery backup is missing: "
                        f"{backup}"
                    )

                temporary_restore = (
                    target
                    .with_suffix(
                        target.suffix
                        + ".restore"
                    )
                )

                shutil.copy2(
                    backup,
                    temporary_restore,
                )

                temporary_restore.replace(
                    target
                )

            elif target.exists():
                target.unlink()

        if self.manifest_path.exists():
            self.manifest_path.unlink()

        self.active = False

        self._cleanup_transaction_directory()

    def recover_if_needed(
        self,
    ) -> bool:
        """
        Recover an unfinished transaction.

        Returns True when recovery was required.
        """
        if not self.manifest_path.exists():
            return False

        self.rollback()

        return True

    def _cleanup_transaction_directory(
        self,
    ) -> None:
        """
        Remove transaction backups after commit or rollback.
        """
        if not self.transaction_dir.exists():
            return

        for path in (
            self.transaction_dir
            .iterdir()
        ):
            if path.is_file():
                path.unlink()

        try:
            self.transaction_dir.rmdir()
        except OSError:
            # Leave unexpected contents untouched rather than deleting
            # data recursively.
            pass

    def __enter__(
        self,
    ) -> PaperRuntimeTransaction:
        self.begin()

        return self

    def __exit__(
        self,
        exc_type: type[
            BaseException
        ] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            self.rollback()

            return False

        self.commit()

        return False