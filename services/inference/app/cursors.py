"""Persist the last completed decision for each instrument."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


class DecisionCursors:
    """Advance monotonic timestamps with an atomic, durable replacement."""

    def __init__(self, path: str = "") -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._owner = None
        self._values: dict[str, datetime] = {}
        self._reload()

    def _reload(self) -> None:
        if self._path and self._path.exists():
            document = json.loads(self._path.read_text())
            if document.get("version") != 1:
                raise ValueError("unsupported decision cursor version")
            for symbol, stamp in document["cursors"].items():
                parsed = datetime.fromisoformat(stamp)
                if parsed.tzinfo is None:
                    raise ValueError("decision cursor must have a timezone")
                self._values[symbol] = parsed.astimezone(timezone.utc)

    def claim_owner(self) -> None:
        """Reject a second runtime sharing this journal on the same host."""
        if self._path is None:
            return
        with self._lock:
            if self._owner is not None:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            owner = open(self._path.with_suffix(".lock"), "a")
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                owner.close()
                raise RuntimeError("decision cursor journal is already owned") from error
            try:
                self._reload()
            except Exception:
                owner.close()
                raise
            self._owner = owner

    def close(self) -> None:
        """Release the runtime's exclusive local journal ownership."""
        with self._lock:
            if self._owner is not None:
                self._owner.close()
                self._owner = None

    def contains(self, symbol: str, stamp: datetime) -> bool:
        with self._lock:
            previous = self._values.get(symbol)
            return previous is not None and stamp <= previous

    def advance(self, symbol: str, stamp: datetime) -> None:
        """Persist before publishing a new in-memory cursor; errors propagate."""
        with self._lock:
            previous = self._values.get(symbol)
            if previous is not None and stamp <= previous:
                return
            values = {**self._values, symbol: stamp.astimezone(timezone.utc)}
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=self._path.name + ".", suffix=".tmp", dir=self._path.parent)
                temporary = Path(name)
                document = {"version": 1, "cursors": {key: value.isoformat() for key, value in values.items()}}
                try:
                    with os.fdopen(fd, "w") as handle:
                        json.dump(document, handle, sort_keys=True)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self._path)
                    directory = os.open(self._path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    temporary.unlink(missing_ok=True)
            self._values = values
