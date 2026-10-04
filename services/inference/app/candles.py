"""Reading the Parquet cold store from Python."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("mft.inference.candles")


CANDLES_DATASET = "candles"


PART_GLOB = "part-*.parquet"
MAX_PENDING_READS = 8
DEFAULT_MAX_ROWS = 100_001
DEFAULT_CACHED_ROWS = 32_768
DEFAULT_CACHED_PARTITIONS = 256
DEFAULT_CACHED_DIRECTORIES = 256
DEFAULT_CACHED_DIRECTORY_ENTRIES = 8_192


class CandleReadLimitError(ValueError):
    """A requested tail exceeds the configured materialized-row budget."""


_PARTITION_SQL = """
WITH deduped AS (
    SELECT symbol, timestamp, open, high, low, close, volume, filename, file_row_number
    FROM read_parquet(?, hive_partitioning = false, filename = true, file_row_number = true)
    WHERE symbol = ?
    QUALIFY row_number() OVER (
        PARTITION BY timestamp ORDER BY filename DESC, file_row_number DESC
    ) = 1
)
SELECT symbol, timestamp, open, high, low, close, volume, filename, file_row_number
FROM deduped ORDER BY timestamp DESC LIMIT ?
"""


@dataclass(frozen=True, slots=True)
class Candle:
    """One aggregated 1-minute bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> Candle:
        """Build from a DuckDB partition-query row."""
        return cls(
            symbol=str(row[0]),
            timestamp=_from_unix_millis(int(row[1])),
            open=float(row[2]),
            high=float(row[3]),
            low=float(row[4]),
            close=float(row[5]),
            volume=int(row[6]),
        )


@runtime_checkable
class CandleStore(Protocol):
    """What the inference loop needs from the Parquet store."""

    async def tail(self, symbol: str, limit: int) -> list[Candle]:
        """Return up to `limit` candles for `symbol`, oldest first."""
        ...


def escape_hive_value(value: str) -> str:
    """Percent-encode a hive partition value."""
    if not value:
        return "%00"
    return "".join(
        char if (char.isascii() and (char.isalnum() or char in "._-")) else
        "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
        for char in value
    )


def candles_root(data_dir: str | os.PathLike[str]) -> Path:
    """`<data_dir>/candles`, defaulting an empty data_dir to \"data\"."""
    base = Path(data_dir) if str(data_dir) else Path("data")
    return base / CANDLES_DATASET


def symbol_glob(root: Path, symbol: str) -> str:
    """The path expression for one symbol."""
    return str(root / f"symbol={escape_hive_value(symbol)}" / "date=*" / PART_GLOB)


def _from_unix_millis(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


class DuckDBCandleStore:
    """A `CandleStore` over the hive-partitioned Parquet candles dataset."""

    def __init__(
        self, root: Path, connection: Any | None = None, *,
        historical_root: Path | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_cached_rows: int = DEFAULT_CACHED_ROWS,
        max_cached_partitions: int = DEFAULT_CACHED_PARTITIONS,
        max_cached_directories: int = DEFAULT_CACHED_DIRECTORIES,
        max_cached_directory_entries: int = DEFAULT_CACHED_DIRECTORY_ENTRIES,
    ) -> None:
        for name, value in (
            ("max_rows", max_rows), ("max_cached_rows", max_cached_rows),
            ("max_cached_partitions", max_cached_partitions),
            ("max_cached_directories", max_cached_directories),
            ("max_cached_directory_entries", max_cached_directory_entries),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        import duckdb

        self._root = Path(root).resolve()
        self._historical_root = (historical_root or self._root.parent / "historical").resolve()
        self._connection = connection if connection is not None else duckdb.connect(
            config={"threads": 2, "memory_limit": "256MB"},
        )
        self._owns_connection = connection is None

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="candles"
        )
        self._closed = False
        self._read_slots = threading.BoundedSemaphore(MAX_PENDING_READS)
        self._max_rows = max_rows
        self._max_cached_rows = max_cached_rows
        self._max_cached_partitions = max_cached_partitions
        self._max_cached_directories = max_cached_directories
        self._max_cached_directory_entries = max_cached_directory_entries
        self._cached_rows = 0
        self._cache: OrderedDict[
            tuple[str, Path], tuple[int, int, int, list[tuple[Candle, str, int, bool]]]
        ] = OrderedDict()
        self._directory_cache: OrderedDict[Path, tuple[int, int, tuple[str, ...]]] = OrderedDict()

    @property
    def capacity(self) -> int:
        """Maximum admitted reads, including the active worker."""
        return MAX_PENDING_READS

    @property
    def root(self) -> Path:
        """The dataset directory being scanned."""
        return self._root

    async def tail(self, symbol: str, limit: int) -> list[Candle]:
        """Return up to `limit` candles for `symbol`, oldest first."""
        if self._closed:
            raise RuntimeError("candle store is closed")
        if type(limit) is not int:
            raise CandleReadLimitError("candle limit must be an integer")
        if limit > self._max_rows:
            raise CandleReadLimitError(f"candle limit exceeds max_rows={self._max_rows}")
        if limit <= 0:
            return []
        if not self._read_slots.acquire(blocking=False):
            raise RuntimeError("candle reader is at capacity")
        try:
            future = self._executor.submit(self._tail_sync, symbol, limit)
        except BaseException:
            self._read_slots.release()
            raise
        future.add_done_callback(lambda _: self._read_slots.release())
        return await asyncio.wrap_future(future)

    async def aclose(self) -> None:
        """Release the connection and the worker thread."""
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        if self._owns_connection:
            self._connection.close()

    def _tail_sync(self, symbol: str, limit: int) -> list[Candle]:
        if limit <= 0:
            return []
        partitions = self._partitions(symbol)
        live_paths = {path for path, _, _ in partitions}
        for key in tuple(self._cache):
            if key[0] == symbol and key[1] not in live_paths:
                self._cached_rows -= len(self._cache.pop(key)[3])

        def rows_for(path: Path, is_live: bool, stamp: os.stat_result) -> list[tuple[Candle, str, int, bool]]:
            key = (symbol, path)
            cached = self._cache.get(key)
            if (
                cached and cached[:2] == (stamp.st_mtime_ns, stamp.st_size)
                and cached[2] >= limit
            ):
                self._cache.move_to_end(key)
                return cached[3][:limit]
            files = sorted(str(p) for p in path.glob(PART_GLOB if is_live else "candles.parquet"))
            if not files:
                rows: list[tuple[Candle, str, int, bool]] = []
            else:
                result = self._connection.execute(_PARTITION_SQL, [files, symbol, limit]).fetchall()
                rows = [(Candle.from_row(row), str(row[7]), int(row[8]), is_live) for row in result]
            if cached:
                self._cached_rows -= len(self._cache.pop(key)[3])
            if len(rows) <= self._max_cached_rows:
                self._cache[key] = (stamp.st_mtime_ns, stamp.st_size, limit, rows)
                self._cached_rows += len(rows)
                while (
                    self._cached_rows > self._max_cached_rows
                    or len(self._cache) > self._max_cached_partitions
                ):
                    _, evicted = self._cache.popitem(last=False)
                    self._cached_rows -= len(evicted[3])
            return rows

        winners: dict[datetime, tuple[Candle, str, int, bool]] = {}

        def merge(rows: list[tuple[Candle, str, int, bool]]) -> None:
            for row in rows:
                candle, filename, row_number, is_live = row
                old = winners.get(candle.timestamp)
                if old is None or (is_live, filename, row_number) > (old[3], old[1], old[2]):
                    winners[candle.timestamp] = row
            if len(winners) > limit:
                for stamp in sorted(winners)[:-limit]:
                    del winners[stamp]

        live = sorted(
            ((path, stamp) for path, is_live, stamp in partitions if is_live),
            key=lambda item: item[0].name,
            reverse=True,
        )
        for path, stamp in live:
            merge(rows_for(path, True, stamp))
            if len(winners) >= limit:
                break

        cutoff = min(winners) if len(winners) >= limit else None
        for path, is_live, stamp in partitions:
            if is_live:
                continue
            bucket_end = datetime.strptime(path.name.removeprefix("to="), "%Y-%m-%d").date()
            if cutoff is not None and bucket_end < cutoff.date():
                continue
            merge(rows_for(path, False, stamp))
        newest = sorted(winners.values(), key=lambda row: row[0].timestamp)[-limit:]
        return [row[0] for row in newest]

    def _partitions(self, symbol: str) -> list[tuple[Path, bool, os.stat_result]]:
        symbol_dir = self._root / f"symbol={escape_hive_value(symbol)}"
        historical_dir = self._historical_root / f"symbol={escape_hive_value(symbol)}"
        found: list[tuple[Path, bool, os.stat_result]] = []
        for base, live in ((symbol_dir, True), (historical_dir, False)):
            for child_name in self._child_dirs(base):
                if live:
                    if not child_name.startswith("date="):
                        continue
                    path = base / child_name
                else:
                    if not child_name.startswith("from="):
                        continue
                    from_path = base / child_name
                    for end_name in self._child_dirs(from_path):
                        if not end_name.startswith("to="):
                            continue
                        path = from_path / end_name
                        if not (path / "candles.parquet").is_file():
                            continue
                        try:
                            stamp = path.stat()
                        except FileNotFoundError:
                            continue
                        found.append((path, False, stamp))
                    continue
                try:
                    stamp = path.stat()
                except FileNotFoundError:
                    continue
                found.append((path, True, stamp))
        return found

    def _child_dirs(self, path: Path) -> tuple[str, ...]:
        try:
            info = path.stat()
        except FileNotFoundError:
            self._directory_cache.pop(path, None)
            return ()
        cached = self._directory_cache.get(path)
        if cached and cached[:2] == (info.st_mtime_ns, info.st_size):
            self._directory_cache.move_to_end(path)
            return cached[2]
        previous = set(cached[2]) if cached else set()
        with os.scandir(path) as entries:
            children = tuple(sorted(entry.name for entry in entries if entry.is_dir(follow_symlinks=False)))
        for removed in previous - set(children):
            deleted = path / removed
            for cached_path in tuple(self._directory_cache):
                if cached_path == deleted or deleted in cached_path.parents:
                    del self._directory_cache[cached_path]
        self._directory_cache.pop(path, None)
        if len(children) <= self._max_cached_directory_entries:
            self._directory_cache[path] = (info.st_mtime_ns, info.st_size, children)
            while (
                len(self._directory_cache) > self._max_cached_directories
                or sum(len(entry[2]) for entry in self._directory_cache.values())
                > self._max_cached_directory_entries
            ):
                self._directory_cache.popitem(last=False)
        return children


__all__ = [
    "CANDLES_DATASET",
    "Candle",
    "CandleReadLimitError",
    "CandleStore",
    "DuckDBCandleStore",
    "PART_GLOB",
    "candles_root",
    "escape_hive_value",
    "symbol_glob",
]
