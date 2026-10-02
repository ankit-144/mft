"""Reading the Parquet cold store from Python.

# Why DuckDB and not pyarrow

Both are declared in the frozen `requirements.txt`, so this is a choice rather
than a constraint. DuckDB wins for two reasons, and the first is not a matter
of taste:

1. `storage.CandleReader.Glob` (`core/storage/reader.go:188`) exists to hand
   this exact path expression to DuckDB — its docstring says "It is the path
   expression DuckDB consumes from the Python side, so a research query and the
   Go reader address the same files." Reading with pyarrow would mean
   re-deriving the hive layout in a second language, and the two derivations
   would drift the first time the layout changed.
2. The pull wants a *tail*: the last N candles of a symbol, ascending. DuckDB
   pushes the sort and the limit into the file readers with predicate
   projection, so only the tail's pages are decoded. pyarrow would need the
   file list globbed in Python, every file's row groups opened, and the merge
   sorted in Python.

# Why the connection is in-memory

`analytics.read_only` defaults to true and `analytics.duckdb_path` is a file
another process may hold. Attaching to it would put inference behind a
cross-process lock for no benefit: the store is read here, never written, and
the scan is over Parquet files that are immutable by C2's invariant.

# The immutability invariant we are leaning on

C2 writes to `.inflight-*.parquet` and renames to `part-<timestamp>.parquet`
only once the file is complete. The glob used here matches `part-*.parquet`
and nothing else, so this reader can never observe a half-written file while
ingestion is flushing beside it. That is the whole reason pull beats push here
(Plan.md §4).
"""

from __future__ import annotations

import asyncio
import glob as globlib
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("mft.inference.candles")

#: Dataset directory name under `storage.data_dir`. Mirrors
#: `storage.DatasetCandles`.
CANDLES_DATASET = "candles"

#: Published-file pattern. Never `.inflight-*`: see the module docstring.
PART_GLOB = "part-*.parquet"

#: How many raw rows to scan to return `limit` distinct candles.
#:
#: A timestamp can appear in more than one file when a backfill and a live
#: flush covered the same minute, and the Go reader resolves that with "a later
#: file wins". Reproducing the tie-break means scanning a few files' worth of
#: rows before de-duplicating, or the de-duplication could not see both copies.
#: Three is generous for a store where each flush covers minutes, not seconds.
_TAIL_OVERSAMPLE = 3

_TAIL_SQL = """
WITH scanned AS (
    SELECT symbol, timestamp, open, high, low, close, volume, filename
    FROM read_parquet({glob}, hive_partitioning = false)
    WHERE symbol = {symbol}
    ORDER BY timestamp DESC
    LIMIT {scan}
),
deduped AS (
    SELECT symbol, timestamp, open, high, low, close, volume
    FROM scanned
    QUALIFY row_number() OVER (
        PARTITION BY timestamp ORDER BY filename DESC
    ) = 1
)
SELECT symbol, timestamp, open, high, low, close, volume
FROM deduped
ORDER BY timestamp ASC
"""


@dataclass(frozen=True, slots=True)
class Candle:
    """One aggregated 1-minute bar. Mirrors `contracts.Candle` (docs §1)."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> Candle:
        """Build from a DuckDB row in the column order of `_TAIL_SQL`."""
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
    """What the inference loop needs from the Parquet store.

    Async because the implementations block: DuckDB calls are synchronous C++
    and would otherwise stall the event loop, which also serves the debug
    endpoints and the scheduler.
    """

    async def tail(self, symbol: str, limit: int) -> list[Candle]:
        """Return up to `limit` candles for `symbol`, oldest first.

        A symbol with no stored candles yields an empty list, not an error:
        one instrument the broker has not streamed yet must not look like a
        fault to the caller, which skips and logs on an empty window anyway.
        """
        ...


def escape_hive_value(value: str) -> str:
    """Percent-encode a hive partition value. Mirrors `escapeHiveValue`.

    Symbols are usually plain uppercase alphanumerics, but nothing guarantees
    it, and a symbol containing '/' or '=' would silently relocate a dataset.
    """
    if not value:
        return "%00"
    return "".join(
        char if (char.isascii() and (char.isalnum() or char in "._-")) else
        "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
        for char in value
    )


def candles_root(data_dir: str | os.PathLike[str]) -> Path:
    """`<data_dir>/candles`, defaulting an empty data_dir to "data".

    Matches `storage.dataDirFor` so the Python and Go readers address one tree.
    """
    base = Path(data_dir) if str(data_dir) else Path("data")
    return base / CANDLES_DATASET


def symbol_glob(root: Path, symbol: str) -> str:
    """The path expression for one symbol. Mirrors `CandleReader.Glob`."""
    return str(root / f"symbol={escape_hive_value(symbol)}" / "date=*" / PART_GLOB)


def _from_unix_millis(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


class DuckDBCandleStore:
    """A `CandleStore` over the hive-partitioned Parquet candles dataset.

    Args:
        root: The dataset directory, normally `<data_dir>/candles`.
        connection: An existing DuckDB connection. One is created in memory
            when omitted, and `owns_connection` is then false.
    """

    def __init__(self, root: Path, connection: Any | None = None) -> None:
        import duckdb  # imported lazily: research tooling should not pay for it

        self._root = Path(root)
        self._connection = connection if connection is not None else duckdb.connect()
        self._owns_connection = connection is None
        # A duckdb connection is not safe for concurrent use, so every query
        # goes through one worker thread and the loop stays responsive.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="candles"
        )
        self._closed = False

    @property
    def root(self) -> Path:
        """The dataset directory being scanned."""
        return self._root

    async def tail(self, symbol: str, limit: int) -> list[Candle]:
        """Return up to `limit` candles for `symbol`, oldest first."""
        return await self._run(self._tail_sync, symbol, limit)

    async def aclose(self) -> None:
        """Release the connection and the worker thread."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        if self._owns_connection:
            self._connection.close()

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def _tail_sync(self, symbol: str, limit: int) -> list[Candle]:
        if limit <= 0:
            return []
        pattern = symbol_glob(self._root, symbol)
        # `read_parquet` raises on a glob that matches nothing; the Go reader
        # returns an empty slice instead, and a symbol the broker has not
        # streamed yet is an ordinary state, not an error.
        if not globlib.glob(pattern):
            logger.debug("no candle files match %s", pattern)
            return []

        sql = _TAIL_SQL.format(
            glob=_sql_literal(pattern), symbol=_sql_literal(symbol), scan=limit * _TAIL_OVERSAMPLE
        )
        rows = self._connection.execute(sql).fetchall()
        candles = [Candle.from_row(row) for row in rows]
        if len(candles) > limit:
            # The oversampled scan can still resolve to more than `limit`
            # distinct minutes when a single flush was very wide.
            candles = candles[-limit:]
        logger.debug("read %d candles for %s from %s", len(candles), symbol, pattern)
        return candles


def _sql_literal(value: str) -> str:
    """Quote a string for inlining into the tail query.

    The symbol reaches the query only through `escape_hive_value` and a
    doubling of embedded quotes, but the value still originates in
    configuration, so it is escaped rather than trusted.
    """
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "CANDLES_DATASET",
    "Candle",
    "CandleStore",
    "DuckDBCandleStore",
    "PART_GLOB",
    "candles_root",
    "escape_hive_value",
    "symbol_glob",
]
