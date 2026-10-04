from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.candles import CandleReadLimitError, DuckDBCandleStore
from support import ANCHOR, make_candles
from test_candles import SCHEMA, write_part


class RecordingConnection:
    def __init__(self, connection: object) -> None:
        self.connection = connection
        self.file_lists: list[list[str]] = []

    def execute(self, query: str, params: list[object]) -> object:
        self.file_lists.append(list(params[0]))
        return self.connection.execute(query, params)  # type: ignore[attr-defined]

    def close(self) -> None:
        self.connection.close()  # type: ignore[attr-defined]


def test_tail_prunes_older_live_dates_after_newest_partition_satisfies_limit(tmp_path: Path) -> None:
    root = tmp_path / "candles"
    days = 12
    for day in range(days):
        rows = make_candles(rows=4, symbol="RELIANCE", start=ANCHOR + timedelta(days=day))
        write_part(root, "RELIANCE", rows, part=f"part-{day:02d}")

    store = DuckDBCandleStore(root)
    recording = RecordingConnection(store._connection)
    store._connection = recording
    try:
        rows = asyncio.run(store.tail("RELIANCE", 3))
    finally:
        asyncio.run(store.aclose())

    assert [row.timestamp for row in rows] == [
        ANCHOR + timedelta(days=days - 1, minutes=n) for n in (1, 2, 3)
    ]
    assert len(recording.file_lists) == 1
    assert all(f"date={(ANCHOR + timedelta(days=days - 1)).date()}" in p for p in recording.file_lists[0])


def test_changed_partition_requeries_only_that_partition(tmp_path: Path) -> None:
    root = tmp_path / "candles"
    first = make_candles(rows=2, symbol="RELIANCE", start=ANCHOR)
    second = make_candles(rows=2, symbol="RELIANCE", start=ANCHOR + timedelta(days=1))
    write_part(root, "RELIANCE", first, part="part-a")
    write_part(root, "RELIANCE", second, part="part-b")

    store = DuckDBCandleStore(root)
    recording = RecordingConnection(store._connection)
    store._connection = recording
    try:
        initial = asyncio.run(store.tail("RELIANCE", 3))
        assert len(initial) == 3
        recording.file_lists.clear()
        later = make_candles(rows=1, symbol="RELIANCE", start=ANCHOR + timedelta(days=2))
        added = write_part(root, "RELIANCE", later, part="part-c")
        rows = asyncio.run(store.tail("RELIANCE", 3))
    finally:
        asyncio.run(store.aclose())

    assert rows[-1].timestamp == later[-1].timestamp
    assert recording.file_lists == [[str(added)]]


def test_unsorted_rows_and_same_file_duplicates_use_last_row(tmp_path: Path) -> None:
    root = tmp_path / "candles"
    rows = make_candles(rows=3, symbol="RELIANCE", start=ANCHOR)
    write_part(root, "RELIANCE", [rows[2], rows[0], replace(rows[0], close=999), rows[1]], part="part-a")
    store = DuckDBCandleStore(root)
    try:
        tail = asyncio.run(store.tail("RELIANCE", 3))
    finally:
        asyncio.run(store.aclose())

    assert [row.timestamp for row in tail] == [row.timestamp for row in rows]
    assert tail[0].close == 999


def test_live_cutoff_prunes_old_history_and_discovers_nested_bucket(tmp_path: Path) -> None:
    root = tmp_path / "candles"
    history = tmp_path / "historical"
    live_start = ANCHOR + timedelta(days=7)
    live_rows = make_candles(rows=2, symbol="RELIANCE", start=live_start)
    write_part(root, "RELIANCE", live_rows, part="part-live")

    def write_history(to_date: str, rows: list[object]) -> Path:
        path = history / "symbol=RELIANCE" / "from=2026-08-01" / f"to={to_date}" / "candles.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            [
                {
                    "symbol": row.symbol,
                    "timestamp": int(row.timestamp.timestamp() * 1000),
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
                for row in rows
            ],
            schema=SCHEMA,
        )
        pq.write_table(table, path)
        return path

    old_row = make_candles(rows=1, symbol="RELIANCE", start=ANCHOR + timedelta(days=5))
    equal_cutoff_row = make_candles(rows=1, symbol="RELIANCE", start=live_start + timedelta(minutes=30))
    write_history("2026-08-09", old_row)
    write_history("2026-08-10", equal_cutoff_row)

    store = DuckDBCandleStore(root, historical_root=history)
    recording = RecordingConnection(store._connection)
    store._connection = recording
    try:
        first = asyncio.run(store.tail("RELIANCE", 2))
        assert first[-1].timestamp == equal_cutoff_row[0].timestamp
        assert len(recording.file_lists) == 2, "only live plus the history bucket at cutoff are read"
        assert not any("to=2026-08-09" in name for paths in recording.file_lists for name in paths)

        recording.file_lists.clear()
        new_row = make_candles(rows=1, symbol="RELIANCE", start=live_start + timedelta(hours=12))
        added = write_history("2026-08-11", new_row)
        second = asyncio.run(store.tail("RELIANCE", 2))
    finally:
        asyncio.run(store.aclose())

    assert second[-1].timestamp == new_row[0].timestamp
    assert recording.file_lists == [[str(added)]]


def test_reader_rejects_overload_without_queueing_executor_work(tmp_path: Path) -> None:
    store = DuckDBCandleStore(tmp_path / "candles")
    store._read_slots = threading.BoundedSemaphore(1)
    started = threading.Event()
    release = threading.Event()

    def blocked_tail(symbol: str, limit: int) -> list[object]:
        started.set()
        release.wait(timeout=2)
        return []

    store._tail_sync = blocked_tail  # type: ignore[method-assign]

    async def exercise() -> None:
        first = asyncio.create_task(store.tail("RELIANCE", 1))
        while not started.is_set():
            await asyncio.sleep(0.001)
        try:
            try:
                await store.tail("TCS", 1)
            except RuntimeError as error:
                assert str(error) == "candle reader is at capacity"
            else:
                raise AssertionError("overloaded reader accepted another executor task")
        finally:
            release.set()
        assert await first == []

    try:
        asyncio.run(exercise())
    finally:
        asyncio.run(store.aclose())


@pytest.mark.parametrize("limit", [True, 1.5, "2", 5])
def test_invalid_or_oversize_limit_is_refused_before_worker_submission(tmp_path: Path, limit: object) -> None:
    store = DuckDBCandleStore(tmp_path / "candles", max_rows=4)

    def forbidden_read(*args: object) -> None:
        raise AssertionError("invalid request reached the reader worker")

    store._executor.submit = forbidden_read  # type: ignore[method-assign]
    try:
        with pytest.raises(CandleReadLimitError):
            asyncio.run(store.tail("RELIANCE", limit))  # type: ignore[arg-type]
        assert store._read_slots.acquire(blocking=False)
        store._read_slots.release()
    finally:
        asyncio.run(store.aclose())


@pytest.mark.parametrize("name", [
    "max_rows", "max_cached_rows", "max_cached_partitions",
    "max_cached_directories", "max_cached_directory_entries",
])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_reader_budgets_require_positive_integers(tmp_path: Path, name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        DuckDBCandleStore(tmp_path / "candles", **{name: value})


@pytest.mark.parametrize("row_budget,partition_budget", [(4, 10), (100, 2)])
def test_partition_cache_is_globally_bounded_and_eviction_preserves_tail(
    tmp_path: Path, row_budget: int, partition_budget: int,
) -> None:
    root = tmp_path / "candles"
    symbols = [f"S{index}" for index in range(5)]
    expected = {symbol: make_candles(rows=3, symbol=symbol) for symbol in symbols}
    for symbol in symbols:
        write_part(root, symbol, expected[symbol], part="part-one")
    store = DuckDBCandleStore(root, max_cached_rows=row_budget, max_cached_partitions=partition_budget)
    recording = RecordingConnection(store._connection)
    store._connection = recording
    try:
        for symbol in symbols:
            assert asyncio.run(store.tail(symbol, 3)) == expected[symbol]
            assert store._cached_rows <= row_budget
            assert len(store._cache) <= partition_budget
            assert store._cached_rows == sum(len(entry[3]) for entry in store._cache.values())
        recording.file_lists.clear()
        assert asyncio.run(store.tail(symbols[-1], 2)) == expected[symbols[-1]][-2:]
        assert recording.file_lists == []
        assert asyncio.run(store.tail(symbols[0], 3)) == expected[symbols[0]]
        assert len(recording.file_lists) == 1
    finally:
        asyncio.run(store.aclose())


def test_large_tail_bypasses_cache_without_losing_rows(tmp_path: Path) -> None:
    root = tmp_path / "candles"
    rows = make_candles(rows=6)
    write_part(root, "RELIANCE", rows, part="part-one")
    store = DuckDBCandleStore(root, max_cached_rows=2)
    try:
        assert asyncio.run(store.tail("RELIANCE", 6)) == rows
        assert store._cached_rows == 0 and not store._cache
        assert asyncio.run(store.tail("RELIANCE", 2)) == rows[-2:]
        assert store._cached_rows == 2
    finally:
        asyncio.run(store.aclose())


def test_directory_cache_bounds_entries_and_names_without_changing_discovery(tmp_path: Path) -> None:
    store = DuckDBCandleStore(
        tmp_path / "candles", max_cached_directories=2, max_cached_directory_entries=3,
    )
    paths = [tmp_path / f"symbol-{index}" for index in range(5)]
    try:
        for path in paths:
            for child in ("date=2026-08-03", "date=2026-08-04"):
                (path / child).mkdir(parents=True)
            assert store._child_dirs(path) == ("date=2026-08-03", "date=2026-08-04")
            assert len(store._directory_cache) <= 2
            assert sum(len(entry[2]) for entry in store._directory_cache.values()) <= 3
        assert store._child_dirs(paths[0]) == ("date=2026-08-03", "date=2026-08-04")
        (paths[0] / "date=2026-08-05").mkdir()
        assert len(store._child_dirs(paths[0])) == 3
        (paths[0] / "date=2026-08-06").mkdir()
        assert len(store._child_dirs(paths[0])) == 4
        assert paths[0] not in store._directory_cache
    finally:
        asyncio.run(store.aclose())


def test_bounded_merge_preserves_newest_history_and_live_precedence(tmp_path: Path) -> None:
    root = tmp_path / "candles"
    all_rows = []
    for day in range(8):
        rows = make_candles(rows=3, start=ANCHOR + timedelta(days=day))
        all_rows.extend(rows)
        source = write_part(root, "RELIANCE", rows, part=f"part-{day}")
        bucket = tmp_path / "historical" / "symbol=RELIANCE" / f"from={rows[0].timestamp.date()}" / f"to={rows[-1].timestamp.date()}"
        bucket.mkdir(parents=True)
        source.rename(bucket / "candles.parquet")
    live = replace(all_rows[-2], close=999)
    write_part(root, "RELIANCE", [live], part="part-live")
    store = DuckDBCandleStore(root, max_cached_rows=4, max_cached_partitions=2)
    try:
        expected = [all_rows[-3], live, all_rows[-1]]
        assert asyncio.run(store.tail("RELIANCE", 3)) == expected
        assert asyncio.run(store.tail("RELIANCE", 3)) == expected
        assert store._cached_rows <= 4 and len(store._cache) <= 2
    finally:
        asyncio.run(store.aclose())
