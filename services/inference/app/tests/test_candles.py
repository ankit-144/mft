"""The DuckDB candle store, against real Parquet files on disk.

Everything else in this suite can fake the store. This file cannot: the query
is the part that has to agree with what C2 writes, and the only way to find
out is to write files in C2's layout and read them back with C2's glob. So the
files here are produced with pyarrow, in the hive layout of
`docs/contracts.md` §3, and read by the same expression
`storage.CandleReader.Glob` hands to DuckDB.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.candles import (
    Candle,
    DuckDBCandleStore,
    candles_root,
    escape_hive_value,
    symbol_glob,
)
from support import ANCHOR, make_candles

#: The Parquet row shape of `storage.Candle` (`core/storage/storage.go`):
#: snake_case names, and the timestamp as Unix **milliseconds** in UTC.
SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("timestamp", pa.int64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)


def write_part(
    root: Path, symbol: str, candles: list[Candle], *, part: str, date: str | None = None
) -> Path:
    """Write one published Parquet file in the layout C2 produces.

    Named `part-*.parquet`, never `.inflight-*`: the reader's glob must not be
    able to see a file that is still being written, and a test that wrote
    in-flight names would not be testing the real reader.
    """
    stamp = date or candles[0].timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")
    directory = root / f"symbol={escape_hive_value(symbol)}" / f"date={stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{part}.parquet"

    table = pa.Table.from_pylist(
        [
            {
                "symbol": c.symbol,
                "timestamp": int(c.timestamp.timestamp() * 1000),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ],
        schema=SCHEMA,
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A Parquet tree with two symbols, two dates and two flushes."""
    root = tmp_path / "candles"

    reliance = make_candles(rows=120, symbol="RELIANCE", start=ANCHOR)
    write_part(root, "RELIANCE", reliance[:60], part="part-20260803T075000", date="2026-08-03")
    write_part(root, "RELIANCE", reliance[60:], part="part-20260803T080000", date="2026-08-03")

    # A second date, to prove the glob crosses partitions and orders correctly.
    later = make_candles(rows=30, symbol="RELIANCE", start=ANCHOR + timedelta(days=1))
    write_part(root, "RELIANCE", later, part="part-20260804T075000", date="2026-08-04")

    write_part(root, "TCS", make_candles(rows=80, symbol="TCS", start=ANCHOR), part="part-20260803T075000")
    return root


def read(store: DuckDBCandleStore, symbol: str, limit: int) -> list[Candle]:
    return asyncio.run(store.tail(symbol, limit))


def test_the_tail_is_the_newest_rows_oldest_first(dataset: Path) -> None:
    store = DuckDBCandleStore(dataset)
    try:
        rows = read(store, "RELIANCE", 10)
    finally:
        asyncio.run(store.aclose())

    assert len(rows) == 10
    assert rows == sorted(rows, key=lambda c: c.timestamp), "ascending by time"
    assert rows[-1].timestamp == ANCHOR + timedelta(days=1, minutes=29)
    assert rows[-1].close == pytest.approx(
        make_candles(rows=30, symbol="RELIANCE", start=ANCHOR + timedelta(days=1))[-1].close
    )


def test_a_limit_larger_than_the_store_returns_everything(dataset: Path) -> None:
    store = DuckDBCandleStore(dataset)
    try:
        rows = read(store, "RELIANCE", 10_000)
    finally:
        asyncio.run(store.aclose())

    assert len(rows) == 150, "120 bars on the first date plus 30 on the second"
    assert len({c.timestamp for c in rows}) == 150, "no duplicates across flushes"


def test_a_symbol_with_no_files_is_empty_rather_than_an_error(dataset: Path) -> None:
    """`read_parquet` raises on a glob that matches nothing.

    A symbol the broker has not streamed yet is an ordinary state, and the Go
    reader returns an empty slice for it, so the Python one must too.
    """
    store = DuckDBCandleStore(dataset)
    try:
        assert read(store, "TCS", 5) != []
        assert read(store, "INFY", 5) == []
    finally:
        asyncio.run(store.aclose())


def test_a_non_positive_limit_is_empty(dataset: Path) -> None:
    store = DuckDBCandleStore(dataset)
    try:
        assert read(store, "RELIANCE", 0) == []
        assert read(store, "RELIANCE", -1) == []
    finally:
        asyncio.run(store.aclose())


def test_the_tail_spans_flushes_and_partitions_in_time_order(dataset: Path) -> None:
    """One query, three files, two dates, and the result is one sorted run."""
    store = DuckDBCandleStore(dataset)
    try:
        rows = read(store, "RELIANCE", 150)
    finally:
        asyncio.run(store.aclose())

    assert len(rows) == 150
    assert all(b.timestamp > a.timestamp for a, b in zip(rows, rows[1:])), "no repeat"
    dates = {c.timestamp.date() for c in rows}
    assert dates == {datetime(2026, 8, 3, tzinfo=timezone.utc).date(), datetime(2026, 8, 4, tzinfo=timezone.utc).date()}
    for day in dates:
        same_day = [c for c in rows if c.timestamp.date() == day]
        assert {
            (b.timestamp - a.timestamp).total_seconds() for a, b in zip(same_day, same_day[1:])
        } == {60.0}, "one bar a minute within a session"


def test_a_duplicate_timestamp_resolves_to_the_later_file(tmp_path: Path) -> None:
    """A backfill and a live flush can cover the same minute.

    C2's reader resolves that with "a later file wins, which is what makes an
    idempotent backfill possible". This does the same, so a re-flushed candle
    supersedes the earlier copy instead of appearing twice — which would trip
    the feature builder's strictly-ascending check and skip the symbol forever.
    """
    root = tmp_path / "candles"
    rows = make_candles(rows=80, symbol="RELIANCE", start=ANCHOR)
    write_part(root, "RELIANCE", rows, part="part-20260803T075000")
    write_part(
        root,
        "RELIANCE",
        [replace(rows[40], close=rows[40].close + 5.0)],
        part="part-20260803T090000",
    )

    store = DuckDBCandleStore(root)
    try:
        tail = read(store, "RELIANCE", 80)
    finally:
        asyncio.run(store.aclose())

    assert len(tail) == 80, "the duplicate is collapsed, not doubled"
    assert len({c.timestamp for c in tail}) == 80
    superseded = next(c for c in tail if c.timestamp == rows[40].timestamp)
    assert superseded.close == rows[40].close + 5.0, "the later file wins"


def test_an_inflight_file_is_invisible(dataset: Path) -> None:
    """The reader can never see a half-written file.

    This is C2's immutability invariant, and it is why reading while ingestion
    writes is safe rather than merely usually fine.
    """
    in_flight = (
        dataset / "symbol=RELIANCE" / "date=2026-08-05" / ".inflight-20260805T075000.parquet"
    )
    in_flight.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([], schema=SCHEMA), in_flight)

    store = DuckDBCandleStore(dataset)
    try:
        assert len(read(store, "RELIANCE", 1000)) == 150
    finally:
        asyncio.run(store.aclose())


def test_the_round_trip_preserves_the_domain_candle(dataset: Path) -> None:
    """What comes out is `contracts.Candle`: UTC, millisecond resolution, float64."""
    store = DuckDBCandleStore(dataset)
    try:
        row = read(store, "TCS", 1)[0]
    finally:
        asyncio.run(store.aclose())

    assert isinstance(row, Candle)
    assert row.symbol == "TCS"
    assert row.timestamp.tzinfo is timezone.utc
    assert row.timestamp.microsecond == 0, "candles are truncated to the minute"
    assert isinstance(row.volume, int)
    assert isinstance(row.close, float)


def test_the_glob_is_the_expression_the_go_reader_hands_to_duckdb(dataset: Path) -> None:
    """One path expression, two languages.

    `storage.CandleReader.Glob` exists so the Go reader and the Python reader
    address the same files. If the two derivations of the hive layout ever
    disagree, inference reads a different store than research does.
    """
    assert symbol_glob(dataset, "RELIANCE") == str(
        dataset / "symbol=RELIANCE" / "date=*" / "part-*.parquet"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("RELIANCE", "RELIANCE"),
        ("M&M", "M%26M"),
        ("BAJAJ-AUTO", "BAJAJ-AUTO"),
        ("NIFTY 50", "NIFTY%2050"),
        ("", "%00"),
    ],
)
def test_hive_values_are_escaped_like_the_go_writer(value: str, expected: str) -> None:
    """A symbol containing '/' or '=' would otherwise relocate a dataset."""
    assert escape_hive_value(value) == expected


def test_the_dataset_root_defaults_to_data() -> None:
    assert candles_root("data") == Path("data/candles")
    assert candles_root("") == Path("data/candles")
    assert candles_root("/srv/mft/data") == Path("/srv/mft/data/candles")
