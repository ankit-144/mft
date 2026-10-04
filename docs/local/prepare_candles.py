#!/usr/bin/env python3
"""Prepare bounded demo candles for the existing Parquet reader; never submit orders."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services/inference"))

from app.candles import Candle, escape_hive_value
from app.synth import SynthSpec, generate


def utc_timestamp(value: str) -> datetime:
    """Accept Unix milliseconds or an ISO timestamp with an explicit UTC offset."""
    if value.isdigit():
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("timestamp needs a timezone, for example 2026-01-05T03:45:00Z")
    return stamp.astimezone(timezone.utc)


def public_candles(symbol: str, bars: int, start: str | None) -> list[Candle]:
    """Fetch one public 1-minute batch; retain closed bars and scale fractional volume."""
    if bars > 1000:
        raise ValueError("Binance demo permits at most 1000 bars per request")
    params = {"symbol": symbol, "interval": "1m", "limit": bars}
    if start:
        params["startTime"] = int(utc_timestamp(start).timestamp() * 1000)
    url = "https://data-api.binance.vision/api/v3/klines?" + urlencode(params)
    with urlopen(url, timeout=20) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("public response exceeds the 1 MB demo limit")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError(f"expected a kline array; received {str(payload)[:200]}")
    if len(payload) > bars:
        raise ValueError("public response exceeds the requested bar count")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    candles = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError("invalid Binance kline row")
        if int(row[6]) >= now_ms:
            continue
        volume = int((Decimal(row[5]) * 1_000_000).to_integral_value(rounding=ROUND_HALF_UP))
        candles.append(Candle(
            symbol, datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
            float(row[1]), float(row[2]), float(row[3]), float(row[4]), volume,
        ))
    return candles


def csv_candles(path: Path, symbol: str, bars: int) -> list[Candle]:
    """Read at most the requested number of chronological, integer-volume CSV bars."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    candles = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("CSV requires timestamp,open,high,low,close,volume headers")
        for row in reader:
            if row.get("symbol") and row["symbol"].strip().upper() != symbol:
                raise ValueError("CSV must contain only the requested symbol")
            candles.append(Candle(
                symbol, utc_timestamp(row["timestamp"]),
                float(row["open"]), float(row["high"]), float(row["low"]),
                float(row["close"]), int(row["volume"]),
            ))
            if len(candles) == bars:
                break
    return candles


def write_candles(candles: list[Candle], data_dir: Path) -> int:
    """Validate closed minute bars and atomically publish canonical Parquet partitions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not candles:
        raise ValueError("no closed candles to write")
    days: dict[str, list[dict[str, object]]] = {}
    previous = -1
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for candle in candles:
        timestamp = int(candle.timestamp.timestamp() * 1000)
        prices = (candle.open, candle.high, candle.low, candle.close)
        if (timestamp <= previous or candle.timestamp.second or candle.timestamp.microsecond
                or timestamp % 60_000 or timestamp + 60_000 > now_ms):
            raise ValueError("timestamps must increase, align to minutes and represent closed bars")
        if any(not math.isfinite(price) or price <= 0 for price in prices):
            raise ValueError("OHLC values must be finite and positive")
        if candle.low > min(candle.open, candle.close) or candle.high < max(candle.open, candle.close):
            raise ValueError("low/high must contain open/close")
        if not 0 <= candle.volume < 2**63:
            raise ValueError("volume must fit a nonnegative int64")
        previous = timestamp
        day = candle.timestamp.astimezone(timezone.utc).date().isoformat()
        days.setdefault(day, []).append(dict(
            symbol=candle.symbol, timestamp=timestamp, open=candle.open, high=candle.high,
            low=candle.low, close=candle.close, volume=candle.volume,
        ))
    schema = pa.schema([
        ("symbol", pa.string()), ("timestamp", pa.int64()),
        ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()),
        ("close", pa.float64()), ("volume", pa.int64()),
    ])
    for day, rows in days.items():
        folder = data_dir / "candles" / f"symbol={escape_hive_value(candles[0].symbol)}" / f"date={day}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"part-demo-{uuid4().hex}.parquet"
        temporary = target.with_suffix(".tmp")
        try:
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return len(days)


def main() -> int:
    """Select an offline or public source and write an isolated demo store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "binance", "csv"), default="synthetic")
    parser.add_argument("--symbol", default="DEMO")
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--start", help="Binance start time, with timezone or Unix milliseconds")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/local-demo")
    args = parser.parse_args()
    try:
        if not 1 <= args.bars <= 10_000:
            raise ValueError("--bars must be between 1 and 10000")
        symbol = args.symbol.strip().upper()
        if not symbol:
            raise ValueError("--symbol cannot be empty")
        if args.source == "synthetic":
            candles = generate(symbol, SynthSpec(bars=args.bars, seed=args.seed))
        elif args.source == "binance":
            candles = public_candles(symbol, args.bars, args.start)
        else:
            if args.csv is None:
                raise ValueError("--csv is required for the CSV source")
            candles = csv_candles(args.csv, symbol, args.bars)
        directory = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
        partitions = write_candles(candles, directory)
        print(json.dumps({
            "source": args.source, "symbol": symbol, "rows": len(candles),
            "partitions": partitions, "data_dir": str(directory),
            "first": candles[0].timestamp.isoformat(), "last": candles[-1].timestamp.isoformat(),
            "volume_units": "base asset x 1000000, rounded" if args.source == "binance" else "integer units",
        }, indent=2))
        return 0
    except (OSError, ValueError, ArithmeticError, KeyError, TypeError, URLError) as error:
        print(f"prepare_candles: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
