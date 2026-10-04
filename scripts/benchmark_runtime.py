#!/usr/bin/env python3
"""Measure bounded scheduling using production components and synthetic I/O."""

from __future__ import annotations

import asyncio
import errno
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "services/inference/.venv/bin/python"
if os.environ.get("MFT_BENCH_ENV") != "1":
    os.environ.update(MFT_BENCH_ENV="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.execv(str(PYTHON), [str(PYTHON), __file__])
sys.path[:0] = [str(ROOT / "services/inference"), str(ROOT / "services/inference/app/tests")]

from app.config import InferenceConfig
from app.features import FeatureBuilder
from app.loop import MinuteScheduler, Outcome
from app.predictor import Predictor
from support import ConstantModel, FakeSender, FakeStore, make_candles


def restricted_socket_writes() -> bool:
    left, right = socket.socketpair()
    try:
        left.send(b"x")
        return False
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EACCES):
            raise
        return True
    finally:
        left.close()
        right.close()


async def measure(concurrency: int) -> dict:
    class Store(FakeStore):
        async def tail(self, symbol, limit):
            await asyncio.sleep(0.01)
            return await super().tail(symbol, limit)

    symbols = [f"S{i}" for i in range(8)]
    candles = {symbol: make_candles(rows=200, symbol=symbol) for symbol in symbols}
    predictor = Predictor(ConstantModel(0.8), model_name="constant")
    config = InferenceConfig(model="heuristic", instruments=symbols, max_concurrency=concurrency)
    durations = []
    try:
        await predictor.load()
        for _ in range(7):
            scheduler = MinuteScheduler(config, Store(candles), predictor, FakeSender(), max_context_age=None)
            before = time.perf_counter()
            decisions = await scheduler.tick()
            durations.append((time.perf_counter() - before) * 1000)
            if any(row.outcome is not Outcome.DRY_RUN for row in decisions):
                raise RuntimeError("benchmark failed its scheduler acceptance check")
        return {"concurrency": concurrency, "symbols": len(symbols), "injected_read_latency_ms": 10,
                "samples_ms": durations, "median_ms": statistics.median(durations)}
    finally:
        await predictor.aclose()


async def main(restricted: bool):
    candles = make_candles(rows=160)
    builder = FeatureBuilder()
    before = time.perf_counter()
    for _ in range(100):
        builder.build(candles)
    features_ms = (time.perf_counter() - before) * 10
    results = {"schema": "mft.benchmark.v2", "synthetic": True,
               "limitations": "Eight symbols; injected I/O; constant model; no real feed/model capacity measured.",
               "sandbox_selector_heartbeat_ms": 5 if restricted else 0,
               "feature_build_160_rows_mean_ms": features_ms,
               "scheduling": [await measure(1), await measure(4)]}
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    restricted = restricted_socket_writes()
    with asyncio.Runner() as runner:
        if restricted:
            loop = runner.get_loop()
            def heartbeat():
                if loop.is_running():
                    loop.call_later(0.005, heartbeat)
            loop.call_later(0.005, heartbeat)
        runner.run(main(restricted))
