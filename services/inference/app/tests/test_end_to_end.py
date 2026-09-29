"""The whole path, wired together: Parquet, DuckDB, features, model, signal.

Every other test in this directory fakes one of those. This one fakes only the
far end — the execution service, which is a list — and puts a real Parquet
tree under the real `CandleReader.Glob` expression, reads it with the real
DuckDB query, builds the real feature table, scores it with the real heuristic
model and posts the result.

It is the test that would fail if the loop worked only against fakes: a glob
that resolves to nothing, a column name that does not match, a window the
builder rejects, a prompt the model refuses. The model is `heuristic` and the
sender is `FakeSender`, because the checkpoint is ~6.6 GB and a real order is
not something a test may place.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.candles import DuckDBCandleStore
from app.config import Config, InferenceConfig, StorageConfig
from app.loop import MinuteScheduler, Outcome
from app.main import create_app
from app.predictor import Predictor
from app.runtime import Runtime
from support import ANCHOR, FakeSender, make_candles
from test_candles import write_part


def config_for(data_dir: Path, **overrides: object) -> Config:
    settings: dict[str, object] = {
        "model": "heuristic",
        "execution_url": "http://execution.invalid:8080",
        "context_rows": 100,
        "horizon_bars": 1,
        "score_threshold": 0.05,
        "order_quantity": 7,
        "instruments": ["RELIANCE"],
        "dry_run": False,
    }
    settings.update(overrides)
    return Config(
        storage=StorageConfig(data_dir=str(data_dir), flush_interval_seconds=300),
        inference=InferenceConfig(**settings),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A real Parquet tree with enough history to score."""
    root = tmp_path / "data" / "candles"
    candles = make_candles(rows=300, symbol="RELIANCE", start=ANCHOR, drift=0.0004)
    for part, chunk in enumerate((candles[:150], candles[150:])):
        write_part(
            root,
            "RELIANCE",
            chunk,
            part=f"part-20260803T0{part}0000",
            date="2026-08-03",
        )
    return root


def test_a_minute_close_produces_a_signal_from_a_real_store(dataset: Path) -> None:
    """The north star, minus the parts a test may not touch.

    Ingestion writes a flushed Parquet file, the loop reads it back through
    DuckDB, the features are built, the heuristic scores them, and a signal is
    posted to a list. The threshold is 0.05 because the heuristic's output
    depends on the window and a 0.55 default would make this a test of luck
    rather than of wiring; the threshold's own behaviour is pinned in
    `test_loop.py`.
    """
    store = DuckDBCandleStore(dataset)
    sender = FakeSender()
    predictor = Predictor.from_name("heuristic")

    async def drive() -> None:
        await predictor.load()
        try:
            scheduler = MinuteScheduler(
                config_for(dataset.parent.parent).inference,
                store,
                predictor,
                sender,
                timezone_name="Asia/Kolkata",
                max_context_age=None,
            )
            decisions = await scheduler.tick()
            assert len(decisions) == 1
            decision = decisions[0]
            assert decision.outcome is Outcome.SENT, decision.detail
            assert decision.signal is not None
            assert decision.signal.symbol == "RELIANCE"
            assert decision.signal.quantity == 7
            assert decision.signal.model == "heuristic-v1"
            assert decision.signal.as_of == ANCHOR + timedelta(minutes=299)
            assert decision.signal.price > 0.0
            assert decision.signal.idempotency_key.startswith("RELIANCE:")
            assert len(sender.sent) == 1
            assert sender.sent[0].idempotency_key == decision.signal.idempotency_key
        finally:
            await predictor.aclose()
            await store.aclose()

    asyncio.run(drive())


def test_the_scheduler_fires_repeatedly_on_its_period(dataset: Path) -> None:
    """`run` really does loop, rather than being correct exactly once.

    The period is 40 ms rather than a minute so the test can watch several
    boundaries go by. Everything else is the production wiring, including the
    store and the model, so a boundary that fires twice in a minute is
    observable here as it would be in a session.
    """
    store = DuckDBCandleStore(dataset)
    sender = FakeSender()
    predictor = Predictor.from_name("heuristic")
    config = config_for(dataset.parent.parent, score_threshold=0.99).inference

    async def drive() -> int:
        await predictor.load()
        try:
            scheduler = MinuteScheduler(
                config,
                store,
                predictor,
                sender,
                timezone_name="Asia/Kolkata",
                max_context_age=None,
                tick_period=timedelta(milliseconds=40),
            )
            stop = asyncio.Event()
            task = asyncio.create_task(scheduler.run(stop))
            await asyncio.sleep(0.25)
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)
            return scheduler.ticks
        finally:
            await predictor.aclose()
            await store.aclose()

    ticks = asyncio.run(drive())

    assert ticks >= 2, "the loop must keep going after one pass"
    assert sender.count == 0, "a 0.99 threshold is not a decision this window makes"


def test_the_endpoints_and_the_loop_read_the_same_store(dataset: Path) -> None:
    """One store, two doors: the context endpoint agrees with what the loop pulls.

    This is the failure a debugging endpoint exists to prevent — a `/v1/context`
    that shows something other than what the model was given is worse than no
    endpoint at all.
    """
    store = DuckDBCandleStore(dataset)
    sender = FakeSender()
    predictor = Predictor.from_name("heuristic")
    runtime = Runtime(
        config=config_for(dataset.parent.parent),
        store=store,
        predictor=predictor,
        sender=sender,
        scheduler=MinuteScheduler(
            config_for(dataset.parent.parent).inference,
            store,
            predictor,
            sender,
            timezone_name="Asia/Kolkata",
            max_context_age=None,
        ),
    )

    async def drive() -> tuple[dict, dict, list]:
        try:
            with TestClient(create_app(runtime, run_scheduler=False)) as client:
                context = client.get(
                    "/v1/context", params={"symbol": "RELIANCE", "rows": 100}
                ).json()
                prediction = client.post(
                    "/v1/predict", json={"symbol": "RELIANCE"}
                ).json()
                decisions = await runtime.scheduler.tick()
            return context, prediction, decisions
        finally:
            await predictor.aclose()
            await store.aclose()

    context, prediction, decisions = asyncio.run(drive())

    from app.signals import rfc3339

    assert prediction["as_of"] == context["as_of"]
    assert rfc3339(decisions[0].as_of) == context["as_of"]
    assert len(context["rows"]) == 100
    assert len(decisions) == 1
    # The signal the loop built refers to the same candle the endpoints report.
    assert decisions[0].signal is not None
    assert decisions[0].signal.as_of.strftime("%Y-%m-%dT%H:%M:%SZ") == context["as_of"]


def test_a_store_that_advances_between_minutes_moves_the_key(dataset: Path) -> None:
    """Ingestion publishing a new flush is what advances the loop's minute.

    The same two reads, the same model, the same window length — and a
    different idempotency key, because the minute close moved. This is the
    pull architecture doing its job: no message was sent to the inference
    service, it simply read again and saw more.
    """
    store = DuckDBCandleStore(dataset)
    sender = FakeSender()
    predictor = Predictor.from_name("heuristic")
    config = config_for(dataset.parent.parent).inference

    async def drive() -> list[str]:
        await predictor.load()
        try:
            scheduler = MinuteScheduler(
                config, store, predictor, sender, timezone_name="Asia/Kolkata",
                max_context_age=None,
            )
            first = await scheduler.tick()
            extension = make_candles(
                rows=1, symbol="RELIANCE", start=ANCHOR + timedelta(minutes=300)
            )
            write_part(
                dataset, "RELIANCE", extension, part="part-20260803T030000", date="2026-08-03"
            )
            second = await scheduler.tick()
            return [d.signal.idempotency_key for d in (first[0], second[0]) if d.signal]
        finally:
            await predictor.aclose()
            await store.aclose()

    keys = asyncio.run(drive())

    assert len(keys) == 2
    assert keys[0] != keys[1]
