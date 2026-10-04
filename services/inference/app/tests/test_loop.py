"""The minute loop: pull, score, threshold, signal, and every way it can skip.

The tests that matter here are the negative ones. A loop that sends a signal
for every symbol every minute is trivially easy to write and is exactly the
thing that loses money; what has to be proven is that it stays quiet when the
context is short, the model is not loaded, the score is below threshold,
`dry_run` is on, or the execution service is throwing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.config import InferenceConfig
from app.features import WARMUP_ROWS
from app.loop import Decision, MinuteScheduler, Outcome
from app.predictor import Predictor
from support import (
    ANCHOR,
    ConstantModel,
    FakeSender,
    FakeStore,
    golden_candles,
    make_candles,
)


def loaded_predictor(score: float = 0.9) -> Predictor:
    """A predictor wrapping a model that always answers `score`.

    A fixed score is what makes a threshold test a threshold test: a real model
    returns a different number on every window, so "below threshold sends
    nothing" would otherwise pass or fail by luck.
    """
    predictor = Predictor(ConstantModel(score), model_name="constant")
    asyncio.run(predictor.load())
    return predictor


def build_scheduler(
    config: InferenceConfig,
    store: FakeStore,
    sender: FakeSender,
    *,
    predictor: Predictor | None = None,
    clock: Any = None,
    max_context_age: timedelta | None = None,
) -> MinuteScheduler:
    """A scheduler over fakes, with the staleness guard off unless asked for.

    The guard is off by default in these tests because the fixture windows are
    anchored in the past; `test_a_stale_context_is_skipped` turns it on.
    """
    return MinuteScheduler(
        config,
        store,
        predictor if predictor is not None else loaded_predictor(),
        sender,
        max_context_age=max_context_age,
        clock=clock,
    )


def live_config(**overrides: Any) -> InferenceConfig:
    """A config that will actually post, against a `FakeSender`."""
    base = {
        "model": "heuristic",
        "execution_url": "http://execution.invalid:8080",
        "context_rows": 100,
        "horizon_bars": 1,
        "score_threshold": 0.5,
        "order_quantity": 10,
        "instruments": ["RELIANCE"],
        "dry_run": False,
    }
    base.update(overrides)
    return InferenceConfig(**base)


async def teardown(predictor: Predictor) -> None:
    await predictor.aclose()


# --- the happy path ------------------------------------------------------


def test_a_passing_score_sends_a_signal_with_the_right_fields() -> None:
    candles = make_candles(rows=200)
    store = FakeStore({"RELIANCE": candles})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.SENT
    assert sender.count == 1
    signal = sender.sent[0]
    assert signal.symbol == "RELIANCE"
    assert signal.side == "BUY"
    assert signal.quantity == 10, "quantity is inference.order_quantity"
    # The price is the close of the last candle, not the predicted value.
    assert signal.price == pytest.approx(candles[-1].close, abs=0.01)
    assert signal.score == pytest.approx(0.9)
    assert signal.model == "constant-v1", "the signal names the loaded weights"
    assert signal.as_of == candles[-1].timestamp
    assert signal.idempotency_key == f"RELIANCE:BUY:{_key(candles[-1].timestamp)}"
    assert decision.order_id == "4412"


def test_a_negative_score_sells() -> None:
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender, predictor=loaded_predictor(-0.8))

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.SENT
    assert sender.sent[0].side == "SELL"
    assert sender.sent[0].idempotency_key.startswith("RELIANCE:SELL:")


def test_the_loop_pulls_a_warmed_window_not_the_whole_store() -> None:
    """`context_rows` is rows for the model, so more candles are pulled than that.

    Pulling exactly `context_rows` would spend 60 of the 100 prompt rows on
    warm-up zeros, which a zero-shot model reads as a market that did not move.
    """
    store = FakeStore({"RELIANCE": make_candles(rows=4000)})
    scheduler = build_scheduler(live_config(), store, FakeSender())

    asyncio.run(scheduler.tick())

    assert store.reads == [("RELIANCE", 100 + WARMUP_ROWS)]


def test_every_instrument_is_scored_in_one_tick() -> None:
    symbols = ["RELIANCE", "TCS", "INFY"]
    store = FakeStore({s: make_candles(rows=200, symbol=s) for s in symbols})
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(instruments=symbols), store, sender, predictor=loaded_predictor(0.7)
    )

    decisions = asyncio.run(scheduler.tick())

    assert [d.symbol for d in decisions] == symbols
    assert all(d.outcome is Outcome.SENT for d in decisions)
    assert {s.symbol for s in sender.sent} == set(symbols)


def test_an_idempotency_key_is_stable_on_retry_and_moves_next_minute() -> None:
    """The load-bearing property, driven through two real ticks.

    Two ticks over the same store replay the same minute: same key, and a
    second POST that C7 would answer with a 409 and the original order id. One
    extra candle is then published, the way ingestion would, and the next tick
    must produce a different key.
    """
    candles = make_candles(rows=200)
    store = FakeStore({"RELIANCE": candles})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender)

    first = asyncio.run(scheduler.tick())[0]
    replay = asyncio.run(scheduler.tick())[0]
    assert first.signal is not None
    assert replay.outcome is Outcome.UNCHANGED
    assert replay.signal is None
    assert first.as_of == replay.as_of

    store.add("RELIANCE", candles + make_candles(rows=1, start=ANCHOR + timedelta(minutes=200)))
    following = asyncio.run(scheduler.tick())[0]
    assert following.as_of != first.as_of
    assert following.signal is not None
    assert following.signal.idempotency_key != first.signal.idempotency_key

    keys = [s.idempotency_key for s in sender.sent]
    assert len(keys) == 2
    assert keys[0] != keys[1]


# --- the threshold -------------------------------------------------------


def test_a_score_below_the_threshold_sends_nothing() -> None:
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(score_threshold=0.75), store, sender, predictor=loaded_predictor(0.6)
    )

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.BELOW_THRESHOLD
    assert decision.score == pytest.approx(0.6)
    assert sender.count == 0
    assert "threshold" in decision.detail


def test_the_threshold_is_compared_against_the_magnitude() -> None:
    """A strong negative conviction is a SELL, not a quiet minute."""
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(score_threshold=0.75), store, sender, predictor=loaded_predictor(-0.9)
    )

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.SENT
    assert sender.sent[0].side == "SELL"


def test_the_threshold_is_inclusive() -> None:
    """`abs(score) >= threshold` trades; the boundary itself is a decision."""
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(score_threshold=0.6), store, sender, predictor=loaded_predictor(0.6)
    )

    assert asyncio.run(scheduler.tick())[0].outcome is Outcome.SENT


# --- dry run -------------------------------------------------------------


def test_dry_run_sends_nothing_at_all(caplog) -> None:
    """The default configuration places no order, and says what it would do."""
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(dry_run=True), store, sender)

    with caplog.at_level("INFO", logger="mft.inference.loop"):
        decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.DRY_RUN
    assert decision.signal is not None, "the signal is still built, so it can be diffed"
    assert sender.count == 0, "dry_run must not reach the execution client"
    assert "would POST" in caplog.text
    assert decision.signal.idempotency_key in caplog.text


def test_dry_run_is_the_default() -> None:
    """`InferenceConfig()` with no arguments must not trade.

    AGENTS.md and `docs/contracts.md` §8 both say the default is `true`, and a
    service that starts with an absent or empty config must not be the one
    exception.
    """
    assert InferenceConfig().dry_run is True
    assert InferenceConfig(model="heuristic", instruments=["RELIANCE"]).dry_run is True


# --- fail closed ---------------------------------------------------------


def test_a_symbol_with_no_candles_is_skipped() -> None:
    store = FakeStore({})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.NO_CONTEXT
    assert sender.count == 0


def test_a_short_window_is_skipped_before_the_model_ever_sees_it() -> None:
    """40 candles cannot produce a warmed row, let alone a prompt table."""
    store = FakeStore({"RELIANCE": make_candles(rows=40)})
    sender = FakeSender()
    model = ConstantModel(0.9)
    predictor = Predictor(model, model_name="constant")
    asyncio.run(predictor.load())
    scheduler = build_scheduler(live_config(), store, sender, predictor=predictor)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.BAD_CONTEXT
    assert "warm-up" in decision.detail
    assert sender.count == 0
    assert model.calls == 0, "no score may be computed on a bad context"
    asyncio.run(teardown(predictor))


def test_a_window_that_is_warmed_but_too_short_for_the_model_is_skipped() -> None:
    """65 candles warm up, leaving 5 rows: below the model's floor of 60.

    A model handed fewer rows than its floor answers with whatever it can still
    attend to, which is a confident number computed from too little. Skipping
    is the only honest outcome.
    """
    store = FakeStore({"RELIANCE": make_candles(rows=WARMUP_ROWS + 5)})
    sender = FakeSender()
    predictor = loaded_predictor()
    scheduler = build_scheduler(
        live_config(context_rows=100, horizon_bars=1), store, sender, predictor=predictor
    )

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.BAD_CONTEXT
    assert "short of" in decision.detail
    assert sender.count == 0
    asyncio.run(teardown(predictor))


def test_a_window_the_builder_rejects_is_skipped() -> None:
    """Corrupt input — here a rewound timestamp — produces no signal."""
    candles = make_candles(rows=200)
    candles[100] = candles[99]
    store = FakeStore({"RELIANCE": candles})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.BAD_CONTEXT
    assert "ascend strictly" in decision.detail
    assert sender.count == 0


def test_a_model_that_is_not_loaded_is_skipped() -> None:
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    predictor = Predictor(ConstantModel(0.9, loaded=False), model_name="constant")
    scheduler = build_scheduler(live_config(), store, sender, predictor=predictor)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.NOT_LOADED
    assert sender.count == 0


def test_a_throwing_execution_post_skips_instead_of_crashing() -> None:
    """An execution service that is down must not take the loop with it."""
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender(raise_with=RuntimeError("connection refused"))
    scheduler = build_scheduler(live_config(), store, sender)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.SEND_FAILED
    assert decision.signal is not None, "the attempted signal is kept for the record"
    assert decision.signal.idempotency_key
    assert sender.count == 1


def test_a_model_that_raises_is_skipped() -> None:
    class Exploding:
        name = "exploding-v1"
        calls = 0

        def is_loaded(self) -> bool:
            return True

        async def load(self) -> None:
            return None

        def predict(self, context: Any, horizon: int) -> float:
            raise RuntimeError("the backend died")

    predictor = Predictor(Exploding(), model_name="exploding")
    asyncio.run(predictor.load())
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender, predictor=predictor)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.ERROR
    assert sender.count == 0
    asyncio.run(teardown(predictor))


def test_a_failing_store_read_is_skipped() -> None:
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    store.fail_with = OSError("no such file")
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.ERROR
    assert "candle read failed" in decision.detail
    assert sender.count == 0


def test_a_stale_context_is_skipped() -> None:
    """A store that has not advanced while the loop runs is a dead feed.

    Scoring it would trade a number computed from a window that ended minutes
    ago, and the reference price in the signal would be just as old.
    """
    candles = make_candles(rows=200, start=datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc))
    now = candles[-1].timestamp + timedelta(minutes=30)
    store = FakeStore({"RELIANCE": candles})
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(),
        store,
        sender,
        clock=lambda: now,
        max_context_age=timedelta(minutes=5),
    )

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.STALE_CONTEXT
    assert sender.count == 0
    assert "has not advanced" in decision.detail


def test_one_bad_symbol_does_not_stop_the_others() -> None:
    """The whole point of per-symbol containment.

    A single instrument whose feed has a gap, a duplicate bar or no data at
    all must not silence the rest of the session.
    """
    symbols = ["RELIANCE", "BROKEN", "TCS"]
    store = FakeStore(
        {
            "RELIANCE": make_candles(rows=200, symbol="RELIANCE"),
            "BROKEN": make_candles(rows=10, symbol="BROKEN"),
            "TCS": make_candles(rows=200, symbol="TCS"),
        }
    )
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(instruments=symbols), store, sender, predictor=loaded_predictor(0.9)
    )

    decisions = asyncio.run(scheduler.tick())
    outcomes = {d.symbol: d.outcome for d in decisions}

    assert outcomes == {
        "RELIANCE": Outcome.SENT,
        "BROKEN": Outcome.BAD_CONTEXT,
        "TCS": Outcome.SENT,
    }
    assert {s.symbol for s in sender.sent} == {"RELIANCE", "TCS"}


def test_an_unexpected_error_in_one_symbol_is_contained() -> None:
    """Even a bug in this component must not silence the other instruments."""

    class Exploding:
        name = "exploding-v1"

        def is_loaded(self) -> bool:
            return True

        async def load(self) -> None:
            return None

        def predict(self, context: Any, horizon: int) -> float:
            raise KeyboardInterrupt if False else RuntimeError("boom")

    predictor = Predictor(Exploding(), model_name="exploding")
    asyncio.run(predictor.load())
    store = FakeStore(
        {
            "RELIANCE": make_candles(rows=200, symbol="RELIANCE"),
            "TCS": make_candles(rows=200, symbol="TCS"),
        }
    )
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(instruments=["RELIANCE", "TCS"]), store, sender, predictor=predictor
    )

    decisions = asyncio.run(scheduler.tick())

    assert [d.outcome for d in decisions] == [Outcome.ERROR, Outcome.ERROR]
    assert sender.count == 0
    asyncio.run(teardown(predictor))


def test_a_horizon_longer_than_one_is_still_checked() -> None:
    """The usable prompt is `horizon` rows shorter than the table."""
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    predictor = loaded_predictor(0.9)
    scheduler = build_scheduler(
        live_config(horizon_bars=2), store, sender, predictor=predictor
    )

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.SENT
    asyncio.run(teardown(predictor))


# --- the schedule --------------------------------------------------------


def test_ticks_land_on_the_minute_boundary() -> None:
    """The next sleep is to the boundary, so ticks do not drift.

    A fixed 60-second period from process start would walk across the minute,
    so some minutes would see a half-closed candle and some would see the same
    candle twice.
    """
    scheduler = build_scheduler(
        live_config(), FakeStore(), FakeSender(), predictor=loaded_predictor()
    )
    at = datetime(2026, 9, 29, 10, 30, 0, tzinfo=timezone.utc)

    assert scheduler._seconds_to_next_boundary(at) == pytest.approx(60.0)
    assert scheduler._seconds_to_next_boundary(
        datetime(2026, 9, 29, 10, 30, 4, tzinfo=timezone.utc)
    ) == pytest.approx(56.0)
    # A pass that overran its window must not spin against the boundary it
    # just missed, so the sleep never falls to zero.
    assert scheduler._seconds_to_next_boundary(
        datetime(2026, 9, 29, 10, 30, 59, 900000, tzinfo=timezone.utc)
    ) >= 0.5


def test_the_loop_stops_when_asked() -> None:
    """`run` returns when the stop event is set, without waiting a whole minute."""
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    scheduler = build_scheduler(live_config(), store, FakeSender())

    async def drive() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(scheduler.run(stop))
        await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(drive())


def test_tick_counts_and_reports_the_last_pass() -> None:
    now = datetime(2026, 9, 29, 10, 31, tzinfo=timezone.utc)
    scheduler = build_scheduler(
        live_config(), FakeStore(), FakeSender(), predictor=loaded_predictor(), clock=lambda: now
    )

    assert scheduler.ticks == 0
    assert scheduler.last_tick is None
    asyncio.run(scheduler.tick())
    assert scheduler.ticks == 1
    assert scheduler.last_tick == now


def test_the_context_a_model_receives_is_the_frozen_schema() -> None:
    """The prompt is C5's `FEATURE_COLUMNS`, in order, with no NaN.

    A context that is missing a column or carries a NaN is rejected by
    `validate_context`, so this is the assertion that the loop hands over
    something usable rather than something that happens to work today.
    """
    from model import FEATURE_COLUMNS

    seen: list[Any] = []

    class Recording:
        name = "recording-v1"

        def is_loaded(self) -> bool:
            return True

        async def load(self) -> None:
            return None

        def predict(self, context: Any, horizon: int) -> float:
            seen.append(context)
            return 0.9

    predictor = Predictor(Recording(), model_name="recording")
    asyncio.run(predictor.load())
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    scheduler = build_scheduler(live_config(), store, FakeSender(), predictor=predictor)

    asyncio.run(scheduler.tick())

    assert len(seen) == 1
    context = seen[0]
    assert list(context.columns) == list(FEATURE_COLUMNS)
    assert len(context) == 100
    assert not context.isna().to_numpy().any()
    asyncio.run(teardown(predictor))


def test_decision_reads_as_a_line() -> None:
    decision = Decision(symbol="RELIANCE", outcome=Outcome.SENT, score=0.9)
    assert str(decision) == "RELIANCE sent score=+0.9000"
    assert Decision(symbol="X", outcome=Outcome.NO_CONTEXT).__str__() == (
        "X no_context score=n/a"
    )


def test_golden_candles_flow_through_the_whole_loop() -> None:
    """The parity fixture, extended past its warm-up, signals like any window.

    Worth a test of its own: the golden test proves the feature math against
    Go, and this proves the same bars flow through pull, build, score and
    signal without anything about them being special.
    """
    from datetime import timedelta

    prefix = golden_candles()
    extra = make_candles(
        rows=100, start=prefix[-1].timestamp + timedelta(minutes=1)
    )
    window = prefix + extra
    store = FakeStore({"RELIANCE": window})
    sender = FakeSender()
    scheduler = build_scheduler(
        live_config(context_rows=100), store, sender, predictor=loaded_predictor(0.9)
    )

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.SENT
    assert decision.signal is not None
    assert decision.signal.price == pytest.approx(window[-1].close, abs=0.01)
    assert decision.signal.as_of == window[-1].timestamp
    # 164 candles warm up to 104, and the prompt is capped at 100.
    assert decision.candidates == 100


def test_a_window_that_warms_up_but_stays_short_is_still_refused() -> None:
    """The golden fixture alone: 64 bars, 4 of them warmed, and a floor of 61.

    Warm enough to build, far too short to score. It is skipped rather than
    padded, because a model handed five rows answers confidently from them.
    """
    store = FakeStore({"RELIANCE": golden_candles()})
    sender = FakeSender()
    scheduler = build_scheduler(live_config(), store, sender)

    decision = asyncio.run(scheduler.tick())[0]

    assert decision.outcome is Outcome.BAD_CONTEXT
    assert sender.count == 0


def _key(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")
