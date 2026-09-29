"""The predictor: one model, one thread, one place that can fall back."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from app.predictor import Predictor
from model import ModelNotLoadedError, ModelUnavailableError


class Recording:
    """An `InferenceModel` that records which thread called it."""

    def __init__(self, *, score: float = 0.5) -> None:
        self.score = score
        self.loaded = False
        self.threads: list[str] = []
        self.max_concurrent = 0
        self._inside = 0

    @property
    def name(self) -> str:
        return "recording-v1"

    def is_loaded(self) -> bool:
        return self.loaded

    async def load(self) -> None:
        self.loaded = True

    def predict(self, context: Any, horizon: int) -> float:
        self.threads.append(threading.current_thread().name)
        self._inside += 1
        self.max_concurrent = max(self.max_concurrent, self._inside)
        try:
            # Long enough that a second caller would overlap if the pool had
            # more than one worker.
            threading.Event().wait(0.01)
            return self.score
        finally:
            self._inside -= 1


def test_a_prediction_runs_off_the_event_loop() -> None:
    """A blocking forward pass must not stall the loop that also serves HTTP."""
    model = Recording()
    predictor = Predictor(model, model_name="recording")
    asyncio.run(predictor.load())

    async def drive() -> tuple[float, str]:
        loop_thread = threading.current_thread().name
        score = await predictor.predict(None, 1)
        return score, loop_thread

    score, loop_thread = asyncio.run(drive())

    assert score == 0.5
    assert model.threads == ["predict_0"], "on the dedicated worker, not the loop"
    assert model.threads[0] != loop_thread
    asyncio.run(predictor.aclose())


def test_two_callers_never_enter_the_model_at_once() -> None:
    """TabFM's prompt table is mutable; two threads inside it is undefined.

    The HTTP endpoint and the scheduler reach the same model. A pool with more
    than one worker would be the obvious way to get throughput and the wrong
    way to get a corrupted prompt.
    """
    model = Recording()
    predictor = Predictor(model, model_name="recording")
    asyncio.run(predictor.load())

    async def drive() -> list[float]:
        return await asyncio.gather(
            *(predictor.predict(None, 1) for _ in range(6))
        )

    scores = asyncio.run(drive())

    assert scores == [0.5] * 6
    assert model.max_concurrent == 1
    assert len(set(model.threads)) == 1
    asyncio.run(predictor.aclose())


def test_load_is_awaited_rather_than_fired_and_forgotten() -> None:
    """`load` is a coroutine, so it belongs on the loop, not the worker."""
    predictor = Predictor(Recording(), model_name="recording")
    asyncio.run(predictor.load())
    assert predictor.is_loaded()
    asyncio.run(predictor.aclose())


def test_an_unavailable_model_falls_back_to_the_heuristic() -> None:
    """C5 says the service absorbs this rather than dying on it.

    A research-grade backend going missing must not take the process down, and
    the fallback still scores — and the signals it produces name it, so a
    reader of the ledger can tell which model made the decision.
    """

    class Broken:
        @property
        def name(self) -> str:
            return "broken-v1"

        def is_loaded(self) -> bool:
            return False

        async def load(self) -> None:
            raise ModelUnavailableError("no weights on this machine")

        def predict(self, context: Any, horizon: int) -> float:
            raise AssertionError("must not be used")

    predictor = Predictor(Broken(), model_name="tabfm")
    asyncio.run(predictor.load())

    assert predictor.is_loaded()
    assert predictor.name == "heuristic-v1"
    assert predictor.model_name == "heuristic"
    asyncio.run(predictor.aclose())


def test_a_failure_of_the_heuristic_itself_is_not_swallowed() -> None:
    """There is nothing to fall back *to*, so the service must fail to start.

    Falling back twice would leave a service that is running, healthy, and
    unable to score, which is the worst of the three outcomes.
    """

    class Broken:
        @property
        def name(self) -> str:
            return "broken-v1"

        def is_loaded(self) -> bool:
            return False

        async def load(self) -> None:
            raise ModelUnavailableError("no weights")

        def predict(self, context: Any, horizon: int) -> float:
            raise AssertionError("must not be used")

    predictor = Predictor(Broken(), model_name="heuristic")
    with pytest.raises(ModelUnavailableError):
        asyncio.run(predictor.load())


def test_predicting_before_loading_is_refused() -> None:
    """C5's own guard, reached through the service's wrapper."""
    predictor = Predictor(Recording(), model_name="recording")
    try:
        with pytest.raises(ModelNotLoadedError):
            asyncio.run(predictor.predict(None, 1))
    finally:
        asyncio.run(predictor.aclose())


def test_the_selector_is_kept_for_logging_and_health() -> None:
    predictor = Predictor(Recording(), model_name="heuristic")
    assert predictor.model_name == "heuristic"
    assert predictor.name == "recording-v1", "the signal names the weights, not the selector"
    asyncio.run(predictor.aclose())
