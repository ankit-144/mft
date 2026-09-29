"""Holding the model and calling it off the event loop.

Two jobs, both about the fact that a model is neither async nor reentrant:

1. `predict` is a synchronous, CPU-bound, stateful call. Running it inline
   would stall the event loop for as long as the model takes, which at a
   minute cadence with a slow backend is long enough to miss the next tick.
2. The same model instance is reached from the scheduler *and* from the
   `POST /v1/predict` debug endpoint. Two threads inside one in-context
   learner is undefined behaviour, and TabFM's prompt table is mutable.

Both are answered by one thread with one worker. The cost is that predictions
are serialised; at three instruments a minute that is not a cost at all, and
the alternative — a pool per caller — is how a shared model gets corrupted.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from model import (
    InferenceModel,
    ModelNotLoadedError,
    ModelUnavailableError,
    create_model,
)

logger = logging.getLogger("mft.inference.predictor")


class Predictor:
    """A loaded `InferenceModel` plus the single thread that may call it.

    Args:
        model: An `InferenceModel` from C5's factory. Not loaded yet.
        model_name: The `inference.model` selector the model was built from,
            kept for logging and for the health endpoint.
    """

    def __init__(self, model: InferenceModel, *, model_name: str = "") -> None:
        self._model = model
        self._model_name = model_name or getattr(model, "name", "unknown")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="predict"
        )
        self._loaded = False

    @classmethod
    def from_name(cls, name: str) -> Predictor:
        """Build the model `inference.model` names. Never loads weights here."""
        return cls(create_model(name), model_name=name)

    @property
    def model_name(self) -> str:
        """The configured selector, e.g. `heuristic` or `tabfm`."""
        return self._model_name

    @property
    def name(self) -> str:
        """The loaded implementation's own name, e.g. `heuristic-v1`.

        This is what goes in a signal's `model` field, and it distinguishes
        weight versions, which is the point of reporting it.
        """
        return self._model.name

    def is_loaded(self) -> bool:
        """Whether `predict` can be called."""
        return self._loaded and self._model.is_loaded()

    async def load(self) -> None:
        """Load the model, falling back to the heuristic if it cannot be had.

        C5 documents `ModelUnavailableError` as something the service absorbs
        rather than dies on: a research-grade backend going missing must not
        take the process down. `inference.dry_run` is the other half of that —
        the service keeps scoring, and keeps the decisions, without trading.

        `load` is awaited on the event loop, not on the worker thread, because
        it is a coroutine: the weights arrive from disk or from a cache and
        there is no CPU work in it to keep the loop busy for.
        """
        try:
            await self._model.load()
        except ModelUnavailableError as err:
            if self._model_name == "heuristic":
                raise
            logger.error(
                "model %r is unavailable (%s); falling back to the heuristic model. "
                "Signals will be produced by heuristic-v1 and carry its name.",
                self._model_name,
                err,
            )
            self._model = create_model("heuristic")
            self._model_name = "heuristic"
            await self._model.load()
        self._loaded = True
        logger.info("predictor ready: selector=%s model=%s", self._model_name, self.name)

    async def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Score a context table, off the event loop, on the single worker.

        The loaded-state guard is repeated here rather than trusted to each
        implementation. C5's own models raise `ModelNotLoadedError` and would
        catch this anyway, but the guard is the service's contract, and a
        model added later behind this wrapper should not be able to answer a
        prediction from an unloaded state.
        """
        if not self.is_loaded():
            raise ModelNotLoadedError(
                f"{self._model_name} is not loaded; call load() before predicting"
            )
        return await self._run(self._model.predict, context, horizon)

    async def aclose(self) -> None:
        """Stop the worker thread."""
        self._executor.shutdown(wait=True)

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)


__all__ = ["ModelNotLoadedError", "ModelUnavailableError", "Predictor"]
