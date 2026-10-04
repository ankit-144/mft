"""Holding the model and calling it off the event loop."""

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
MAX_PENDING_MODEL_CALLS = 8


class PredictorBusyError(RuntimeError):
    """The serialized model worker has reached its admitted work limit."""


class Predictor:
    """A loaded `InferenceModel` plus the single thread that may call it."""

    def __init__(self, model: InferenceModel, *, model_name: str = "") -> None:
        self._model = model
        self._model_name = model_name or getattr(model, "name", "unknown")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="predict"
        )
        self._loaded = False
        self._closed = False
        self._pending = asyncio.Semaphore(self.capacity)

    @property
    def capacity(self) -> int:
        """Maximum admitted model calls, including the active worker."""
        return MAX_PENDING_MODEL_CALLS

    @classmethod
    def from_name(cls, name: str) -> Predictor:
        """Build the model `inference.model` names."""
        return cls(create_model(name), model_name=name)

    @property
    def model_name(self) -> str:
        """The configured model selector."""
        return self._model_name

    @property
    def name(self) -> str:
        """The loaded implementation's model identity."""
        return self._model.name

    def is_loaded(self) -> bool:
        """Whether `predict` can be called."""
        return self._loaded and self._model.is_loaded()

    async def load(self) -> None:
        """Load the model, falling back to the heuristic if it cannot be had."""
        try:
            await self._run(lambda: asyncio.run(self._model.load()))
        except ModelUnavailableError as err:
            if self._model_name != "tabfm":
                raise
            logger.error(
                "model %r is unavailable (%s); falling back to the heuristic model. "
                "Signals will be produced by heuristic-v1 and carry its name.",
                self._model_name,
                err,
            )
            self._model = create_model("heuristic")
            self._model_name = "heuristic"
            await self._run(lambda: asyncio.run(self._model.load()))
        self._loaded = True
        logger.info("predictor ready: selector=%s model=%s", self._model_name, self.name)

    async def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Score a context table, off the event loop, on the single worker."""
        if not self.is_loaded():
            raise ModelNotLoadedError(
                f"{self._model_name} is not loaded; call load() before predicting"
            )
        return await self._run(self._model.predict, context, horizon)

    async def aclose(self) -> None:
        """Stop the worker thread."""
        if not self._closed:
            self._closed = True
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    async def _run(self, fn: Any, *args: Any) -> Any:
        if self._closed:
            raise RuntimeError("predictor is closed")
        if self._pending.locked():
            raise PredictorBusyError("model worker is at capacity")
        await self._pending.acquire()
        try:
            if self._closed:
                raise RuntimeError("predictor is closed")
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor, fn, *args)
        except BaseException:
            self._pending.release()
            raise
        try:
            return await asyncio.shield(future)
        finally:
            if future.done():
                self._pending.release()
            else:
                future.add_done_callback(lambda _: self._pending.release())


__all__ = ["ModelNotLoadedError", "ModelUnavailableError", "Predictor", "PredictorBusyError"]
