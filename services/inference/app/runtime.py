"""Assembling the service from configuration.

One place decides what a configured inference service *is*: which model, which
Parquet tree, which execution endpoint, and whether the scheduler runs at all.
`main` renders it over HTTP, the tests build one with fakes, and neither has to
know how the pieces are wired.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from model import available_models

from .candles import CandleStore, DuckDBCandleStore, candles_root
from .config import Config, InferenceConfig
from .execution import ExecutionClient
from .loop import MinuteScheduler, SignalSender, staleness_guard
from .predictor import Predictor

logger = logging.getLogger("mft.inference.runtime")


@dataclass(slots=True)
class Runtime:
    """Everything the HTTP layer needs, assembled and not yet started."""

    config: Config
    store: CandleStore
    predictor: Predictor
    sender: SignalSender
    scheduler: MinuteScheduler
    owns_store: bool = True
    owns_sender: bool = True
    _stop: asyncio.Event | None = None
    _task: asyncio.Task[None] | None = None

    @property
    def inference(self) -> InferenceConfig:
        """The `inference` section."""
        return self.config.inference

    @property
    def dry_run(self) -> bool:
        """Whether orders are actually placed. True by default."""
        return self.inference.dry_run

    async def startup(self, *, run_scheduler: bool = True) -> asyncio.Task[None] | None:
        """Load the model, then start the loop. Returns the scheduler task."""
        await self.predictor.load()
        if not run_scheduler:
            return None
        if not self.inference.instruments:
            logger.warning(
                "inference.instruments is empty; the scheduler will idle. "
                "Set it in the config to trade anything."
            )
        stop = asyncio.Event()
        self._stop = stop
        task = asyncio.create_task(self.scheduler.run(stop), name="minute-scheduler")
        self._task = task
        return task
    async def shutdown(self) -> None:
        """Stop the loop and release every resource this runtime created."""
        if self._stop is not None:
            self._stop.set()
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as err:  # noqa: BLE001 - shutdown is best effort
                logger.warning("scheduler stopped with %s: %s", type(err).__name__, err)
        await self.sender.aclose()
        await self.predictor.aclose()
        if self.owns_store:
            close = getattr(self.store, "aclose", None)
            if close is not None:
                await close()


def build_runtime(config: Config, *, root: Path | None = None) -> Runtime:
    """Build a runtime from a validated config.

    Args:
        config: The frozen config, already through `load_config`.
        root: Overrides the Parquet dataset root. The tests point this at a
            temporary tree; production leaves it `None` so `storage.data_dir`
            decides.

    Raises:
        RuntimeError: If `inference.model` names a model that is not
            registered. C5 registers `heuristic` always and `tabfm` only when
            the backend imports, so this is the honest failure for a venv built
            with `--skip-model` that was never told to use the heuristic.
    """
    inference = config.inference
    known = ", ".join(available_models()) or "<none>"
    if inference.model not in available_models():
        raise RuntimeError(
            f"inference.model is {inference.model!r} but the registered models "
            f"are: {known}. Install the model backend, or set inference.model: heuristic."
        )

    dataset_root = candles_root(root if root is not None else config.storage.data_dir)
    store = DuckDBCandleStore(dataset_root)
    predictor = Predictor.from_name(inference.model)
    sender = ExecutionClient(inference.execution_url)
    scheduler = MinuteScheduler(
        inference,
        store,
        predictor,
        sender,
        timezone_name=config.app.timezone,
        # Tied to the flush interval rather than fixed, so a store that
        # publishes more often than the default is not skipped every minute.
        max_context_age=staleness_guard(config.storage.flush_interval_seconds),
    )

    logger.info(
        "runtime built: model=%s dry_run=%s dataset=%s execution=%s",
        inference.model,
        inference.dry_run,
        dataset_root,
        inference.execution_url,
    )
    return Runtime(
        config=config, store=store, predictor=predictor, sender=sender, scheduler=scheduler
    )


__all__ = ["Runtime", "build_runtime"]
