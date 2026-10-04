"""The production loop: pull, score, signal, once per minute close."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd

from model import InvalidContextError, MAX_CONTEXT_ROWS, MIN_CONTEXT_ROWS

from .candles import CandleStore
from .config import InferenceConfig
from .cursors import DecisionCursors
from .features import WARMUP_ROWS, FeatureBuilder, FeatureError, FeatureTable
from .predictor import Predictor
from .signals import Signal

logger = logging.getLogger("mft.inference.loop")


TICK_PERIOD = timedelta(minutes=1)


MIN_SLEEP_SECONDS = 0.5


DEFAULT_MAX_CONTEXT_AGE = timedelta(minutes=5)


STALENESS_MARGIN = timedelta(minutes=1)


_MIN_USABLE_ROWS = MIN_CONTEXT_ROWS


@runtime_checkable
class SignalSender(Protocol):
    """What the loop needs from the execution client."""

    async def submit(self, signal: Signal) -> object:
        """Deliver a signal."""
        ...

    async def aclose(self) -> None:
        """Release resources."""
        ...


class Outcome(str, Enum):
    """Why a symbol did or did not produce a signal."""

    SENT = "sent"
    DRY_RUN = "dry_run"
    BELOW_THRESHOLD = "below_threshold"
    NO_DIRECTION = "no_direction"
    NO_CONTEXT = "no_context"
    BAD_CONTEXT = "bad_context"
    STALE_CONTEXT = "stale_context"
    NOT_LOADED = "not_loaded"
    SEND_FAILED = "send_failed"
    ERROR = "error"
    UNCHANGED = "unchanged"

    @property
    def is_failure(self) -> bool:
        """Whether this outcome is something an operator should look at."""
        return self in _FAILURES


_FAILURES = frozenset(
    {
        Outcome.NO_CONTEXT,
        Outcome.BAD_CONTEXT,
        Outcome.STALE_CONTEXT,
        Outcome.NOT_LOADED,
        Outcome.SEND_FAILED,
        Outcome.ERROR,
    }
)


@dataclass(frozen=True, slots=True)
class Decision:
    """What the loop decided about one symbol at one minute close."""

    symbol: str
    outcome: Outcome
    detail: str = ""
    score: float | None = None
    signal: Signal | None = None
    order_id: str = ""
    candidates: int = 0
    as_of: datetime | None = None

    def __str__(self) -> str:
        score = "n/a" if self.score is None else f"{self.score:+.4f}"
        tail = f" [{self.detail}]" if self.detail else ""
        return f"{self.symbol} {self.outcome.value} score={score}{tail}"


class MinuteScheduler:
    """Scores every configured instrument once per minute close."""

    def __init__(
        self,
        config: InferenceConfig,
        store: CandleStore,
        predictor: Predictor,
        sender: SignalSender,
        *,
        timezone_name: str = "Asia/Kolkata",
        builder: FeatureBuilder | None = None,
        max_context_age: timedelta | None = DEFAULT_MAX_CONTEXT_AGE,
        tick_period: timedelta = TICK_PERIOD,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._predictor = predictor
        self._sender = sender
        self._builder = builder if builder is not None else FeatureBuilder(timezone_name)
        self._max_context_age = max_context_age
        self._tick_period = tick_period
        self._clock = clock if clock is not None else _utcnow
        self._ticks = 0
        self._last_tick: datetime | None = None
        self._cursors = DecisionCursors(config.cursor_path)
        self._symbol_locks: dict[str, asyncio.Lock] = {}
        self._outcomes: dict[str, int] = {}

    @property
    def config(self) -> InferenceConfig:
        """The `inference` section this loop runs from."""
        return self._config

    def claim_owner(self) -> None:
        """Reserve this scheduler's persistent cursor journal for its runtime."""
        self._cursors.claim_owner()

    def close(self) -> None:
        """Release cursor ownership after all evaluations have stopped."""
        self._cursors.close()

    @property
    def ticks(self) -> int:
        """Completed passes over the instrument list."""
        return self._ticks

    @property
    def last_tick(self) -> datetime | None:
        """When the last pass started."""
        return self._last_tick

    @property
    def instruments(self) -> tuple[str, ...]:
        """The instruments this loop scores."""
        return tuple(self._config.instruments)

    @property
    def context_rows(self) -> int:
        """Rows handed to the model, after the warm-up margin is removed."""
        return min(self._config.context_rows, MAX_CONTEXT_ROWS)

    @property
    def builder(self) -> FeatureBuilder:
        """The feature builder shared by every symbol."""
        return self._builder


    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Fire on every minute boundary until cancelled or `stop` is set."""
        logger.info(
            "scheduler started: instruments=%s cadence=%s dry_run=%s threshold=%.3f",
            ",".join(self.instruments) or "<none>",
            self._tick_period,
            self._config.dry_run,
            self._config.score_threshold,
        )
        while True:
            delay = self._seconds_to_next_boundary()
            if stop is None:
                await asyncio.sleep(delay)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    return
                except TimeoutError:
                    pass
            await self.tick()

    def _seconds_to_next_boundary(self, now: datetime | None = None) -> float:
        """Seconds until the next tick boundary, floored against spinning."""
        moment = (now or self._clock()).astimezone(timezone.utc)
        period = self._tick_period.total_seconds()
        next_boundary = (moment.timestamp() // period + 1) * period
        floor = min(MIN_SLEEP_SECONDS, period / 10.0)
        return max(next_boundary - moment.timestamp(), floor)

    async def tick(self) -> list[Decision]:
        """Score every instrument once."""
        self._ticks += 1
        self._last_tick = self._clock()
        async def score(symbol: str) -> Decision:
            try:
                return await self.evaluate(symbol)
            except Exception as err:  # noqa: BLE001
                logger.exception("unhandled error scoring %s: %s", symbol, err)
                return Decision(
                    symbol=symbol,
                    outcome=Outcome.ERROR,
                    detail=f"{type(err).__name__}: {err}",
                )
        decisions: list[Decision] = []
        symbols = self.instruments
        concurrency = min(
            self._config.max_concurrency,
            getattr(self._store, "capacity", self._config.max_concurrency),
            getattr(self._predictor, "capacity", self._config.max_concurrency),
        )
        for start in range(0, len(symbols), concurrency):
            group = symbols[start:start + concurrency]
            decisions.extend(await asyncio.gather(*(score(symbol) for symbol in group)))
        for decision in decisions:
            key = decision.outcome.value
            self._outcomes[key] = self._outcomes.get(key, 0) + 1
            self._log(decision)
        return decisions

    @property
    def outcome_counts(self) -> dict[str, int]:
        return dict(self._outcomes)

    def _log(self, decision: Decision) -> None:
        if decision.outcome in (Outcome.SENT, Outcome.DRY_RUN):
            logger.info("signal %s", decision)
        elif decision.outcome.is_failure:
            logger.warning("skipped %s", decision)
        else:
            logger.debug("no signal %s", decision)


    async def evaluate(self, symbol: str) -> Decision:
        """Pull, build, score, and maybe signal one symbol."""
        lock = self._symbol_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            decision = await self._evaluate(symbol)
            completed = {Outcome.SENT, Outcome.DRY_RUN, Outcome.BELOW_THRESHOLD, Outcome.NO_DIRECTION, Outcome.SEND_FAILED}
            if decision.outcome in completed and decision.as_of is not None:
                await asyncio.to_thread(self._cursors.advance, symbol, decision.as_of)
            return decision

    async def _evaluate(self, symbol: str) -> Decision:
        if not self._predictor.is_loaded():
            return Decision(
                symbol=symbol,
                outcome=Outcome.NOT_LOADED,
                detail="model is not loaded; no signal is built on a guess",
            )

        horizon = self._config.horizon_bars


        wanted = min(self._config.context_rows, MAX_CONTEXT_ROWS)
        try:
            candles = await self._store.tail(symbol, wanted + WARMUP_ROWS)
        except Exception as err:  # noqa: BLE001
            logger.warning("cannot read candles for %s: %s", symbol, err)
            return Decision(
                symbol=symbol,
                outcome=Outcome.ERROR,
                detail=f"candle read failed: {type(err).__name__}: {err}",
            )

        if not candles:
            return Decision(
                symbol=symbol,
                outcome=Outcome.NO_CONTEXT,
                detail="no candles stored for this symbol",
            )

        stale = self._staleness_of(candles[-1].timestamp)
        if stale is not None:
            return Decision(
                symbol=symbol,
                outcome=Outcome.STALE_CONTEXT,
                detail=f"newest candle is {stale} old: the store has not advanced",
                candidates=len(candles),
                as_of=candles[-1].timestamp,
            )

        if self._cursors.contains(symbol, candles[-1].timestamp):
            return Decision(symbol=symbol, outcome=Outcome.UNCHANGED, as_of=candles[-1].timestamp)

        try:
            table = self._builder.build(candles)
        except FeatureError as err:
            logger.warning("feature build failed for %s: %s", symbol, err)
            return Decision(
                symbol=symbol,
                outcome=Outcome.BAD_CONTEXT,
                detail=f"feature build failed: {err}",
                candidates=len(candles),
                as_of=candles[-1].timestamp,
            )

        try:
            context = self._context_for(table, wanted, horizon)
        except FeatureError as err:
            return Decision(
                symbol=symbol,
                outcome=Outcome.BAD_CONTEXT,
                detail=str(err),
                candidates=len(candles),
                as_of=table.as_of,
            )

        try:
            score = await self._predictor.predict(context, horizon)
        except InvalidContextError as err:
            logger.warning("model rejected the context for %s: %s", symbol, err)
            return Decision(
                symbol=symbol,
                outcome=Outcome.BAD_CONTEXT,
                detail=f"model rejected the context: {err}",
                candidates=len(context),
                as_of=table.as_of,
            )
        except Exception as err:  # noqa: BLE001
            logger.exception("model failed for %s: %s", symbol, err)
            return Decision(
                symbol=symbol,
                outcome=Outcome.ERROR,
                detail=f"model failed: {type(err).__name__}: {err}",
                candidates=len(context),
                as_of=table.as_of,
            )

        decision = await self._maybe_signal(
            symbol, score, context, table, candles[-1].close
        )
        logger.debug(
            "%s: %d candles, %d context rows, %d columns",
            symbol,
            len(candles),
            len(context),
            len(context.columns),
        )
        return decision

    def _context_for(
        self, table: FeatureTable, wanted: int, horizon: int
    ) -> pd.DataFrame:
        """Trim the warm-up, cap the prompt, and check the model can use it."""
        context = table.context(wanted)
        needed = max(_MIN_USABLE_ROWS + horizon, horizon + 1)
        if len(context) < needed:
            raise FeatureError(
                f"{len(context)} warmed rows is short of the {needed} the model "
                f"needs to predict {horizon} bar(s) ahead"
            )
        return context

    async def _maybe_signal(
        self,
        symbol: str,
        score: float,
        context: pd.DataFrame,
        table: FeatureTable,
        last_close: float,
    ) -> Decision:
        """Threshold, derive a side, and either log or post the signal."""
        threshold = self._config.score_threshold
        magnitude = abs(score)
        if magnitude < threshold:
            return Decision(
                symbol=symbol,
                outcome=Outcome.BELOW_THRESHOLD,
                detail=f"|{score:+.4f}| < threshold {threshold:.4f}",
                score=score,
                candidates=len(context),
                as_of=table.as_of,
            )

        try:
            signal = Signal.from_score(
                symbol=symbol,
                score=score,
                price=last_close,
                quantity=self._config.order_quantity,
                model=self._predictor.name,
                as_of=table.as_of,
            )
        except ValueError as err:


            return Decision(
                symbol=symbol,
                outcome=Outcome.NO_DIRECTION,
                detail=str(err),
                score=score,
                candidates=len(context),
                as_of=table.as_of,
            )

        if self._config.dry_run:


            logger.info(
                "dry_run: would POST %s to %s with %s",
                signal.idempotency_key,
                self._config.execution_url,
                signal.to_wire(),
            )
            return Decision(
                symbol=symbol,
                outcome=Outcome.DRY_RUN,
                detail="dry_run is on; no order was placed",
                score=score,
                signal=signal,
                candidates=len(context),
                as_of=table.as_of,
            )

        try:
            ack = await self._sender.submit(signal)
        except Exception as err:  # noqa: BLE001


            logger.error(
                "signal %s for %s was not delivered: %s",
                signal.idempotency_key,
                symbol,
                err,
            )
            return Decision(
                symbol=symbol,
                outcome=Outcome.SEND_FAILED,
                detail=f"{type(err).__name__}: {err}",
                score=score,
                signal=signal,
                candidates=len(context),
                as_of=table.as_of,
            )

        return Decision(
            symbol=symbol,
            outcome=Outcome.SENT,
            detail=str(ack),
            score=score,
            signal=signal,
            order_id=getattr(ack, "order_id", ""),
            candidates=len(context),
            as_of=table.as_of,
        )


    def _staleness_of(self, as_of: datetime) -> timedelta | None:
        """How old the newest candle is, or `None` if that is unacceptable."""
        if self._max_context_age is None:
            return None
        age = self._clock().astimezone(timezone.utc) - as_of.astimezone(timezone.utc)
        return age if age > self._max_context_age or age < -TICK_PERIOD else None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def staleness_guard(flush_interval_seconds: int) -> timedelta:
    """The staleness guard for a store that publishes every N seconds."""
    if flush_interval_seconds <= 0:
        return DEFAULT_MAX_CONTEXT_AGE
    return max(
        DEFAULT_MAX_CONTEXT_AGE,
        timedelta(seconds=flush_interval_seconds) + STALENESS_MARGIN,
    )

__all__ = [
    "DEFAULT_MAX_CONTEXT_AGE",
    "Decision",
    "MIN_SLEEP_SECONDS",
    "MinuteScheduler",
    "Outcome",
    "STALENESS_MARGIN",
    "SignalSender",
    "TICK_PERIOD",
    "staleness_guard",
]
