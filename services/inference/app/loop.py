"""The production loop: pull, score, signal, once per minute close.

This is the component that closes the north star, and it is deliberately not
an HTTP endpoint. `docs/contracts.md` §7 says so explicitly: the production
path is an internal scheduler that fires on each minute close, pulls context
from DuckDB, runs the model and POSTs to execution.

# Why pull, and what that buys

Plan.md §4 chose pull over push for cross-service data. At a 60-second
decision cadence, push latency buys nothing, and pull is what makes the rest
of the design possible:

* **Missed minutes are free.** If a tick takes too long, the next boundary
  simply runs again and pulls whatever is now in the store. There is no queue
  to back up and no gap in the decision record.
* **A restart catches up.** The process comes back, reads the newest candles
  it can find, and continues. Nothing replays a stream it missed.
* **The service is stateless.** Nothing is pushed *at* it, so it can be
  restarted at any minute without a handshake.

# Why the loop fails closed, per symbol

Every step that could produce a confidently wrong signal is a skip, and every
skip is logged with the reason. A short window, a feature build that rejects
its input, a context the model rejects, a model that is not loaded, a context
older than the guard, a zero score with no direction — each is one symbol
being skipped, never the loop and never another symbol. An exception escaping
one symbol is contained here rather than propagated, because a single symbol's
broker hiccup must not silence the other instruments for the rest of the
session.

The one thing that does stop the loop is `dry_run` being on. Not "stops
sending" — it logs the signal it would have sent, with the full body, so a
research run leaves a decision record that can be diffed against a live one.

# Why the context is pulled with a warm-up margin

`inference.context_rows` is the number of rows handed to the model. Features
for the first 60 rows of any window are context-only by contract, so a window
of exactly `context_rows` candles would spend most of its prompt on zeros. The
loop pulls `context_rows + WARMUP_ROWS` candles, builds the table, and drops
the warm-up rows before scoring.

# Why the staleness guard is tied to the flush interval

The newest *published* candle lags the clock. C2 flushes on
`storage.flush_interval_seconds` and only then is the file renamed into the
`part-*.parquet` glob this loop reads, so a store that is working perfectly
still reports a context that is up to one flush interval old. A guard tighter
than that would skip every minute of a healthy system.

So the guard is derived from the flush interval rather than guessed, and
`staleness_guard` is what the wiring calls. A deployment that flushes every
30 seconds does not also have to remember to shorten the guard, and one that
flushes every ten minutes is not skipped by a guard that assumed the default.
"""

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
from .features import WARMUP_ROWS, FeatureBuilder, FeatureError, FeatureTable
from .predictor import Predictor
from .signals import Signal

logger = logging.getLogger("mft.inference.loop")

#: The decision cadence. One minute is a property of the market, not a knob:
#: the candles are 1-minute bars, so a faster tick would score the same bar
#: twice and a slower one would skip bars outright.
TICK_PERIOD = timedelta(minutes=1)

#: Ceiling on the minimum sleep between ticks, so a pass that overran its
#: window cannot spin against the boundary it just missed. See
#: `MinuteScheduler._seconds_to_next_boundary`.
MIN_SLEEP_SECONDS = 0.5

#: Floor on the staleness guard, applied whatever the flush interval is.
#: A context that has not advanced while the loop has run is a stalled store
#: or a dead feed, and scoring it would trade a stale number.
DEFAULT_MAX_CONTEXT_AGE = timedelta(minutes=5)

#: Slack added to the flush interval, covering the flush that is in progress
#: and the clock skew between the writer and this reader. A minute is ample
#: for both processes on one machine, which is the deployment Plan.md §2
#: describes.
STALENESS_MARGIN = timedelta(minutes=1)

#: Fewest rows a model may be given, independent of what was configured.
#: `MIN_CONTEXT_ROWS` is C5's warm-up floor for the model itself; nothing below
#: it is a usable prompt table whichever backend is selected.
_MIN_USABLE_ROWS = MIN_CONTEXT_ROWS


@runtime_checkable
class SignalSender(Protocol):
    """What the loop needs from the execution client."""

    async def submit(self, signal: Signal) -> object:
        """Deliver a signal. Raises `ExecutionError` when it did not land."""
        ...

    async def aclose(self) -> None:
        """Release resources."""
        ...


class Outcome(str, Enum):
    """Why a symbol did or did not produce a signal.

    A `str` enum so it logs as a readable word rather than a repr, and so a
    caller can compare it to a string from a metric label.
    """

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
    """Scores every configured instrument once per minute close.

    Args:
        config: The frozen `inference` section.
        store: Where candles are pulled from.
        predictor: The loaded model and the single thread that may call it.
        sender: Where signals are posted. Required even in `dry_run`, so the
            production wiring is identical to the research wiring.
        builder: The feature builder. One is built in `timezone_name` when
            omitted; it holds nothing but the zone, so it is shared.
        max_context_age: Reject a context whose newest candle is older than
            this. `None` disables the guard; it is on by default because a
            context that has not advanced is a stalled store, not a market.
        tick_period: The decision cadence. Defaults to one minute; the tests
            drive `tick` directly and never wait on this.
        clock: Injected UTC clock, so the staleness guard is testable.
    """

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

    @property
    def config(self) -> InferenceConfig:
        """The `inference` section this loop runs from."""
        return self._config

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

    # -- the loop ---------------------------------------------------------

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Fire on every minute boundary until cancelled or `stop` is set.

        The sleep is to the next boundary, not for a fixed period, so ticks
        do not drift: a pass that takes four seconds still starts on the next
        minute, not 64 seconds after the one before it.
        """
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
        """Seconds until the next tick boundary, floored against spinning.

        Boundaries are aligned to the wall clock in UTC, which is the same
        grid the candle timestamps are truncated to. A pass that overran its
        window would otherwise find the next boundary in the past and tick
        again immediately, so the wait never falls below a tenth of the period
        — capped at half a second, which is all the protection a one-minute
        cadence needs.
        """
        moment = (now or self._clock()).astimezone(timezone.utc)
        period = self._tick_period.total_seconds()
        next_boundary = (moment.timestamp() // period + 1) * period
        floor = min(MIN_SLEEP_SECONDS, period / 10.0)
        return max(next_boundary - moment.timestamp(), floor)

    async def tick(self) -> list[Decision]:
        """Score every instrument once. Never raises.

        Returns one `Decision` per instrument, in configured order. An
        exception from one symbol is contained and recorded as
        `Outcome.ERROR` so the others still run.
        """
        self._ticks += 1
        self._last_tick = self._clock()
        decisions: list[Decision] = []
        for symbol in self.instruments:
            try:
                decisions.append(await self.evaluate(symbol))
            except Exception as err:  # noqa: BLE001 - one symbol must not stop the rest
                logger.exception("unhandled error scoring %s: %s", symbol, err)
                decisions.append(
                    Decision(
                        symbol=symbol,
                        outcome=Outcome.ERROR,
                        detail=f"{type(err).__name__}: {err}",
                    )
                )
        for decision in decisions:
            self._log(decision)
        return decisions

    def _log(self, decision: Decision) -> None:
        if decision.outcome in (Outcome.SENT, Outcome.DRY_RUN):
            logger.info("signal %s", decision)
        elif decision.outcome.is_failure:
            logger.warning("skipped %s", decision)
        else:
            logger.debug("no signal %s", decision)

    # -- one symbol -------------------------------------------------------

    async def evaluate(self, symbol: str) -> Decision:
        """Pull, build, score, and maybe signal one symbol.

        Returns rather than raises for every expected failure, because the
        caller iterates a list and one symbol's bad minute is not the end of
        the session for the others.
        """
        if not self._predictor.is_loaded():
            return Decision(
                symbol=symbol,
                outcome=Outcome.NOT_LOADED,
                detail="model is not loaded; no signal is built on a guess",
            )

        horizon = self._config.horizon_bars
        # Capped at the model's own row budget before the warm-up margin is
        # added, so a `context_rows` larger than TabFM can attend to costs
        # nothing at the store either.
        wanted = min(self._config.context_rows, MAX_CONTEXT_ROWS)
        try:
            candles = await self._store.tail(symbol, wanted + WARMUP_ROWS)
        except Exception as err:  # noqa: BLE001 - a bad read is a skip
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
        except Exception as err:  # noqa: BLE001 - a model fault is a skip
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
        """Trim the warm-up, cap the prompt, and check the model can use it.

        Raises:
            FeatureError: If fewer rows survive than the model needs. That is
                a skip, never a shorter prompt: a model handed fewer rows than
                its floor answers with whatever it can still attend to, which
                is a confident number computed from too little.
        """
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
        """Threshold, derive a side, and either log or post the signal.

        `last_close` is the close of the newest candle, which is the reference
        price in the signal. It is not the predicted value and not a feature:
        the frozen 18-column schema carries no close, and a score in [-1, 1] is
        not a price.
        """
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
            # A score of exactly 0 only reaches here with threshold 0, but a
            # score outside [-1, 1] should never be traded either.
            return Decision(
                symbol=symbol,
                outcome=Outcome.NO_DIRECTION,
                detail=str(err),
                score=score,
                candidates=len(context),
                as_of=table.as_of,
            )

        if self._config.dry_run:
            # The signal is built and logged in full precisely so a research
            # run leaves a record a live run can be diffed against. Nothing
            # leaves this process.
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
        except Exception as err:  # noqa: BLE001 - a failed order must not kill the loop
            # The signal is logged with its key so the next minute's state can
            # be reconciled, and the loop moves on. Retrying here would be
            # wrong: the client's retries are bounded, and a decision that
            # outlives its minute is a decision made on a stale context.
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

    # -- helpers ----------------------------------------------------------

    def _staleness_of(self, as_of: datetime) -> timedelta | None:
        """How old the newest candle is, or `None` if that is unacceptable."""
        if self._max_context_age is None:
            return None
        age = self._clock().astimezone(timezone.utc) - as_of.astimezone(timezone.utc)
        return age if age > self._max_context_age else None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def staleness_guard(flush_interval_seconds: int) -> timedelta:
    """The staleness guard for a store that publishes every N seconds.

    One flush interval plus a minute of slack, and never below
    `DEFAULT_MAX_CONTEXT_AGE`.
    """
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
