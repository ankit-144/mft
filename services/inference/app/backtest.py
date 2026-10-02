"""`make backtest` — replay stored candles through the live decision path.

`make backtest` runs `python -m app.backtest`, and until this module existed
that target failed. It is the cheapest end-to-end validation of the platform
available without a broker account: no credentials, no network, no live feed,
no order anywhere. That is the entire point — somebody should be able to run it
on a laptop, with nothing else started, and see whether the
feature → model → decision → risk chain produces sane numbers.

# Why it lives in this service, and not in a research package

Because the backtest calls `FeatureBuilder`, `Predictor`, `side_for_score` and
`idempotency_key` — the same four objects `loop.py` calls at every minute
close. A backtest built on its own feature code validates code the live system
does not run, and the only symptom is a beautiful equity curve that a live
minute cannot reproduce. One feature path and one model path is the entire
argument for this module being here.

# No lookahead, and why that is the only property that matters

A backtest that leaks the future is worse than no backtest, because it
manufactures a result that looks like an edge and is not. Every decision is
taken on `candles[: i + 1]` and the bar at `candles[i + horizon]` is the label.
It is read exactly once, after the score exists, to compute realised P&L — and
never handed to the builder or the model. `FeatureBuilder` is already causal
(trailing windows only, no recursion, no state carried between rows), so the
harness's whole job is to not break that by slicing wrong, and then to *prove*
it: `tests/test_backtest.py` runs the harness on a series whose final
`horizon` bars are replaced with a violent move, and asserts the run is
identical to the same series with those bars removed.

# Nothing here talks to the execution service

There is no HTTP client in this module and there must never be one. A backtest
posts nowhere. Risk is evaluated in-process by re-implementing the offline-
checkable subset of `docs/contracts.md` §6, using C6's own reason-code
vocabulary, so a refusal in a backtest and a refusal in production are reported
with the same word. Where this module's arithmetic could drift from the Go risk
engine, the drift is named in `RISK_DRIFT_NOTES` rather than hidden, and every
run prints those notes.

# Two walk-forward modes, and why the default is the slow one

`FeatureBuilder.build` is O(len(window)), so rebuilding the table at every one
of n decision bars is O(n²). At the measured ~8.4 µs per feature row that is
about 17 s for 2000 bars and about 7 minutes for 10000, per symbol. Two
modes:

* `rebuild` (default) — build from `candles[: i + 1]` at every step, exactly
  the loop above. One build per bar, nothing clever, and the slicing is right
  there in the source to be read.
* `incremental` — build the table once over the whole series and slice rows
  `[: i + 1]` per step. O(n) instead of O(n²), and *identical* output: every
  feature at row `i` is a pure function of candles `0..i` with the same
  summation order either way, which `test_backtest.py` asserts exactly, not
  approximately. It differs in one respect, stated here because it is real: it
  validates the whole series up front, so a malformed bar *after* the last
  decision point is an error in this mode and invisible in `rebuild`.

The run is bounded by `--max-bars` and refuses a longer series rather than
truncating it, and the projected wall time of the chosen mode is logged before
any work starts.

# What a run does and does not tell you

On synthetic data (`--synthetic`, the default when the store is empty) it
proves the plumbing and nothing else. The summary says so in the loudest terms
this module can manage, and every artifact it writes carries
`"synthetic": true` and a disclaimer, because the failure mode is a reader
skimming an equity curve and mistaking it for a strategy result.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model import MAX_CONTEXT_ROWS, MIN_CONTEXT_ROWS

from .candles import Candle, CandleStore, DuckDBCandleStore, candles_root
from .config import (
    Config,
    ConfigError,
    InferenceConfig,
    configure_logging,
    load_config,
    resolve_path,
)
from .features import SESSION_OPEN_MINUTE, WARMUP_ROWS, FeatureBuilder, FeatureError, trading_zone
from .predictor import Predictor
from .signals import SIDE_BUY, SIDE_SELL, Side, idempotency_key, rfc3339, side_for_score
from .synth import (
    BARS_PER_YEAR,
    DEFAULT_START_PRICE,
    SESSION_BARS_PER_DAY,
    SynthSpec,
)
from .synth import generate_many

logger = logging.getLogger("mft.inference.backtest")

# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------

#: Model selectors that need `--allow-weights` before this module will build
#: them. `heuristic` has no checkpoint and is always available.
WEIGHTED_MODELS: Final[frozenset[str]] = frozenset({"tabfm"})

#: The backtest's default model. Deliberately *not* `inference.model`: that key
#: defaults to `tabfm` in `docs/contracts.md` §8, and a research tool that
#: loaded 6.6 GB of non-commercial weights because nobody passed a flag would be
#: a foot-gun aimed at the person who wanted to see a number quickly.
DEFAULT_MODEL: Final[str] = "heuristic"

#: Bars per symbol the `rebuild` walk-forward will run on without being asked
#: to. About 90 s of feature building per symbol at 5000 bars, so this is
#: generous rather than convenient; raise it with `--max-bars`, or switch to
#: `--feature-mode incremental`.
DEFAULT_MAX_BARS: Final[int] = 5000

#: Equity-curve points written to the JSON artifact. The full curve is kept in
#: memory for the metrics; a 6000-point curve in a file a human opens is noise.
MAX_JSON_CURVE_POINTS: Final[int] = 1000

#: Measured wall time for one feature row in `FeatureBuilder.build` on this
#: machine, used only to project a run's cost before starting it. It is an
#: estimate and the run reports its own measured time afterwards; the point is
#: that a refusal names a number rather than saying "this might take a while".
FEATURE_ROW_SECONDS: Final[float] = 8.4e-6

#: Per-trade rows printed to the terminal. The JSON carries every one.
DEFAULT_MAX_TRADES_SHOWN: Final[int] = 10

#: The banner printed the moment `--model tabfm --allow-weights` is accepted.
#: Someone will run this with the real model eventually, and they need to have
#: been told what they just loaded.
WEIGHTS_WARNING: Final[str] = """
########################################################################
#  TabFM SELECTED.                                                         #
#                                                                        #
#  The TabFM checkpoint is ~6.6 GB and is licensed                          #
#  `tabfm-non-commercial-v1.0`: NON-COMMERCIAL, NON-PRODUCTION.            #
#  See Plan.md section 5.                                                  #
#                                                                        #
#  This run is research on a laptop. It is not evidence for trading        #
#  real money, and it does not become evidence by looking convincing.     #
#  Nothing it prints is a licence to deploy.                               #
########################################################################
"""

#: Places this module could drift from `services/execution/risk` (C6), named
#: rather than left for a reader to discover. A backtest that quietly diverges
#: from the production risk gate validates a gate nobody runs.
RISK_DRIFT_NOTES: Final[tuple[str, ...]] = (
    "Debounce is a SYMBOL:SIDE ledger with execution.debounce_ttl_seconds, as "
    "docs/contracts.md section 6 check 5 describes. The duplicate ledger is keyed "
    "by app.signals.idempotency_key so this module and the live loop cannot "
    "disagree about what a key is; the two are different keys and the contract "
    "names both.",
    "Lot size is not in the frozen config (docs/contracts.md section 8), so check 6 "
    "is applied without the qty % lot_size term.",
    "max_open_positions counts distinct symbols, because the contract's Portfolio "
    "is a map keyed by symbol. A second leg on an already-open symbol does not "
    "consume a slot, which is the literal reading of len(OpenPositions).",
    "The market-hours window is the NSE session from features.SESSION_OPEN_MINUTE, "
    "not a value read from the risk engine, plus execution.market_holidays.",
    "The drawdown and daily-loss checks read this simulation's equity, marked to the "
    "last known close per symbol. C6 reads a broker portfolio.",
    "No margin or borrow model: a SELL is covered by the whole of execution.capital "
    "and its proceeds are treated as fully available. That flatters any strategy "
    "that leans on shorts.",
)

# --------------------------------------------------------------------------
# Charges
# --------------------------------------------------------------------------

# Every rate below is an ASSUMPTION. None of it is read from the broker or from
# the config, because `docs/contracts.md` §8 declares no charge keys, so a run
# cannot know this broker's schedule. They are named constants rather than
# inline literals so a reader can change one and see the effect.
#
# What is *not* an assumption is the shape, and the shape is what this platform's
# own defaults make expensive. With `inference.order_quantity: 10` on a ~Rs
# 1,400 stock the entry notional is about Rs 14,000, and one round trip costs:
#
#   brokerage   2 x Rs 4.20  (0.03% per leg; the Rs 20 floor only binds above a
#                            Rs 66,667 notional, so at this size the percentage
#                            is what applies)                  = Rs  8.40
#   STT         2 x 0.1% of notional                          = Rs 28.00
#   stamp duty  0.015% of the buy leg                         = Rs  2.10
#   exchange    2 x 0.00297%                                  = Rs  0.83
#   SEBI        2 x 0.0001%                                   = Rs  0.03
#   DP          one flat charge on the sell                   = Rs 15.93
#   GST         18% of brokerage + exchange + SEBI + DP       = Rs  4.53
#                                                              -----------
#                                                              Rs 59.82
#
# which is 0.43% of the entry notional, against a 1-bar standard deviation of
# 0.30 / sqrt(94500) = 0.098% on the same series. So a 1-minute decision has to
# be right by about four and a half of its own standard deviations on every
# single trade merely to break even. That is the finding the cost model exists
# to produce, and a backtest that omits these rates reports it as an edge.
#
# What these numbers are modelled on, and which a user must re-verify against
# their own broker:
#   - brokerage 0.03% capped at Rs 20 per executed order (a flat-rate schedule)
#   - STT 0.1% on both the buy and the sell side of cash/delivery equity
#   - exchange transaction charge ~0.00297% of turnover, both sides
#   - SEBI turnover fee Rs 10 per crore = 0.0001%, both sides
#   - DP charge levied per scrip on the sell transaction
#   - stamp duty 0.015% on the buy side only
#   - GST at 18% on brokerage + exchange + SEBI + DP, and not on STT

#: Brokerage as a percentage of turnover. The binding term at a small order.
BROKERAGE_PCT: Final[float] = 0.03
#: Brokerage floor per executed order, in rupees. Binds above a Rs 66,667
#: notional, which a 10-share order only reaches on a ~Rs 6,700 stock.
BROKERAGE_CAP_RUPEES: Final[float] = 20.0
#: Securities transaction tax, buy side.
STT_BUY_PCT: Final[float] = 0.10
#: Securities transaction tax, sell side.
STT_SELL_PCT: Final[float] = 0.10
#: Exchange transaction charge, both sides.
EXCHANGE_TRANSACTION_PCT: Final[float] = 0.00297
#: SEBI turnover fee, both sides.
SEBI_TURNOVER_PCT: Final[float] = 0.0001
#: Depository participant charge, levied per scrip on the sell transaction.
DP_CHARGE_RUPEES: Final[float] = 15.93
#: Stamp duty, buy side only.
STAMP_DUTY_BUY_PCT: Final[float] = 0.015
#: GST, on brokerage and the transaction charges but not on STT.
GST_PCT: Final[float] = 18.0


class ChargeModel(BaseModel):
    """Indian equity cash/delivery charges, per executed order.

    A model rather than a block of constants so a run can report exactly which
    rates produced its numbers, and so the gross figure can be shown beside the
    net one. Charges are per *leg*: one entry leg and one exit leg per trade.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    brokerage_pct: float = BROKERAGE_PCT
    brokerage_cap_rupees: float = BROKERAGE_CAP_RUPEES
    stt_buy_pct: float = STT_BUY_PCT
    stt_sell_pct: float = STT_SELL_PCT
    exchange_transaction_pct: float = EXCHANGE_TRANSACTION_PCT
    sebi_turnover_pct: float = SEBI_TURNOVER_PCT
    dp_charge_rupees: float = DP_CHARGE_RUPEES
    stamp_duty_buy_pct: float = STAMP_DUTY_BUY_PCT
    gst_pct: float = GST_PCT

    @classmethod
    def zero(cls) -> ChargeModel:
        """A charge model that charges nothing.

        Only for reporting the gross number beside the net one. A run never
        defaults to this: a costless backtest is not a backtest.
        """
        return cls(
            brokerage_pct=0.0,
            brokerage_cap_rupees=0.0,
            stt_buy_pct=0.0,
            stt_sell_pct=0.0,
            exchange_transaction_pct=0.0,
            sebi_turnover_pct=0.0,
            dp_charge_rupees=0.0,
            stamp_duty_buy_pct=0.0,
            gst_pct=0.0,
        )

    def leg(self, *, is_buy: bool, quantity: int, price: float) -> ChargeBreakdown:
        """Charges for one executed order.

        Args:
            is_buy: Buy leg or sell leg. Stamp duty and DP are one-sided.
            quantity: Whole shares.
            price: Execution price in rupees.
        """
        notional = float(quantity) * price
        brokerage = min(notional * self.brokerage_pct / 100.0, self.brokerage_cap_rupees)
        exchange = notional * self.exchange_transaction_pct / 100.0
        sebi = notional * self.sebi_turnover_pct / 100.0
        stt = notional * (self.stt_buy_pct if is_buy else self.stt_sell_pct) / 100.0
        stamp = notional * self.stamp_duty_buy_pct / 100.0 if is_buy else 0.0
        dp = 0.0 if is_buy else self.dp_charge_rupees
        gst = (brokerage + exchange + sebi + dp) * self.gst_pct / 100.0
        return ChargeBreakdown(
            notional=notional,
            brokerage=brokerage,
            exchange=exchange,
            sebi=sebi,
            stt=stt,
            stamp_duty=stamp,
            dp_charge=dp,
            gst=gst,
            total=brokerage + exchange + sebi + stt + stamp + dp + gst,
        )

    def round_trip(self, side: Side, quantity: int, entry: float, exit_: float) -> ChargeBreakdown:
        """Both legs of a round trip, summed, with the notional of both legs."""
        first = self.leg(is_buy=side == SIDE_BUY, quantity=quantity, price=entry)
        second = self.leg(is_buy=side == SIDE_SELL, quantity=quantity, price=exit_)
        return ChargeBreakdown(
            notional=first.notional + second.notional,
            brokerage=first.brokerage + second.brokerage,
            exchange=first.exchange + second.exchange,
            sebi=first.sebi + second.sebi,
            stt=first.stt + second.stt,
            stamp_duty=first.stamp_duty + second.stamp_duty,
            dp_charge=first.dp_charge + second.dp_charge,
            gst=first.gst + second.gst,
            total=first.total + second.total,
        )

    def assumptions(self) -> dict[str, float]:
        """The rates, for the report. Every value is an assumption."""
        return {
            "brokerage_pct": self.brokerage_pct,
            "brokerage_cap_rupees": self.brokerage_cap_rupees,
            "stt_buy_pct": self.stt_buy_pct,
            "stt_sell_pct": self.stt_sell_pct,
            "exchange_transaction_pct": self.exchange_transaction_pct,
            "sebi_turnover_pct": self.sebi_turnover_pct,
            "dp_charge_rupees": self.dp_charge_rupees,
            "stamp_duty_buy_pct": self.stamp_duty_buy_pct,
            "gst_pct": self.gst_pct,
        }


@dataclass(frozen=True, slots=True)
class ChargeBreakdown:
    """One leg's charges, or a round trip's, itemised."""

    notional: float = 0.0
    brokerage: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stt: float = 0.0
    stamp_duty: float = 0.0
    dp_charge: float = 0.0
    gst: float = 0.0
    total: float = 0.0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class BacktestError(RuntimeError):
    """Raised for a run that cannot honestly be started.

    Refusing loudly is the whole policy: a run that quietly substituted a
    different model, a different cost model, or a truncated window is worse
    than a run that does not happen.
    """


class ExecutionLimits(BaseModel):
    """The frozen `execution` section, which is this harness's risk policy.

    Defaults are copied from `core/config/config.go`'s `Validate`, including
    the ones it leaves alone. `capital` in particular is **not** defaulted by Go
    — it stays 0 when the key is absent — so it defaults to 0 here too, and
    `load_execution_limits` refuses a run that would divide by it.

    `app.config.Config` does not model this section (`extra="ignore"`: the
    inference service has no business holding a risk policy), so this module
    reads the same frozen YAML directly. That is a real gap in `app.config`
    rather than a private schema, and it is reported in the component writeup.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    capital: float = 0.0
    debounce_ttl_seconds: int = 300
    idempotency_ttl_seconds: int = 86400
    max_position_pct: float = 10.0
    max_open_positions: int = 10
    max_drawdown_pct: float = 5.0
    daily_loss_limit: float = 25000.0
    max_order_quantity: int = 500
    market_holidays: list[str] = Field(default_factory=list)

    def drawdown_pct_of(self, peak: float, equity: float) -> float:
        """Drawdown from `peak` as a percentage, matching check 3's formula."""
        if peak <= 0.0:
            return 100.0
        return (peak - equity) / peak * 100.0

    def position_cap(self, equity: float) -> float:
        """The largest order value `max_position_pct` allows at this equity."""
        return max(equity, 0.0) * self.max_position_pct / 100.0


def load_execution_limits(path: str | Path | None = None) -> ExecutionLimits:
    """Read the `execution` section out of the same config the Go services read.

    Args:
        path: An explicit config path, else `MFT_CONFIG`, else the default.
            Resolved with `app.config.resolve_path`, so a run from
            `services/inference` finds the repository-root file the same way
            `make run-inference` does.

    Raises:
        BacktestError: If the file cannot be read or parsed, if the section is
            not a mapping, or if the values are not internally consistent.
    """
    resolved = resolve_path(path)
    try:
        document = yaml.safe_load(resolved.read_text()) or {}
    except OSError as err:
        raise BacktestError(f"cannot read config {resolved}: {err}") from err
    except yaml.YAMLError as err:
        raise BacktestError(f"cannot parse config {resolved}: {err}") from err

    if not isinstance(document, dict):
        raise BacktestError(f"config {resolved} is not a YAML mapping")
    section = document.get("execution") or {}
    if not isinstance(section, dict):
        raise BacktestError(f"config {resolved}: 'execution' is not a YAML mapping")

    try:
        limits = ExecutionLimits.model_validate(section)
    except ValidationError as err:
        raise BacktestError(f"invalid execution section in {resolved}: {err}") from err

    if limits.capital <= 0.0:
        raise BacktestError(
            f"config {resolved} has execution.capital = {limits.capital}. "
            "core/config/config.go does not default it, and every order would fail "
            "RISK_MAX_POSITION against zero equity. Set execution.capital to the rupees "
            "the account would actually trade."
        )
    if not 0.0 < limits.max_position_pct <= 100.0:
        raise BacktestError(
            f"execution.max_position_pct = {limits.max_position_pct} is outside (0, 100]"
        )
    if limits.max_open_positions < 0:
        raise BacktestError("execution.max_open_positions must not be negative")
    if limits.debounce_ttl_seconds < 0:
        raise BacktestError("execution.debounce_ttl_seconds must not be negative")
    for holiday in limits.market_holidays:
        try:
            date.fromisoformat(holiday)
        except ValueError as err:
            raise BacktestError(
                f"execution.market_holidays: {holiday!r} is not a YYYY-MM-DD date"
            ) from err
    return limits


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


class Reason(str, Enum):
    """`docs/contracts.md` §6 rejection codes, verbatim.

    A `str` enum so a rejection serialises to the same word the execution
    service puts in its `{"error": ...}` body, and so a backtest refusal and a
    production refusal are greppable as the same thing.
    """

    MAX_POSITION = "RISK_MAX_POSITION"
    MAX_POSITIONS = "RISK_MAX_POSITIONS"
    MAX_DRAWDOWN = "RISK_MAX_DRAWDOWN"
    DAILY_LOSS = "RISK_DAILY_LOSS"
    DEBOUNCED = "RISK_DEBOUNCED"
    BAD_QUANTITY = "RISK_BAD_QUANTITY"
    MARKET_CLOSED = "RISK_MARKET_CLOSED"
    DUPLICATE = "RISK_DUPLICATE"


#: Each code's position in the contract's ordered list of checks.
REASON_CHECK: Final[dict[Reason, int]] = {
    Reason.MAX_POSITION: 1,
    Reason.MAX_POSITIONS: 2,
    Reason.MAX_DRAWDOWN: 3,
    Reason.DAILY_LOSS: 4,
    Reason.DEBOUNCED: 5,
    Reason.BAD_QUANTITY: 6,
    Reason.MARKET_CLOSED: 7,
    Reason.DUPLICATE: 8,
}


class BarOutcome(str, Enum):
    """What the harness did with one scored bar, before the risk gate.

    Named after `app.loop.Outcome`'s non-failure cases so the two are
    comparable in a log line: `below_threshold` and `no_direction` mean the
    same thing in the live loop and here.
    """

    BELOW_THRESHOLD = "below_threshold"
    NO_DIRECTION = "no_direction"
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    TRADED = "traded"


FeatureMode = Literal["rebuild", "incremental"]

#: Per-symbol decision-bar counts below this are still run; below the walk's own
#: first-decision index there is nothing to run at all.
MIN_USABLE_BARS: Final[int] = WARMUP_ROWS + MIN_CONTEXT_ROWS + 1


# --------------------------------------------------------------------------
# Phase 1 — the walk-forward, one symbol at a time
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """A bar that passed the threshold and has a side, awaiting the risk gate."""

    symbol: str
    index: int
    as_of: datetime
    exit_as_of: datetime
    price: float
    exit_price: float
    score: float
    side: Side
    quantity: int
    key: str

    @property
    def notional(self) -> float:
        """Order value at the reference price, which is what check 1 sees."""
        return self.quantity * self.price


@dataclass(slots=True)
class SymbolWalk:
    """Everything phase 1 produced for one symbol."""

    symbol: str
    candles: list[Candle]
    candidates: list[Candidate]
    bars: int
    scored: int
    outcomes: Counter[str] = field(default_factory=Counter)
    feature_seconds: float = 0.0
    model_seconds: float = 0.0

    def first_bar(self) -> datetime | None:
        """Timestamp of the first bar this symbol holds, if it holds any."""
        return self.candles[0].timestamp if self.candles else None



@dataclass(frozen=True, slots=True)
class WalkForward:
    """Scores every bar of one symbol, one bar at a time, with no lookahead.

    The loop is this module's contract with itself:

        for i in range(first_decision, len(candles) - horizon):
            context = candles[: i + 1]            # only up to and including i
            score   = model.predict(context, horizon)
            realised = (close[i + horizon] - close[i]) / close[i]   # the label

    `first_decision` is not `WARMUP_ROWS`. `FeatureTable.context` drops the
    warm-up rows, and `model.validate_context` then refuses fewer than
    `MIN_CONTEXT_ROWS + horizon` survivors. A model handed a shorter prompt
    answers from whatever it can still attend to, which is a confident number
    computed from too little — so the backtest starts where the live loop can
    actually trade, not where the table first has a predictable row.

    Args:
        builder: The same `FeatureBuilder` the live loop uses.
        predictor: The same `Predictor` the live loop uses. One per run, so the
            model is loaded once and every symbol shares it.
        inference: The frozen `inference` section: horizon, context rows,
            threshold, order quantity.
        mode: `rebuild` or `incremental`; see the module docstring.
    """

    builder: FeatureBuilder
    predictor: Predictor
    inference: InferenceConfig
    mode: FeatureMode = "rebuild"

    @property
    def horizon(self) -> int:
        """Bars ahead the label lives at."""
        return self.inference.horizon_bars

    @property
    def context_rows(self) -> int:
        """Rows handed to the model, capped at the model's own budget."""
        return min(self.inference.context_rows, MAX_CONTEXT_ROWS)

    @property
    def first_decision(self) -> int:
        """Index of the first bar with a long enough warmed context to score."""
        return WARMUP_ROWS + MIN_CONTEXT_ROWS + self.horizon - 1

    async def run(self, symbol: str, candles: Sequence[Candle]) -> SymbolWalk:
        """Score one symbol's series bar by bar.

        Returns rather than raises for input the builder rejects — a window
        under the warm-up, or one the model cannot prompt from — because one bad
        symbol must not take the whole run down. The reason is logged and the
        symbol contributes nothing.
        """
        horizon = self.horizon
        rows = self.context_rows
        wanted = self.first_decision
        last = len(candles) - horizon
        walk = SymbolWalk(
            symbol=symbol, candles=list(candles), candidates=[], bars=len(candles), scored=0
        )
        if last <= wanted:
            logger.warning(
                "%s: %d candles is short of the %d bars the first decision needs "
                "(warm-up %d + model floor %d + horizon %d); nothing was scored",
                symbol,
                len(candles),
                wanted + 1,
                WARMUP_ROWS,
                MIN_CONTEXT_ROWS,
                horizon,
            )
            return walk

        frame: pd.DataFrame | None = None
        if self.mode == "incremental":
            try:
                frame = self.builder.build(candles).frame
            except FeatureError as err:
                logger.warning("%s: feature build failed: %s", symbol, err)
                return walk

        for index in range(wanted, last):
            built = time.perf_counter()
            if frame is None:
                try:
                    table = self.builder.build(candles[: index + 1])
                except FeatureError as err:
                    # A slice of a series the full build already accepted, so
                    # this is a belt-and-braces skip rather than an expected path.
                    logger.warning("%s at bar %d: feature build failed: %s", symbol, index, err)
                    break
                context = table.context(rows)
            else:
                context = context_from_frame(frame, index, rows)
            walk.feature_seconds += time.perf_counter() - built

            model_start = time.perf_counter()
            score = await self.predictor.predict(context, horizon)
            walk.model_seconds += time.perf_counter() - model_start

            walk.scored += 1
            if abs(score) < self.inference.score_threshold:
                walk.outcomes[BarOutcome.BELOW_THRESHOLD.value] += 1
                continue
            try:
                side = side_for_score(score)
            except ValueError as err:
                # A score of exactly 0.0 with a threshold of 0, or something
                # outside [-1, 1]. Refused, not coerced into a direction.
                logger.debug("%s at bar %d: %s", symbol, index, err)
                walk.outcomes[BarOutcome.NO_DIRECTION.value] += 1
                continue

            as_of = candles[index].timestamp
            walk.candidates.append(
                Candidate(
                    symbol=symbol,
                    index=index,
                    as_of=as_of,
                    exit_as_of=candles[index + horizon].timestamp,
                    price=candles[index].close,
                    # The label. Read here and nowhere else, and only after the
                    # score already exists.
                    exit_price=candles[index + horizon].close,
                    score=score,
                    side=side,
                    quantity=self.inference.order_quantity,
                    key=idempotency_key(symbol, side, as_of),
                )
            )
            walk.outcomes[BarOutcome.CANDIDATE.value] += 1

        logger.info(
            "%s: %d bars, %d scored, %d candidates, features %.2fs, model %.2fs",
            symbol,
            walk.bars,
            walk.scored,
            len(walk.candidates),
            walk.feature_seconds,
            walk.model_seconds,
        )
        return walk


def context_from_frame(frame: pd.DataFrame, upto: int, rows: int) -> pd.DataFrame:
    """The model prompt for bar `upto`, sliced out of an already-built table.

    Equivalent, value for value, to
    `FeatureBuilder.build(candles[: upto + 1]).context(rows)`, which
    `test_backtest.py` asserts rather than assumes. The slice truncates to
    `[: upto + 1]` *before* the warm-up is dropped, so a row derived from a
    candle after `upto` cannot reach the prompt even though it exists in the
    frame.
    """
    window = frame.iloc[: upto + 1].iloc[WARMUP_ROWS:]
    if len(window) > rows:
        window = window.iloc[-rows:]
    return window.reset_index(drop=True)


# --------------------------------------------------------------------------
# Phase 2 — the portfolio
# --------------------------------------------------------------------------


@dataclass(slots=True)
class OpenPosition:
    """A filled order waiting for its label bar."""

    candidate: Candidate
    entry_leg: ChargeBreakdown


@dataclass(frozen=True, slots=True)
class Trade:
    """One round trip: entered at a scored bar, exited `horizon` bars later.

    The exit price is the label the model was never shown. `net_pnl` and
    `net_return` include both legs' charges, because a return that does not is
    a return on a fiction.
    """

    symbol: str
    side: Side
    quantity: int
    score: float
    key: str
    entry_as_of: datetime
    exit_as_of: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    charges: float
    net_pnl: float
    net_return: float
    bars_held: int
    entry_leg: ChargeBreakdown
    exit_leg: ChargeBreakdown

    def to_json(self) -> dict[str, Any]:
        """The per-trade row, as it appears in the report and the JSON."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "score": round(self.score, 6),
            "idempotency_key": self.key,
            "entry_as_of": rfc3339(self.entry_as_of),
            "exit_as_of": rfc3339(self.exit_as_of),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "charges": round(self.charges, 4),
            "net_pnl": round(self.net_pnl, 4),
            "net_return_pct": round(self.net_return * 100.0, 6),
            "bars_held": self.bars_held,
            "entry_leg": round(self.entry_leg.total, 4),
            "exit_leg": round(self.exit_leg.total, 4),
        }


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One step of the portfolio's mark-to-market curve.

    The curve has three kinds of point: the opening capital, one per candidate
    timestamp, and a closing point after the last position is realised. The two
    ends exist so the curve starts and ends where the run does.
    """

    as_of: datetime
    equity: float
    open_positions: int
    cash: float


@dataclass(slots=True)
class Simulation:
    """Phase 2's output: the trades, the curve, and why the rest were refused."""

    trades: list[Trade] = field(default_factory=list)
    curve: list[EquityPoint] = field(default_factory=list)
    rejections: Counter[str] = field(default_factory=Counter)
    rejections_by_symbol: dict[str, Counter[str]] = field(default_factory=dict)
    outcome_totals: Counter[str] = field(default_factory=Counter)
    outcomes_by_symbol: dict[str, Counter[str]] = field(default_factory=dict)
    filled: int = 0
    final_equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    gross_pnl: float = 0.0
    charges: float = 0.0
    notional: float = 0.0
    exposed_steps: int = 0
    closed_at_end: int = 0
    drawdown_halted: bool = False

    def count_rejection(self, symbol: str, reason: Reason) -> None:
        """Record one risk refusal, globally and per symbol."""
        self.rejections[reason.value] += 1
        self.rejections_by_symbol.setdefault(symbol, Counter())[reason.value] += 1

    def count_outcome(self, symbol: str, outcome: BarOutcome) -> None:
        """Record one risk-gate outcome, globally and per symbol."""
        self.outcome_totals[outcome.value] += 1
        self.outcomes_by_symbol.setdefault(symbol, Counter())[outcome.value] += 1


#: Marks: per symbol, the bar timestamps and their closes, ascending.
Marks = dict[str, tuple[list[datetime], list[float]]]


def build_marks(walks: Sequence[SymbolWalk]) -> Marks:
    """The per-symbol mark series, from every bar and not only the candidates."""
    out: Marks = {}
    for walk in walks:
        if walk.candles:
            out[walk.symbol] = (
                [candle.timestamp for candle in walk.candles],
                [candle.close for candle in walk.candles],
            )
    return out


def mark_at(marks: Marks, symbol: str, moment: datetime) -> float:
    """The last close of `symbol` at or before `moment`.

    A symbol with no bar at `moment` is marked at its most recent close, which
    is the honest thing to do with a gap in the tape: the position is worth
    what the last thing anyone traded was worth. A symbol with no bar at all
    before `moment` falls back to its first close, so a run cannot divide into
    a mark of zero.
    """
    series = marks.get(symbol)
    if not series:
        return 0.0
    stamps, closes = series
    index = bisect.bisect_right(stamps, moment) - 1
    return closes[0] if index < 0 else closes[index]


def _debounce_key(candidate: Candidate) -> str:
    """`SYMBOL:SIDE`, the key `docs/contracts.md` §6 gives the debounce."""
    return f"{candidate.symbol}:{candidate.side}"


class Portfolio:
    """A single-account, single-currency simulation of the risk gate.

    Steps one merged timeline of candidate timestamps. At each timestamp:

    1. positions whose label bar has arrived are closed and their P&L realised;
    2. the IST day rolls over if the date changed, resetting the daily loss;
    3. each candidate at that timestamp goes through the seven checks of
       `docs/contracts.md` §6 in order, and the ones that pass become positions;
    4. equity is marked, the peak and drawdown are updated, and the curve gets a
       point.

    Exits run before entries at the same timestamp on purpose. A position opened
    at bar `i` with `horizon` 1 closes at bar `i + 1`, and the decision for bar
    `i + 1` should be able to use the cash it released; the reverse order would
    double-count a bar of capital and refuse a trade the live engine would have
    accepted.
    """

    def __init__(self, limits: ExecutionLimits, charges: ChargeModel, *, zone: tzinfo) -> None:
        self._limits = limits
        self._charges = charges
        self._zone = zone
        self._cash = limits.capital
        self._peak = limits.capital
        self._open: list[OpenPosition] = []
        self._holidays = frozenset(limits.market_holidays)
        # Check 5: a SYMBOL:SIDE ledger with a TTL, as the contract words it.
        self._debounce: dict[str, datetime] = {}
        # Check 8: keyed by the same function the live loop uses, so the two
        # cannot disagree about what a key is.
        self._keys: dict[str, datetime] = {}
        self._current_day: date | None = None
        self._realised_today = 0.0

    def run(
        self,
        walks: Sequence[SymbolWalk],
        candidates_by_time: dict[datetime, list[Candidate]],
    ) -> Simulation:
        """Step the portfolio across every candidate timestamp."""
        marks = build_marks(walks)
        result = Simulation(peak_equity=self._limits.capital, final_equity=self._limits.capital)
        result.curve.append(
            EquityPoint(
                as_of=self._timeline_start(walks, candidates_by_time),
                equity=self._limits.capital,
                open_positions=0,
                cash=self._cash,
            )
        )

        for moment in sorted(candidates_by_time):
            self._roll_day(moment)
            self._close_arrived(moment, result)
            # The peak moves before the checks, because a winning exit at this
            # timestamp is equity and check 3 measures from the peak.
            self._peak = max(self._peak, self._equity(marks, moment))
            for candidate in sorted(candidates_by_time[moment], key=lambda c: c.symbol):
                self._consider(candidate, marks, result)

            equity = self._equity(marks, moment)
            self._peak = max(self._peak, equity)
            result.max_drawdown_pct = max(
                result.max_drawdown_pct, self._limits.drawdown_pct_of(self._peak, equity)
            )
            result.curve.append(
                EquityPoint(as_of=moment, equity=equity, open_positions=len(self._open),
                            cash=self._cash)
            )
            if self._open:
                result.exposed_steps += 1

        # A position opened on the last candidate bar exits after the timeline
        # ends. Its label bar is still in the data, so it is closed there rather
        # than left dangling: a run where the trade list and the equity curve
        # disagree by one leg is a run nobody can check. The curve gets a closing
        # point so it ends where the run does.
        tail = list(self._open)
        for position in sorted(tail, key=lambda p: (p.candidate.exit_as_of, p.candidate.symbol)):
            self._realise(position, result)
        self._open = []
        result.closed_at_end = len(tail)
        if tail:
            result.curve.append(
                EquityPoint(
                    as_of=max(
                        max(p.candidate.exit_as_of for p in tail),
                        result.curve[-1].as_of,
                    ),
                    equity=self._cash,
                    open_positions=0,
                    cash=self._cash,
                )
            )

        result.trades.sort(key=lambda t: (t.entry_as_of, t.symbol))
        result.final_equity = self._cash
        result.peak_equity = max(self._peak, result.final_equity)
        result.max_drawdown_pct = max(
            result.max_drawdown_pct,
            self._limits.drawdown_pct_of(result.peak_equity, result.final_equity),
        )
        result.drawdown_halted = result.max_drawdown_pct > self._limits.max_drawdown_pct
        return result

    # -- the gate ---------------------------------------------------------

    def _consider(self, candidate: Candidate, marks: Marks, result: Simulation) -> None:
        """Run the contract's checks; fill if all of them pass."""
        equity = self._equity(marks, candidate.as_of)
        refusal = self._check(candidate, equity)
        if refusal is not None:
            result.count_rejection(candidate.symbol, refusal)
            result.count_outcome(candidate.symbol, BarOutcome.REJECTED)
            return

        is_buy = candidate.side == SIDE_BUY
        entry = self._charges.leg(is_buy=is_buy, quantity=candidate.quantity, price=candidate.price)
        # A BUY spends cash; a SELL raises it, less its own charges. The short
        # proceeds count as fully available: no margin, no borrow. Both are
        # optimistic and both are said so in the report.
        self._cash += (
            -candidate.notional - entry.total if is_buy else candidate.notional - entry.total
        )
        self._open.append(OpenPosition(candidate=candidate, entry_leg=entry))
        self._debounce[_debounce_key(candidate)] = candidate.as_of
        self._keys[candidate.key] = candidate.as_of
        result.filled += 1
        result.count_outcome(candidate.symbol, BarOutcome.TRADED)

    def _check(self, candidate: Candidate, equity: float) -> Reason | None:
        """The seven checks of `docs/contracts.md` §6, in order. First one wins."""
        limits = self._limits

        # 1. max_position_pct
        if candidate.notional > limits.position_cap(equity):
            return Reason.MAX_POSITION
        # 2. max_positions. The contract's Portfolio is a map keyed by symbol, so
        #    this counts distinct symbols; a second leg on an already-open
        #    symbol does not consume a slot.
        open_symbols = {p.candidate.symbol for p in self._open}
        if candidate.symbol not in open_symbols and len(open_symbols) >= limits.max_open_positions:
            return Reason.MAX_POSITIONS
        # 3. max_drawdown_pct
        if limits.drawdown_pct_of(self._peak, equity) > limits.max_drawdown_pct:
            return Reason.MAX_DRAWDOWN
        # 4. daily_loss_limit
        if self._realised_today < -limits.daily_loss_limit:
            return Reason.DAILY_LOSS
        # 5. debounce, per symbol and side
        previous = self._debounce.get(_debounce_key(candidate))
        if previous is not None and 0.0 <= (candidate.as_of - previous).total_seconds() < (
            limits.debounce_ttl_seconds
        ):
            return Reason.DEBOUNCED
        # 6. quantity. Lot size is not in the frozen config, so the
        #    `qty % lot_size` term of the contract is not applied here; see
        #    RISK_DRIFT_NOTES.
        if not 1 <= candidate.quantity <= limits.max_order_quantity:
            return Reason.BAD_QUANTITY
        # 7. market_hours
        if not self._is_market_open(candidate.as_of):
            return Reason.MARKET_CLOSED
        # 8. duplicate idempotency key
        seen = self._keys.get(candidate.key)
        if seen is not None and 0.0 <= (candidate.as_of - seen).total_seconds() < (
            limits.idempotency_ttl_seconds
        ):
            return Reason.DUPLICATE
        return None

    def _is_market_open(self, as_of: datetime) -> bool:
        """09:15–15:30 IST on a weekday that is not a configured holiday."""
        local = as_of.astimezone(self._zone)
        if local.weekday() >= 5:
            return False
        if local.date().isoformat() in self._holidays:
            return False
        return SESSION_OPEN_MINUTE <= local.hour * 60 + local.minute < (
            SESSION_OPEN_MINUTE + SESSION_BARS_PER_DAY
        )

    # -- accounting -------------------------------------------------------

    def _roll_day(self, moment: datetime) -> None:
        """Reset the daily realised loss when the IST date changes.

        Only on a change. Resetting on every timestamp would make check 4 a
        no-op, because a one-bar horizon realises the previous fill and then
        starts counting from zero again on the very next bar.
        """
        local_day = moment.astimezone(self._zone).date()
        if self._current_day is None:
            self._current_day = local_day
            return
        if local_day != self._current_day:
            logger.debug(
                "day rolled %s -> %s with %.2f realised",
                self._current_day,
                local_day,
                self._realised_today,
            )
            self._realised_today = 0.0
        self._current_day = local_day

    def _close_arrived(self, moment: datetime, result: Simulation) -> None:
        """Realise every position whose label bar has arrived by `moment`."""
        still_open: list[OpenPosition] = []
        for position in sorted(
            self._open, key=lambda p: (p.candidate.exit_as_of, p.candidate.symbol)
        ):
            if position.candidate.exit_as_of > moment:
                still_open.append(position)
            else:
                self._realise(position, result)
        self._open = still_open

    def _realise(self, position: OpenPosition, result: Simulation) -> None:
        """Close one position at its label bar and book the P&L."""
        candidate = position.candidate
        direction = 1.0 if candidate.side == SIDE_BUY else -1.0
        exit_leg = self._charges.leg(
            is_buy=candidate.side == SIDE_SELL,
            quantity=candidate.quantity,
            price=candidate.exit_price,
        )
        gross = direction * (candidate.exit_price - candidate.price) * candidate.quantity
        charges = position.entry_leg.total + exit_leg.total
        net = gross - charges
        self._cash += direction * candidate.exit_price * candidate.quantity - exit_leg.total
        self._realised_today += net

        result.trades.append(
            Trade(
                symbol=candidate.symbol,
                side=candidate.side,
                quantity=candidate.quantity,
                score=candidate.score,
                key=candidate.key,
                entry_as_of=candidate.as_of,
                exit_as_of=candidate.exit_as_of,
                entry_price=candidate.price,
                exit_price=candidate.exit_price,
                gross_pnl=gross,
                charges=charges,
                net_pnl=net,
                net_return=net / candidate.notional if candidate.notional > 0.0 else 0.0,
                bars_held=max(
                    1, round((candidate.exit_as_of - candidate.as_of).total_seconds() / 60.0)
                ),
                entry_leg=position.entry_leg,
                exit_leg=exit_leg,
            )
        )
        result.gross_pnl += gross
        result.charges += charges
        result.notional += position.entry_leg.notional + exit_leg.notional

    def _equity(self, marks: Marks, moment: datetime) -> float:
        """Cash plus every open position marked at the last known close."""
        total = self._cash
        for position in self._open:
            mark = mark_at(marks, position.candidate.symbol, moment)
            direction = 1.0 if position.candidate.side == SIDE_BUY else -1.0
            total += direction * position.candidate.quantity * mark
        return total

    def _timeline_start(
        self, walks: Sequence[SymbolWalk], candidates_by_time: dict[datetime, list[Candidate]]
    ) -> datetime:
        """Where the curve begins: the first bar, or the first candidate."""
        if candidates_by_time:
            return min(candidates_by_time)
        starts = [walk.first_bar() for walk in walks if walk.first_bar() is not None]
        return min(starts) if starts else datetime.now(timezone.utc)  # pragma: no cover


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

SYNTHETIC_DISCLAIMER: Final[str] = (
    "SYNTHETIC INPUT. This run used a geometric-Brownian-motion series generated by "
    "app/synth.py. It has no relationship to any real market: no overnight gaps, no "
    "cross-symbol correlation, no news, no halts, and volume that does not confirm "
    "price moves. Every return, Sharpe and hit rate below measures the plumbing of the "
    "feature -> model -> decision -> risk chain, and nothing else. It is NOT evidence of "
    "an edge, and no number here should be quoted as one. A model that appears to "
    "forecast this data is forecasting a random number generator whose parameters this "
    "module chose."
)

PARQUET_DISCLAIMER: Final[str] = (
    "Stored candles from the Parquet cold store, scored through the same feature, model, "
    "decision and risk path the live loop uses. The path is validated; the result is not. "
    "A walk-forward over one history is not an out-of-sample estimate, the costs are "
    "assumptions rather than a broker's published schedule, and 'BUY is long, SELL is "
    "short, both held horizon bars' is an intraday model, not delivery buy-and-hold."
)

CHARGE_DISCLAIMER: Final[str] = (
    "Every charge rate in this run is an ASSUMPTION baked into app/backtest.py, not a "
    "quote from any broker. docs/contracts.md section 8 declares no charge keys, so a run "
    "cannot know this broker's schedule. Re-verify every rate against your broker's "
    "current contract before believing anything below."
)

POSITION_MODEL_NOTE: Final[str] = (
    "Position model: one order is one round trip over horizon bars. BUY is long, SELL is "
    "short, which matches intraday (MIS) execution and is broker.product's default. It is "
    "NOT delivery (CNC) buy-and-hold, which would carry a position across sessions. A "
    "short is treated as fully covered by execution.capital with no margin haircut and no "
    "borrow cost, which flatters any strategy that leans on shorts."
)


@dataclass(frozen=True, slots=True)
class Metrics:
    """The headline numbers, all net of charges unless the name says gross."""

    trades: int
    wins: int
    losses: int
    hit_rate: float
    gross_return: float
    net_return: float
    mean_trade_return: float
    median_trade_return: float
    best_trade_return: float
    worst_trade_return: float
    sharpe: float
    sharpe_per_bar: float
    bars_per_year: int
    max_drawdown_pct: float
    turnover: float
    notional: float
    exposure_pct: float
    charges: float
    charges_per_trade: float
    gross_pnl: float
    net_pnl: float
    position_steps: int
    steps: int

    def to_json(self) -> dict[str, Any]:
        """Plain data, for the JSON artifact."""
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "hit_rate": round(self.hit_rate, 6),
            "gross_return_pct": round(self.gross_return * 100.0, 6),
            "net_return_pct": round(self.net_return * 100.0, 6),
            "mean_trade_return_pct": round(self.mean_trade_return * 100.0, 6),
            "median_trade_return_pct": round(self.median_trade_return * 100.0, 6),
            "best_trade_return_pct": round(self.best_trade_return * 100.0, 6),
            "worst_trade_return_pct": round(self.worst_trade_return * 100.0, 6),
            "sharpe_annualised": round(self.sharpe, 6),
            "sharpe_per_bar": round(self.sharpe_per_bar, 6),
            "bars_per_year": self.bars_per_year,
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "turnover_x_capital": round(self.turnover, 6),
            "traded_notional": round(self.notional, 4),
            "exposure_pct": round(self.exposure_pct, 6),
            "charges_total": round(self.charges, 4),
            "charges_per_trade": round(self.charges_per_trade, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "net_pnl": round(self.net_pnl, 4),
            "position_steps": self.position_steps,
            "equity_steps": self.steps,
        }


@dataclass(frozen=True, slots=True)
class SymbolSummary:
    """One symbol's row of the per-symbol breakdown."""

    symbol: str
    bars: int
    scored: int
    candidates: int
    trades: int
    wins: int
    hit_rate: float
    gross_pnl: float
    charges: float
    net_pnl: float
    net_return_on_capital: float
    rejections: dict[str, int]
    outcomes: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "bars_scored": self.scored,
            "candidates": self.candidates,
            "trades": self.trades,
            "wins": self.wins,
            "hit_rate": round(self.hit_rate, 6),
            "gross_pnl": round(self.gross_pnl, 4),
            "charges": round(self.charges, 4),
            "net_pnl": round(self.net_pnl, 4),
            "net_return_on_capital_pct": round(self.net_return_on_capital * 100.0, 6),
            "rejections": dict(sorted(self.rejections.items())),
            "outcomes": dict(sorted(self.outcomes.items())),
        }


@dataclass(frozen=True, slots=True)
class Timing:
    """What the model and the features cost, which decides whether this is liveable."""

    total_seconds: float
    feature_seconds: float
    model_seconds: float
    bars_scored: int
    symbols: int

    @property
    def model_ms_per_bar(self) -> float:
        return self.model_seconds / self.bars_scored * 1000.0 if self.bars_scored else 0.0

    @property
    def feature_ms_per_bar(self) -> float:
        return self.feature_seconds / self.bars_scored * 1000.0 if self.bars_scored else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "total_seconds": round(self.total_seconds, 4),
            "feature_seconds": round(self.feature_seconds, 4),
            "model_seconds": round(self.model_seconds, 4),
            "bars_scored": self.bars_scored,
            "model_ms_per_bar": round(self.model_ms_per_bar, 6),
            "feature_ms_per_bar": round(self.feature_ms_per_bar, 6),
            "symbols": self.symbols,
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything a run produced. `to_json` is the machine-readable artifact."""

    metrics: Metrics
    per_symbol: list[SymbolSummary]
    trades: list[Trade]
    curve: list[EquityPoint]
    rejections: dict[str, int]
    outcomes: dict[str, int]
    timing: Timing
    synthetic: bool
    model_selector: str
    model_name: str
    feature_mode: FeatureMode
    horizon: int
    context_rows: int
    threshold: float
    order_quantity: int
    limits: ExecutionLimits
    charges: ChargeModel
    symbols: list[str]
    bars_per_symbol: dict[str, int]
    seed: int | None
    data_source: str
    max_bars: int
    first_decision: datetime | None = None
    last_decision: datetime | None = None
    drawdown_halted: bool = False
    closed_after_timeline: int = 0
    fills: int = 0

    @property
    def disclaimer(self) -> str:
        """The sentence that has to survive a skim."""
        return SYNTHETIC_DISCLAIMER if self.synthetic else PARQUET_DISCLAIMER

    def to_json(self) -> dict[str, Any]:
        """The `--json` artifact.

        `synthetic` and `disclaimer` are top-level and unmissable, and the flag
        is repeated inside `data` because an artifact that gets split up loses
        its outermost keys first.
        """
        curve = self.curve
        sampled = _downsample(curve, MAX_JSON_CURVE_POINTS)
        return {
            "schema": "mft.backtest.v1",
            "generated_at": rfc3339(datetime.now(timezone.utc)),
            "synthetic": self.synthetic,
            "disclaimer": self.disclaimer,
            "charge_disclaimer": CHARGE_DISCLAIMER,
            "position_model": POSITION_MODEL_NOTE,
            "risk_drift_notes": list(RISK_DRIFT_NOTES),
            "data": {
                "source": self.data_source,
                "synthetic": self.synthetic,
                "seed": self.seed,
                "symbols": list(self.symbols),
                "bars_per_symbol": dict(sorted(self.bars_per_symbol.items())),
                "bars_per_year": BARS_PER_YEAR,
            },
            "config": {
                "model_selector": self.model_selector,
                "model": self.model_name,
                "feature_mode": self.feature_mode,
                "horizon_bars": self.horizon,
                "context_rows": self.context_rows,
                "score_threshold": self.threshold,
                "order_quantity": self.order_quantity,
                "max_bars": self.max_bars,
            },
            "execution_limits": self.limits.model_dump(),
            "charge_model": {
                "name": "indian_equity_cash_delivery",
                "assumptions": self.charges.assumptions(),
                "disclaimer": CHARGE_DISCLAIMER,
            },
            "metrics": self.metrics.to_json(),
            "per_symbol": [row.to_json() for row in self.per_symbol],
            "rejections": dict(sorted(self.rejections.items())),
            "outcomes": dict(sorted(self.outcomes.items())),
            "trades": [trade.to_json() for trade in self.trades],
            "equity_curve": [
                {
                    "as_of": rfc3339(point.as_of),
                    "equity": round(point.equity, 4),
                    "open_positions": point.open_positions,
                }
                for point in sampled
            ],
            "equity_curve_points": len(curve),
            "timing": self.timing.to_json(),
            "walk_forward": {
                "mode": self.feature_mode,
                "complexity": (
                    "O(n^2) per symbol" if self.feature_mode == "rebuild" else "O(n) per symbol"
                ),
                "label_is": f"candles[i + {self.horizon}].close, read only after the score exists",
                "first_decision_index": WARMUP_ROWS + MIN_CONTEXT_ROWS + self.horizon - 1,
                # Positions whose label bar falls after the last candidate
                # timestamp. They are closed at that bar, so the trade list and
                # the equity curve agree; the count is reported because a reader
                # comparing the curve's last point to the trade list should know
                # why there is a closing point at all.
                "positions_closed_after_the_last_candidate": self.closed_after_timeline,
                # Every fill becomes a trade, so these two must be equal. They
                # are both in the artifact precisely so a reader can check that
                # rather than take it on trust.
                "fills": self.fills,
                "trades": len(self.trades),
            },
        }


def _downsample(points: Sequence[EquityPoint], limit: int) -> list[EquityPoint]:
    """Thin the curve for the JSON, keeping the first and last points."""
    if len(points) <= limit:
        return list(points)
    stride = math.ceil(len(points) / limit)
    thinned = list(points[::stride])
    if thinned[-1] is not points[-1]:
        thinned.append(points[-1])
    return thinned


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


async def run_backtest(
    symbol_candles: dict[str, list[Candle]],
    *,
    config: Config,
    limits: ExecutionLimits,
    predictor: Predictor,
    mode: FeatureMode = "rebuild",
    max_bars: int = DEFAULT_MAX_BARS,
    charges: ChargeModel | None = None,
    synthetic: bool = False,
    seed: int | None = None,
    data_source: str = "parquet",
) -> BacktestResult:
    """Score every symbol walk-forward, then simulate the portfolio.

    Args:
        symbol_candles: Ascending candles per symbol, already filtered to the
            requested date window.
        config: The validated config, for `app.timezone` and `inference`.
        limits: The `execution` section, as the risk policy.
        predictor: A loaded `Predictor`. The caller owns loading and closing it.
        mode: Walk-forward mode; see the module docstring.
        max_bars: Hard cap on bars per symbol. A longer series is refused, not
            truncated.
        charges: Charge model. The default is the module's Indian equity
            assumptions; `ChargeModel.zero()` exists for comparison only.
        synthetic: Whether the candles came from `app.synth`. Recorded in every
            artifact this run produces.
        seed: The synthetic seed, when there was one.
        data_source: A label for the report: `parquet` or `synthetic`.

    Raises:
        BacktestError: If no symbol has any candles, or one has more than
            `max_bars`.
    """
    if not symbol_candles:
        raise BacktestError("no symbols to backtest; check inference.instruments or --symbol")

    charge_model = charges if charges is not None else ChargeModel()
    zone = trading_zone(config.app.timezone)
    walk_forward = WalkForward(
        builder=FeatureBuilder(zone), predictor=predictor, inference=config.inference, mode=mode
    )

    for symbol, candles in symbol_candles.items():
        if len(candles) > max_bars:
            raise BacktestError(
                f"{symbol} has {len(candles)} bars, over the --max-bars limit of {max_bars}. The "
                f"'{mode}' walk-forward rebuilds the feature table at every bar, which is O(n^2): "
                f"about "
                f"{projected_rebuild_minutes(len(candles), config.inference.horizon_bars):.1f} "
                "minutes of feature building alone for this symbol. Raise --max-bars, narrow "
                "--from/--to, or use --feature-mode incremental, which produces identical numbers "
                "in O(n). Refusing to truncate: a silently shortened run is a different run."
            )

    started = time.perf_counter()
    walks = [
        await walk_forward.run(symbol, candles) for symbol, candles in symbol_candles.items()
    ]

    candidates_by_time: dict[datetime, list[Candidate]] = {}
    for walk in walks:
        for candidate in walk.candidates:
            candidates_by_time.setdefault(candidate.as_of, []).append(candidate)

    simulation = Portfolio(limits, charge_model, zone=zone).run(walks, candidates_by_time)
    elapsed = time.perf_counter() - started
    bars_scored = sum(walk.scored for walk in walks)
    outcomes = Counter()
    for walk in walks:
        outcomes.update(walk.outcomes)
    for outcome, count in simulation.outcome_totals.items():
        outcomes[outcome] += count

    decisions = [c.as_of for group in candidates_by_time.values() for c in group]
    return BacktestResult(
        metrics=_metrics(simulation, limits),
        per_symbol=_per_symbol(walks, simulation, limits),
        trades=simulation.trades,
        curve=simulation.curve,
        rejections=dict(sorted(simulation.rejections.items())),
        outcomes=dict(sorted(outcomes.items())),
        timing=Timing(
            total_seconds=elapsed,
            feature_seconds=sum(walk.feature_seconds for walk in walks),
            model_seconds=sum(walk.model_seconds for walk in walks),
            bars_scored=bars_scored,
            symbols=len(walks),
        ),
        synthetic=synthetic,
        model_selector=predictor.model_name,
        model_name=predictor.name,
        feature_mode=mode,
        horizon=config.inference.horizon_bars,
        context_rows=walk_forward.context_rows,
        threshold=config.inference.score_threshold,
        order_quantity=config.inference.order_quantity,
        limits=limits,
        charges=charge_model,
        symbols=list(symbol_candles),
        bars_per_symbol={symbol: len(candles) for symbol, candles in symbol_candles.items()},
        seed=seed,
        data_source=data_source,
        max_bars=max_bars,
        first_decision=min(decisions) if decisions else None,
        last_decision=max(decisions) if decisions else None,
        drawdown_halted=simulation.drawdown_halted,
        closed_after_timeline=simulation.closed_at_end,
        fills=simulation.filled,
    )


def projected_rebuild_minutes(bars: int, horizon: int) -> float:
    """Minutes of feature building a `rebuild` walk-forward would spend on `bars`.

    Sum of `i + 1` rows over the decision range, at the measured per-row cost.
    An estimate, and labelled as one in the message that uses it; the run
    reports its own measured time as soon as it finishes.
    """
    first = WARMUP_ROWS + MIN_CONTEXT_ROWS + horizon - 1
    last = max(bars - horizon, first)
    rows = sum(range(first + 1, last + 1)) if last > first else 0
    return rows * FEATURE_ROW_SECONDS / 60.0


def _metrics(simulation: Simulation, limits: ExecutionLimits) -> Metrics:
    """Headline numbers, from the trades and the equity curve."""
    trades = simulation.trades
    returns = sorted(trade.net_return for trade in trades)
    wins = sum(1 for trade in trades if trade.net_pnl > 0.0)
    net_pnl = simulation.final_equity - limits.capital
    steps = len(simulation.curve)
    return Metrics(
        trades=len(trades),
        wins=wins,
        losses=len(trades) - wins,
        hit_rate=wins / len(trades) if trades else 0.0,
        gross_return=simulation.gross_pnl / limits.capital if limits.capital > 0.0 else 0.0,
        net_return=net_pnl / limits.capital if limits.capital > 0.0 else 0.0,
        mean_trade_return=(sum(returns) / len(returns)) if returns else 0.0,
        median_trade_return=_median(returns),
        best_trade_return=returns[-1] if returns else 0.0,
        worst_trade_return=returns[0] if returns else 0.0,
        sharpe=sharpe_of(simulation.curve),
        sharpe_per_bar=sharpe_of(simulation.curve) / math.sqrt(BARS_PER_YEAR)
        if BARS_PER_YEAR > 0.0
        else 0.0,
        bars_per_year=BARS_PER_YEAR,
        max_drawdown_pct=simulation.max_drawdown_pct,
        turnover=simulation.notional / limits.capital if limits.capital > 0.0 else 0.0,
        notional=simulation.notional,
        exposure_pct=simulation.exposed_steps / steps if steps else 0.0,
        charges=simulation.charges,
        charges_per_trade=simulation.charges / len(trades) if trades else 0.0,
        gross_pnl=simulation.gross_pnl,
        net_pnl=net_pnl,
        position_steps=sum(point.open_positions for point in simulation.curve),
        steps=steps,
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2 == 1:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def sharpe_of(curve: Sequence[EquityPoint]) -> float:
    """Annualised Sharpe of the per-timestamp portfolio equity returns.

    Annualisation is `sqrt(BARS_PER_YEAR)` with `BARS_PER_YEAR` = 375 bars a
    session x 252 sessions, and the report states it so a reader can undo it. A
    flat curve has no dispersion and therefore no Sharpe; 0.0 on a flat curve is
    the honest answer where `inf` would be a lie.
    """
    returns = bar_returns(curve)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    deviation = math.sqrt(variance)
    if deviation <= 0.0:
        return 0.0
    return mean / deviation * math.sqrt(BARS_PER_YEAR)


def bar_returns(curve: Sequence[EquityPoint]) -> list[float]:
    """Per-timestamp portfolio equity returns.

    A non-positive starting equity is a diverged curve and yields no returns
    rather than a division by zero dressed up as a number.
    """
    out: list[float] = []
    for previous, current in zip(curve, curve[1:]):
        if previous.equity <= 0.0:
            return []
        out.append(current.equity / previous.equity - 1.0)
    return out


def _per_symbol(
    walks: Sequence[SymbolWalk], simulation: Simulation, limits: ExecutionLimits
) -> list[SymbolSummary]:
    """One row per symbol, in the order it was run."""
    rows: list[SymbolSummary] = []
    for walk in walks:
        trades = [t for t in simulation.trades if t.symbol == walk.symbol]
        wins = sum(1 for t in trades if t.net_pnl > 0.0)
        gross = sum(t.gross_pnl for t in trades)
        charges = sum(t.charges for t in trades)
        net = sum(t.net_pnl for t in trades)
        rows.append(
            SymbolSummary(
                symbol=walk.symbol,
                bars=walk.bars,
                scored=walk.scored,
                candidates=len(walk.candidates),
                trades=len(trades),
                wins=wins,
                hit_rate=wins / len(trades) if trades else 0.0,
                gross_pnl=gross,
                charges=charges,
                net_pnl=net,
                net_return_on_capital=net / limits.capital if limits.capital > 0.0 else 0.0,
                rejections=dict(
                    sorted(simulation.rejections_by_symbol.get(walk.symbol, {}).items())
                ),
                outcomes=dict(sorted(walk.outcomes.items())),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


def window_candles(
    candles: Sequence[Candle], start: datetime | None, end: datetime | None
) -> list[Candle]:
    """Keep the bars whose timestamp falls in `[start, end]`, inclusive.

    Both bounds are compared in UTC after conversion, so a caller passing
    `2026-01-01T00:00:00+05:30` and one passing `2025-12-31T18:30:00Z` select
    the same bars.
    """
    low = start.astimezone(timezone.utc) if start is not None else None
    high = end.astimezone(timezone.utc) if end is not None else None
    return [
        candle
        for candle in candles
        if (low is None or candle.timestamp.astimezone(timezone.utc) >= low)
        and (high is None or candle.timestamp.astimezone(timezone.utc) <= high)
    ]


async def load_candles(
    store: CandleStore, symbols: Sequence[str], limit: int
) -> dict[str, list[Candle]]:
    """Pull each symbol's tail from the store, dropping the ones with nothing.

    A symbol the broker has never streamed is an ordinary state, not a fault,
    so an empty read is logged and skipped rather than raised.
    """
    out: dict[str, list[Candle]] = {}
    for symbol in symbols:
        try:
            candles = await store.tail(symbol, limit)
        except Exception as err:  # noqa: BLE001 - one bad read is one symbol gone
            logger.warning("cannot read candles for %s: %s", symbol, err)
            continue
        if not candles:
            logger.warning("no stored candles for %s", symbol)
            continue
        out[symbol] = candles
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

RULE: Final[str] = "=" * 78
THIN: Final[str] = "-" * 78


def format_summary(
    result: BacktestResult, *, max_trades_shown: int = DEFAULT_MAX_TRADES_SHOWN
) -> str:
    """Render the human report. One block, logged once, never `print`ed.

    The order is deliberate: the disclaimer sits above the numbers rather than
    below them, and the headline says SYNTHETIC when it is synthetic, so a
    reader skimming the top three lines cannot miss it.
    """
    metrics = result.metrics
    lines: list[str] = []
    if result.synthetic:
        lines += [RULE, "MFT BACKTEST — SYNTHETIC INPUT. THIS IS NOT A STRATEGY RESULT.", RULE]
    else:
        lines += [RULE, "MFT BACKTEST — STORED CANDLES", RULE]
    lines += [
        "",
        _wrap(result.disclaimer),
        "",
        _wrap(CHARGE_DISCLAIMER),
        "",
        _wrap(POSITION_MODEL_NOTE),
        "",
        "RUN",
        THIN,
        f"  model              {result.model_selector} ({result.model_name})",
        f"  input              {result.data_source}"
        + (f", seed {result.seed}, SYNTHETIC" if result.synthetic else ", from Parquet"),
        f"  symbols            {', '.join(result.symbols) or '<none>'}",
        f"  bars               {_counts(result.bars_per_symbol)}",
        f"  horizon            {result.horizon} bar(s); "
        f"label = candles[i + {result.horizon}].close",
        f"  context rows       {result.context_rows} (after the {WARMUP_ROWS}-row warm-up)",
        f"  score threshold    {result.threshold:.4f}",
        f"  order quantity     {result.order_quantity} shares",
        f"  capital            Rs {result.limits.capital:,.2f}",
        f"  walk-forward       {result.feature_mode} — "
        + (
            "O(n^2) per symbol, one feature build per bar"
            if result.feature_mode == "rebuild"
            else "O(n) per symbol, identical numbers (proven in test_backtest.py)"
        ),
        f"  scored bars        {result.timing.bars_scored} "
        f"(first decision index {WARMUP_ROWS + MIN_CONTEXT_ROWS + result.horizon - 1})",
        f"  decision window    {_span(result.first_decision, result.last_decision)}",
        "",
        "RESULTS (net of charges unless the line says gross)",
        THIN,
        f"  trades             {metrics.trades}  ({metrics.wins} win / {metrics.losses} loss)",
        f"  hit rate           {_pct(metrics.hit_rate)}",
        f"  return, gross      {_pct(metrics.gross_return)}  (before charges, for comparison only)",
        f"  return, net        {_pct(metrics.net_return)}  "
        f"(Rs {metrics.net_pnl:+,.2f} on Rs {result.limits.capital:,.0f})",
        f"  per trade          mean {_pct(metrics.mean_trade_return)}  "
        f"median {_pct(metrics.median_trade_return)}  "
        f"best {_pct(metrics.best_trade_return)}  worst {_pct(metrics.worst_trade_return)}",
        f"  Sharpe, annualised {metrics.sharpe:.3f}   (per bar {metrics.sharpe_per_bar:+.4f}, "
        f"x sqrt({metrics.bars_per_year}))",
        f"                     annualisation is 375 bars/session x 252 sessions. A 1-minute",
        f"                     strategy's figure is enormous by construction; read the",
        f"                     per-bar number to see what the annualisation did.",
        f"  max drawdown       {metrics.max_drawdown_pct:.2f}%  "
        f"(limit {result.limits.max_drawdown_pct:.2f}%)"
        + ("  ← THE GATE HALTED TRADING" if result.drawdown_halted else ""),
        f"  turnover           {metrics.turnover:.2f}x capital  "
        f"(Rs {metrics.notional:,.0f} traded notional)",
        f"  exposure           {metrics.exposure_pct * 100.0:.2f}% of the timeline "
        "with a position open",
        f"  charges            Rs {metrics.charges:,.2f} total, "
        f"Rs {metrics.charges_per_trade:,.2f} per trade",
        f"  final equity       Rs {result.limits.capital + metrics.net_pnl:,.2f} "
        f"(capital + the sum of {metrics.trades} trade P&Ls)",
        "",
        "COST OF THE MODEL ITSELF",
        THIN,
        f"  wall time          {result.timing.total_seconds:.2f}s total",
        f"  model              {result.timing.model_seconds:.3f}s over "
        f"{result.timing.bars_scored} bars "
        f"= {result.timing.model_ms_per_bar:.3f} ms/bar",
        f"  features           {result.timing.feature_seconds:.3f}s = "
        f"{result.timing.feature_ms_per_bar:.3f} ms/bar  "
        + (
            "(walk-forward rebuild only; the live loop builds one bounded window a minute)"
            if result.feature_mode == "rebuild"
            else "(one build for the whole series)"
        ),
        f"  budget             a 1-minute cadence allows 60000 ms/bar; this run used "
        f"{result.timing.model_ms_per_bar + result.timing.feature_ms_per_bar:.3f} ms/bar",
        "",
        "PER SYMBOL",
        THIN,
        f"  {'symbol':<11}{'bars':>6}{'cand':>6}{'trades':>8}{'hit':>8}{'gross P&L':>12}"
        f"{'charges':>11}{'net P&L':>12}  reasons",
    ]
    for row in result.per_symbol:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(row.rejections.items())) or "-"
        lines.append(
            f"  {row.symbol:<11}{row.bars:>6}{row.candidates:>6}{row.trades:>8}"
            f"{row.hit_rate * 100.0:>7.1f}%{row.gross_pnl:>+12,.2f}{row.charges:>11,.2f}"
            f"{row.net_pnl:>+12,.2f}  {reasons}"
        )

    lines += ["", "PER-BAR OUTCOMES (before the risk gate)"]
    lines.append(
        "  "
        + (
            ", ".join(f"{k}={v}" for k, v in result.outcomes.items())
            if result.outcomes
            else "nothing"
        )
    )

    lines += ["", "RISK GATE — docs/contracts.md §6, in the contract's own order"]
    if result.rejections:
        ordered = sorted(
            result.rejections.items(),
            key=lambda kv: (REASON_CHECK.get(Reason(kv[0]), 99), kv[0]),
        )
        for code, count in ordered:
            lines.append(f"  {code:<22} {count:>7}   (check {REASON_CHECK[Reason(code)]})")
    else:
        lines.append("  nothing was rejected")
    lines += ["", "  these are the same reason codes the execution service returns"]
    for note in RISK_DRIFT_NOTES:
        lines.append(_wrap(f"- {note}", indent="    "))

    lines += ["", f"PER-TRADE TABLE ({len(result.trades)} trades, net of charges)"]
    if not result.trades:
        lines.append("  no trade was filled")
    else:
        lines.append(
            f"  {'entry':<21}{'exit':<21}{'sym':<10}{'side':<5}{'qty':>4}{'score':>8}"
            f"{'entry px':>11}{'exit px':>11}{'gross':>11}{'charges':>10}{'net':>11}{'net%':>9}"
        )
        for trade in result.trades[:max_trades_shown]:
            lines.append(
                f"  {rfc3339(trade.entry_as_of):<21}{rfc3339(trade.exit_as_of):<21}"
                f"{trade.symbol:<10}{trade.side:<5}{trade.quantity:>4}{trade.score:>+8.4f}"
                f"{trade.entry_price:>11.2f}{trade.exit_price:>11.2f}{trade.gross_pnl:>+11.2f}"
                f"{trade.charges:>10.2f}{trade.net_pnl:>+11.2f}{trade.net_return * 100.0:>+8.3f}%"
            )
        if len(result.trades) > max_trades_shown:
            lines.append(
                f"  ... {len(result.trades) - max_trades_shown} more; --json writes every trade"
            )

    lines += [
        "",
        "WHAT THIS RUN DOES NOT TELL YOU",
        THIN,
        "  * Out-of-sample performance. This is one history, scored once, by the",
        "    same model that would see it live. There is no held-out set, no",
        "    cross-validation and no parameter fitting, so a good number here is",
        "    not evidence of a good number anywhere else.",
        "  * Whether the model has any predictive power. On synthetic data it",
        "    provably does not: the series is a random walk. A positive Sharpe",
        "    here is a property of the cost and size assumptions, not of a signal.",
        "  * Anything about live execution. No fills, no slippage, no partial",
        "    rejects, no circuit limits, no broker downtime, no square-off.",
        "  * The real risk engine. The gate above is an offline re-implementation",
        "    of the checks that make sense without a broker; see the drift notes.",
        "",
        RULE,
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int = 76, indent: str = "  ") -> str:
    """Reflow a paragraph to `width`, every line carrying `indent`."""
    lines: list[str] = []
    current: str | None = None
    for word in text.split():
        if current is None:
            current = indent + word
        elif len(current) + 1 + len(word) > width:
            lines.append(current)
            current = indent + word
        else:
            current = f"{current} {word}"
    lines.append(current or indent.rstrip())
    return "\n".join(lines)


def _pct(fraction: float) -> str:
    return f"{fraction * 100.0:+.2f}%"


def _counts(bars_per_symbol: dict[str, int]) -> str:
    if not bars_per_symbol:
        return "<none>"
    return ", ".join(f"{symbol}={count}" for symbol, count in sorted(bars_per_symbol.items()))


def _span(first: datetime | None, last: datetime | None) -> str:
    if first is None or last is None:
        return "<no bar passed the threshold>"
    if first == last:
        return rfc3339(first)
    return f"{rfc3339(first)} .. {rfc3339(last)}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

EXIT_OK: Final[int] = 0
EXIT_REFUSED: Final[int] = 2


def build_parser() -> argparse.ArgumentParser:
    """The `python -m app.backtest` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.backtest",
        description=(
            "Replay stored or synthetic candles through the live feature, model, decision "
            "and risk path. Places no order and contacts nothing."
        ),
        epilog=(
            "Defaults to the heuristic model, because inference.model is tabfm in the frozen "
            "config and the TabFM weights are ~6.6 GB of non-commercial checkpoint. "
            "--model tabfm needs --allow-weights."
        ),
    )
    parser.add_argument("--config", default=None, help="config file (default: $MFT_CONFIG)")
    parser.add_argument(
        "--symbol", action="append", dest="symbols", metavar="SYM",
        help="instrument to run; repeatable. Default: every configured instrument.",
    )
    parser.add_argument("--from", dest="start", metavar="YYYY-MM-DD",
                        help="window start, inclusive, in IST")
    parser.add_argument("--to", dest="end", metavar="YYYY-MM-DD",
                        help="window end, inclusive of the whole day, in IST")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL,
                        help=f"model selector (default: {DEFAULT_MODEL}, not inference.model)")
    parser.add_argument(
        "--allow-weights", action="store_true",
        help=f"permit a weighted model such as {'/'.join(sorted(WEIGHTED_MODELS))}. "
             "It will load ~6.6 GB of non-commercial checkpoint.",
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="generate the input with app/synth.py instead of reading Parquet")
    parser.add_argument("--bars", type=int, default=2000, help="synthetic bars per symbol")
    parser.add_argument("--seed", type=int, default=20260929, help="synthetic generator seed")
    parser.add_argument("--start-price", type=float, default=DEFAULT_START_PRICE,
                        help="synthetic first price")
    parser.add_argument(
        "--feature-mode", choices=("rebuild", "incremental"), default="rebuild",
        help="rebuild (default, O(n^2), auditable) or incremental (O(n), identical numbers)",
    )
    parser.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS,
                        help=f"refuse a run longer than this per symbol "
                             f"(default: {DEFAULT_MAX_BARS})")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="write the full machine-readable report here")
    parser.add_argument("--log-level", default=None,
                        choices=("debug", "info", "warn", "error"))
    parser.add_argument("--max-trades-shown", type=int, default=DEFAULT_MAX_TRADES_SHOWN,
                        help="per-trade rows to print; --json always carries every trade")
    return parser


def parse_day(value: str | None, label: str) -> datetime | None:
    """Parse `YYYY-MM-DD` into midnight IST on that day.

    The zone matters: `--from 2026-01-01` has to mean 09:15 IST on the 1st, and
    a UTC reading of the same string would silently start the run six hours and
    thirty minutes early.
    """
    if value is None:
        return None
    try:
        day = date.fromisoformat(value)
    except ValueError as err:
        raise BacktestError(f"--{label} {value!r} is not a YYYY-MM-DD date") from err
    return datetime(day.year, day.month, day.day, tzinfo=trading_zone("Asia/Kolkata"))


def parse_day_end(value: str | None) -> datetime | None:
    """The last instant of the day named by `--to`, IST: 23:59:59.999999."""
    start = parse_day(value, "to")
    return None if start is None else start + timedelta(days=1) - timedelta(microseconds=1)


def plan_model(args: argparse.Namespace, config: Config) -> str:
    """Decide the model, refusing a weighted one that was not asked for.

    The default is `heuristic` and not `inference.model`, whatever the config
    says. A research tool that loaded 6.6 GB of weights because nobody passed a
    flag would be aimed squarely at the person who wanted a number quickly.
    """
    selector = (args.model or DEFAULT_MODEL).strip().lower()
    if selector in WEIGHTED_MODELS and not args.allow_weights:
        raise BacktestError(
            f"refusing to build model {selector!r}: it loads the TabFM checkpoint, ~6.6 GB, "
            "licensed non-commercial and non-production (Plan.md section 5). Re-run with "
            "--allow-weights if you have the memory and the licence, or use --model heuristic, "
            "which needs no weights and no licence."
        )
    if selector in WEIGHTED_MODELS:
        logger.warning("%s", WEIGHTS_WARNING)
    if config.inference.model in WEIGHTED_MODELS and selector != config.inference.model:
        logger.info(
            "config says inference.model: %s; running %s. The backtest does not follow that key "
            "by default — see the module docstring.",
            config.inference.model,
            selector,
        )
    return selector


def _synthetic_input(
    symbols: Sequence[str], args: argparse.Namespace, zone: str
) -> dict[str, list[Candle]]:
    if args.bars < 0:
        raise BacktestError(f"--bars must not be negative, got {args.bars}")
    logger.info(
        "synthetic input: seed=%d, %d bars per symbol — this proves the plumbing, not an edge",
        args.seed,
        args.bars,
    )
    spec = SynthSpec(bars=args.bars, seed=args.seed, start_price=args.start_price)
    return generate_many(symbols, spec, zone_name=zone)


async def _main_async(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    configure_logging(args.log_level or config.app.log_level)
    limits = load_execution_limits(args.config)

    symbols = [s.strip().upper() for s in (args.symbols or config.inference.instruments)]
    if not symbols:
        raise BacktestError(
            "no instruments to run: set inference.instruments in the config, or pass --symbol"
        )
    selector = plan_model(args, config)
    # Arguments are validated before the store is touched, so a reversed window
    # is reported as a reversed window rather than as an empty cold store.
    start = parse_day(args.start, "from")
    end = parse_day_end(args.end)
    if start is not None and end is not None and start > end:
        raise BacktestError(f"--from {args.start} is after --to {args.end}")
    if args.bars < 0 and args.synthetic:
        raise BacktestError(f"--bars must not be negative, got {args.bars}")
    if args.max_bars < 1:
        raise BacktestError(f"--max-bars must be at least 1, got {args.max_bars}")

    predictor = Predictor.from_name(selector)
    try:
        await predictor.load()
        logger.info("model loaded: selector=%s name=%s", selector, predictor.name)

        synthetic = bool(args.synthetic)
        if synthetic:
            raw = _synthetic_input(symbols, args, config.app.timezone)
        else:
            store = DuckDBCandleStore(candles_root(config.storage.data_dir))
            try:
                # `max_bars + 1` so a symbol with more history than the limit
                # arrives over the limit and is refused, rather than arriving
                # exactly at it and looking like a complete run.
                raw = await load_candles(store, symbols, max(args.max_bars, 1) + 1)
            finally:
                await store.aclose()
            if not raw:
                raise BacktestError(
                    f"no candles under {config.storage.data_dir}/candles for {', '.join(symbols)}. "
                    "Run ingestion to fill the store, or pass --synthetic to generate a series."
                )

        if start is not None or end is not None:
            windowed = {s: window_candles(c, start, end) for s, c in raw.items()}
            for symbol in sorted(s for s, c in windowed.items() if not c):
                logger.warning("%s: no bars in the requested window", symbol)
            raw = {s: c for s, c in windowed.items() if c}
            if not raw:
                raise BacktestError(f"no bars between {args.start} and {args.end} for any symbol")

        result = await run_backtest(
            raw,
            config=config,
            limits=limits,
            predictor=predictor,
            mode=args.feature_mode,
            max_bars=args.max_bars,
            synthetic=synthetic,
            seed=args.seed if synthetic else None,
            data_source="synthetic" if synthetic else "parquet",
        )
    finally:
        await predictor.aclose()

    logger.info("\n%s", format_summary(result, max_trades_shown=args.max_trades_shown))

    if args.json_path:
        path = Path(args.json_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result.to_json(), indent=2, allow_nan=False) + "\n")
        except OSError as err:
            raise BacktestError(f"cannot write {path}: {err}") from err
        logger.info("wrote %s (synthetic=%s)", path, result.synthetic)

    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m app.backtest`, and for the tests.

    Returns a process exit code rather than raising, so a refusal is a message
    and a status rather than a traceback.
    """
    args = build_parser().parse_args(argv)
    # Logging before the config is read, so a config error is still reported.
    configure_logging(args.log_level or "info")
    try:
        return asyncio.run(_main_async(args))
    except (BacktestError, ConfigError) as err:
        logger.error("backtest refused: %s", err)
        return EXIT_REFUSED
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.warning("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover - the `python -m app.backtest` path
    raise SystemExit(main())


__all__ = [
    "BARS_PER_YEAR",
    "BROKERAGE_CAP_RUPEES",
    "BROKERAGE_PCT",
    "CHARGE_DISCLAIMER",
    "DEFAULT_MAX_BARS",
    "DEFAULT_MODEL",
    "DP_CHARGE_RUPEES",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXCHANGE_TRANSACTION_PCT",
    "GST_PCT",
    "BacktestError",
    "BacktestResult",
    "BarOutcome",
    "Candidate",
    "ChargeBreakdown",
    "ChargeModel",
    "EquityPoint",
    "ExecutionLimits",
    "FeatureMode",
    "Metrics",
    "MIN_USABLE_BARS",
    "OpenPosition",
    "Portfolio",
    "Reason",
    "REASON_CHECK",
    "RISK_DRIFT_NOTES",
    "STAMP_DUTY_BUY_PCT",
    "STT_BUY_PCT",
    "STT_SELL_PCT",
    "SEBI_TURNOVER_PCT",
    "SymbolSummary",
    "SymbolWalk",
    "Timing",
    "Trade",
    "WalkForward",
    "WEIGHTED_MODELS",
    "WEIGHTS_WARNING",
    "build_marks",
    "context_from_frame",
    "format_summary",
    "load_candles",
    "load_execution_limits",
    "main",
    "mark_at",
    "parse_day",
    "parse_day_end",
    "plan_model",
    "projected_rebuild_minutes",
    "run_backtest",
    "sharpe_of",
    "window_candles",
]
