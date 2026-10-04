"""Replay stored candles with the shared feature, predictor and signal boundaries."""

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


WEIGHTED_MODELS: Final[frozenset[str]] = frozenset({"tabfm"})


DEFAULT_MODEL: Final[str] = "heuristic"


DEFAULT_MAX_BARS: Final[int] = 5000


MAX_JSON_CURVE_POINTS: Final[int] = 1000


FEATURE_ROW_SECONDS: Final[float] = 8.4e-6


DEFAULT_MAX_TRADES_SHOWN: Final[int] = 10


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
    "Intraday candidates whose horizon label crosses the local session/date boundary "
    "are refused with RISK_SESSION_CROSSING; this harness does not model broker square-off.",
    "No margin or borrow model: a SELL is covered by the whole of execution.capital "
    "and its proceeds are treated as fully available. That flatters any strategy "
    "that leans on shorts.",
)


BROKERAGE_PCT: Final[float] = 0.03


BROKERAGE_CAP_RUPEES: Final[float] = 20.0

STT_BUY_PCT: Final[float] = 0.10

STT_SELL_PCT: Final[float] = 0.10

EXCHANGE_TRANSACTION_PCT: Final[float] = 0.00297

SEBI_TURNOVER_PCT: Final[float] = 0.0001

DP_CHARGE_RUPEES: Final[float] = 15.93

STAMP_DUTY_BUY_PCT: Final[float] = 0.015

GST_PCT: Final[float] = 18.0


class ChargeModel(BaseModel):
    """Indian equity cash/delivery charges, per executed order."""

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
        """A charge model that charges nothing."""
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
        """Charges for one executed order."""
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
        """The rates, for the report."""
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


class BacktestError(RuntimeError):
    """Raised for a run that cannot honestly be started."""


class ExecutionLimits(BaseModel):
    """The frozen `execution` section, which is this harness's risk policy."""

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
    """Read the `execution` section out of the same config the Go services read."""
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


class Reason(str, Enum):
    """Contract rejection codes plus this simulator's session-boundary policy."""

    MAX_POSITION = "RISK_MAX_POSITION"
    MAX_POSITIONS = "RISK_MAX_POSITIONS"
    MAX_DRAWDOWN = "RISK_MAX_DRAWDOWN"
    DAILY_LOSS = "RISK_DAILY_LOSS"
    DEBOUNCED = "RISK_DEBOUNCED"
    BAD_QUANTITY = "RISK_BAD_QUANTITY"
    MARKET_CLOSED = "RISK_MARKET_CLOSED"
    DUPLICATE = "RISK_DUPLICATE"
    SESSION_CROSSING = "RISK_SESSION_CROSSING"


REASON_CHECK: Final[dict[Reason, int]] = {
    Reason.MAX_POSITION: 1,
    Reason.MAX_POSITIONS: 2,
    Reason.MAX_DRAWDOWN: 3,
    Reason.DAILY_LOSS: 4,
    Reason.DEBOUNCED: 5,
    Reason.BAD_QUANTITY: 6,
    Reason.MARKET_CLOSED: 7,
    Reason.DUPLICATE: 8,
    Reason.SESSION_CROSSING: 9,
}


class BarOutcome(str, Enum):
    """What the harness did with one scored bar, before the risk gate."""

    BELOW_THRESHOLD = "below_threshold"
    NO_DIRECTION = "no_direction"
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    TRADED = "traded"


FeatureMode = Literal["rebuild", "incremental"]


MIN_USABLE_BARS: Final[int] = WARMUP_ROWS + MIN_CONTEXT_ROWS + 1


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
    """Scores every bar of one symbol, one bar at a time, with no lookahead."""

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
        """Score one symbol's series bar by bar."""
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
    """The model prompt for bar `upto`, sliced out of an already-built table."""
    window = frame.iloc[: upto + 1].iloc[WARMUP_ROWS:]
    if len(window) > rows:
        window = window.iloc[-rows:]
    return window.reset_index(drop=True)


from .backtest_portfolio import (
    EquityPoint, Marks, OpenPosition, Portfolio, Simulation, Trade, build_marks, mark_at,
)


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
    """Everything a run produced."""

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
        """The `--json` artifact."""
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


                "positions_closed_after_the_last_candidate": self.closed_after_timeline,


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
    """Score every symbol walk-forward, then simulate the portfolio."""
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
    """Minutes of feature building a `rebuild` walk-forward would spend on `bars`."""
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
    """Annualised Sharpe of the per-timestamp portfolio equity returns."""
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
    """Per-timestamp portfolio equity returns."""
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


def window_candles(
    candles: Sequence[Candle], start: datetime | None, end: datetime | None
) -> list[Candle]:
    """Keep the bars whose timestamp falls in `[start, end]`, inclusive."""
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
    """Pull each symbol's tail from the store, dropping the ones with nothing."""
    out: dict[str, list[Candle]] = {}
    for symbol in symbols:
        try:
            candles = await store.tail(symbol, limit)
        except Exception as err:  # noqa: BLE001
            logger.warning("cannot read candles for %s: %s", symbol, err)
            continue
        if not candles:
            logger.warning("no stored candles for %s", symbol)
            continue
        out[symbol] = candles
    return out


RULE: Final[str] = "=" * 78
THIN: Final[str] = "-" * 78

def format_summary(
    result: BacktestResult, *, max_trades_shown: int = DEFAULT_MAX_TRADES_SHOWN
) -> str:
    """Render the human report through the focused reporting module."""
    from .backtest_reporting import format_summary as render

    return render(result, max_trades_shown=max_trades_shown)


EXIT_OK: Final[int] = 0
EXIT_REFUSED: Final[int] = 2

def build_parser() -> argparse.ArgumentParser:
    """The `python -m app.backtest` argument parser."""
    from .backtest_cli import build_parser as build

    return build()

def parse_day(value: str | None, label: str) -> datetime | None:
    """Parse a CLI day argument as midnight in the exchange timezone."""
    from .backtest_cli import parse_day as parse

    return parse(value, label)

def parse_day_end(value: str | None) -> datetime | None:
    """Parse a CLI end-day argument as its final instant."""
    from .backtest_cli import parse_day_end as parse

    return parse(value)

def plan_model(args: argparse.Namespace, config: Config) -> str:
    """Resolve the CLI model choice without loading it."""
    from .backtest_cli import plan_model as plan

    return plan(args, config)

def _synthetic_input(symbols: Sequence[str], args: argparse.Namespace, zone: str) -> dict[str, list[Candle]]:
    from .backtest_cli import _synthetic_input as generate

    return generate(symbols, args, zone)

async def _main_async(args: argparse.Namespace) -> int:
    from .backtest_cli import _main_async as run

    return await run(args)

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m app.backtest`, kept at its historical path."""
    from .backtest_cli import main as run

    return run(argv)


if __name__ == "__main__":  # pragma: no cover
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
