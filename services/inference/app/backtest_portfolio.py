"""Portfolio accounting and replay across every market-bar timestamp."""

from __future__ import annotations

import bisect
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backtest import (
        BarOutcome,
        Candidate,
        ChargeBreakdown,
        ChargeModel,
        ExecutionLimits,
        Reason,
        SymbolWalk,
    )

from .features import SESSION_OPEN_MINUTE
from .signals import SIDE_BUY, SIDE_SELL, Side, rfc3339
from .synth import SESSION_BARS_PER_DAY

logger = logging.getLogger("mft.inference.backtest.portfolio")


@dataclass(slots=True)
class OpenPosition:
    """A filled order waiting for its label bar."""

    candidate: Candidate
    entry_leg: ChargeBreakdown


@dataclass(frozen=True, slots=True)
class Trade:
    """One round trip: entered at a scored bar, exited `horizon` bars later."""

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
    """One step of the portfolio's mark-to-market curve."""

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
    """The last close of `symbol` at or before `moment`."""
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
    """A single-account, single-currency simulation of the risk gate."""

    def __init__(self, limits: ExecutionLimits, charges: ChargeModel, *, zone: tzinfo) -> None:
        self._limits = limits
        self._charges = charges
        self._zone = zone
        self._cash = limits.capital
        self._peak = limits.capital
        self._open: list[OpenPosition] = []
        self._holidays = frozenset(limits.market_holidays)

        self._debounce: dict[str, datetime] = {}


        self._keys: dict[str, datetime] = {}
        self._current_day: date | None = None
        self._realised_today = 0.0

    def run(
        self,
        walks: Sequence[SymbolWalk],
        candidates_by_time: dict[datetime, list[Candidate]],
    ) -> Simulation:
        """Step the portfolio across every market bar, including non-candidates."""
        marks = build_marks(walks)
        result = Simulation(peak_equity=self._limits.capital, final_equity=self._limits.capital)
        timeline = sorted(
            {c.timestamp for walk in walks for c in walk.candles}
            | set(candidates_by_time)
        )
        start = timeline[0] - timedelta(microseconds=1) if timeline else self._timeline_start(walks, candidates_by_time)
        result.curve.append(
            EquityPoint(
                as_of=start,
                equity=self._limits.capital,
                open_positions=0,
                cash=self._cash,
            )
        )

        for moment in timeline:
            self._roll_day(moment)
            self._close_arrived(moment, result)


            self._peak = max(self._peak, self._equity(marks, moment))
            for candidate in sorted(candidates_by_time.get(moment, []), key=lambda c: c.symbol):
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


    def _consider(self, candidate: Candidate, marks: Marks, result: Simulation) -> None:
        """Run the contract's checks; fill if all of them pass."""
        from .backtest import BarOutcome

        equity = self._equity(marks, candidate.as_of)
        refusal = self._check(candidate, equity)
        if refusal is not None:
            result.count_rejection(candidate.symbol, refusal)
            result.count_outcome(candidate.symbol, BarOutcome.REJECTED)
            return

        is_buy = candidate.side == SIDE_BUY
        entry = self._charges.leg(is_buy=is_buy, quantity=candidate.quantity, price=candidate.price)


        self._cash += (
            -candidate.notional - entry.total if is_buy else candidate.notional - entry.total
        )
        self._open.append(OpenPosition(candidate=candidate, entry_leg=entry))
        self._debounce[_debounce_key(candidate)] = candidate.as_of
        self._keys[candidate.key] = candidate.as_of
        result.filled += 1
        result.count_outcome(candidate.symbol, BarOutcome.TRADED)

    def _check(self, candidate: Candidate, equity: float) -> Reason | None:
        """The seven checks of `docs/contracts.md` §6, in order."""
        from .backtest import Reason

        limits = self._limits


        if candidate.notional > limits.position_cap(equity):
            return Reason.MAX_POSITION


        open_symbols = {p.candidate.symbol for p in self._open}
        if candidate.symbol not in open_symbols and len(open_symbols) >= limits.max_open_positions:
            return Reason.MAX_POSITIONS

        if limits.drawdown_pct_of(self._peak, equity) > limits.max_drawdown_pct:
            return Reason.MAX_DRAWDOWN

        if self._realised_today < -limits.daily_loss_limit:
            return Reason.DAILY_LOSS

        previous = self._debounce.get(_debounce_key(candidate))
        if previous is not None and 0.0 <= (candidate.as_of - previous).total_seconds() < (
            limits.debounce_ttl_seconds
        ):
            return Reason.DEBOUNCED


        if not 1 <= candidate.quantity <= limits.max_order_quantity:
            return Reason.BAD_QUANTITY

        if not self._is_market_open(candidate.as_of):
            return Reason.MARKET_CLOSED
        entry_day = candidate.as_of.astimezone(self._zone).date()
        exit_local = candidate.exit_as_of.astimezone(self._zone)
        if exit_local.date() != entry_day or not self._is_market_open(candidate.exit_as_of):
            return Reason.SESSION_CROSSING

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


    def _roll_day(self, moment: datetime) -> None:
        """Reset the daily realised loss when the IST date changes."""
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
