"""The 18-column feature table, ported from `core/features` (C3).

# Why this is a port and not a call

The feature math exists in Go (`core/features`, C3) and the model consumes a
`pandas.DataFrame`. The alternative to writing it a second time was:

  a) call a Go HTTP endpoint that returns the feature table;
  b) have ingestion or execution expose it;
  c) reimplement it here, pinned to the same golden vectors as the Go side.

(a) and (b) were rejected because neither endpoint exists, both are outside
this component's ownership, and both make the minute-close path depend on a Go
service being up and fast at precisely the moment every other service is busy
with the same minute close. Plan.md §4 chose pull precisely so that inference
survives the rest of the platform being down. Routing the feature table through
an HTTP hop would spend that decision to avoid ~200 lines of arithmetic.

So this is (c), and the price of (c) is drift, which is paid for with a golden
test instead of a comment. `tests/fixtures/feature_golden.json` holds the same
64-bar fixture and the same seven expected feature rows that
`core/features/golden_test.go` asserts, transcribed from that file. Both sides
assert against the same numbers to the same tolerance (1e-9 relative), and
C3's own comment records that its values were produced by a *third*
implementation written from docs/contracts.md §4 rather than from the Go code.
The vectors are therefore a property of the contract, not of either
implementation.

# What makes drift detectable rather than silent

Every feature here is a pure function of a fixed trailing window ending at its
own bar. None is recursive, none is seeded from an earlier row. C3 guarantees
that "the newest row of a 100-candle build and of a 100000-candle build are
bit-identical", and this port keeps the property, which is what lets inference
pull a bounded window at every minute close and get the same number a backfill
over the same minutes would have computed.

The arithmetic is written the way Go writes it, not the way numpy would.
`_mean` and `_stdev` sum left to right rather than calling `np.mean`: numpy uses
pairwise summation, whose rounding differs from Go's plain loop by around 1e-13
relative. That is far below anything a model acts on, but the golden test
compares at 1e-9 and a future edit that swaps in a vectorised sum would then
fail a parity test for a reason that looks like a bug.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from model import FEATURE_COLUMNS

from .candles import Candle

logger = logging.getLogger("mft.inference.features")

#: Rows at the head of a table that are context-only. Mirrors
#: `features.WarmupRows` in `core/features/features.go`, which is the longest
#: window in the schema (`ret_60`); row WARMUP_ROWS is the first row at which
#: every column is fully informed.
WARMUP_ROWS = 60

#: NSE cash session open, minutes past midnight in the trading zone. Anchors
#: `minute_of_session`; mirrors `features.SessionOpenMinute`.
SESSION_OPEN_MINUTE = 9 * 60 + 15

#: Trailing window lengths, in bars. Mirrors the constants of the same name in
#: `core/features/features.go`.
WINDOW_RET1 = 1
WINDOW_RET5 = 5
WINDOW_RET15 = 15
WINDOW_RET60 = 60
WINDOW_VOL5 = 5
WINDOW_VOL20 = 20
WINDOW_VOL_Z = 20
WINDOW_RSI = 14
WINDOW_SMA = 10

#: Asia/Kolkata as a fixed offset, used when the IANA database is unavailable.
#: India has observed UTC+05:30 without daylight saving since 1945, so this is
#: exact for every timestamp the platform can hold. C3's tests fall back the
#: same way; the zoneinfo database is simply not in `requirements.txt`.
IST_FIXED_OFFSET = timezone(timedelta(hours=5, minutes=30), "IST")


class FeatureError(RuntimeError):
    """Raised when a candle window cannot be turned into a feature table.

    Distinct from C5's `InvalidContextError`, which reports a table the *model*
    rejects. This one reports input the *builder* rejects: too few candles, a
    window mixing symbols, a malformed OHLC bar.
    """


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """A built feature table. `frame` is row-for-row aligned with the input."""

    columns: tuple[str, ...]
    frame: pd.DataFrame
    as_of: datetime

    @property
    def warmup_rows(self) -> int:
        """How many warm-up rows this table holds, and `context` therefore drops.

        Always `WARMUP_ROWS` for a table this builder was willing to build, but
        the property reports what was actually dropped rather than assuming it,
        so a caller showing a context can say how much of the window it did not
        see.
        """
        return min(WARMUP_ROWS, len(self.frame))

    def context(self, rows: int | None = None) -> pd.DataFrame:
        """Return the most recent fully-warmed `rows` rows as a model prompt.

        Warm-up rows are dropped rather than zero-padded into the prompt. They
        are context-only by contract, and a row of mostly zeros reads to a
        zero-shot model as a real market state in which nothing moved.
        """
        if rows is None:
            table = self.frame.iloc[WARMUP_ROWS:]
        else:
            if rows <= 0:
                raise FeatureError(f"context rows must be positive, got {rows}")
            table = self.frame.iloc[WARMUP_ROWS:]
            if len(table) > rows:
                table = table.iloc[-rows:]
        return table.reset_index(drop=True)


def trading_zone(name: str | None) -> tzinfo:
    """Resolve a configured zone name, mirroring C3's `time.LoadLocation` path.

    An unset name means UTC, which is the honest reading of an unset value and
    not a safe default: it makes both calendar features UTC-derived, and a UTC
    `minute_of_session` for the 09:15 IST open reads -345 instead of 0.
    """
    if not name:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        if name == "Asia/Kolkata":
            logger.warning("tz database has no %s; using the fixed +05:30 offset", name)
            return IST_FIXED_OFFSET
        raise FeatureError(f"unknown timezone {name!r}") from None


class FeatureBuilder:
    """Turns a candle window into the frozen 18-column table.

    Args:
        zone: Trading location for `minute_of_session` and `hour_of_day`.
            Candles are stored in UTC; these two features are calendar values.
    """

    def __init__(self, zone: str | tzinfo | None = "Asia/Kolkata") -> None:
        self._zone: tzinfo = zone if isinstance(zone, tzinfo) else trading_zone(zone)

    @property
    def zone(self) -> tzinfo:
        """The trading location this builder labels bars with."""
        return self._zone

    @property
    def columns(self) -> tuple[str, ...]:
        """The frozen column order, taken from C5 rather than restated here."""
        return FEATURE_COLUMNS

    def build(self, candles: Sequence[Candle]) -> FeatureTable:
        """Return one feature row per input candle, oldest first.

        Pure: no clock, no I/O, no configuration read, no randomness. Every
        timestamp it uses comes from the candles it was handed, which is what
        makes the table reproducible from a Parquet pull and comparable between
        a backtest and a live minute.
        """
        if len(candles) < WARMUP_ROWS:
            raise FeatureError(
                f"{len(candles)} candles is under the {WARMUP_ROWS}-bar warm-up, "
                "refusing to build a table with no predictable row"
            )
        _validate(candles)

        s = _series(candles)
        loc = self._zone

        # One accumulator per column, so the build never materialises a row
        # only to transpose it into the frame at the end.
        cols: dict[str, list[float]] = {name: [] for name in FEATURE_COLUMNS}

        for i, candle in enumerate(candles):
            row: dict[str, float] = {}

            # Returns. ret_1 is first defined at bar 1; ret_60 at bar 60, which
            # is exactly where the warm-up ends.
            if i >= WINDOW_RET1:
                row["ret_1"] = s.log_ret[i]
            if i >= WINDOW_RET5:
                row["ret_5"] = s.ret(WINDOW_RET5, i)
            if i >= WINDOW_RET15:
                row["ret_15"] = s.ret(WINDOW_RET15, i)
            if i >= WINDOW_RET60:
                row["ret_60"] = s.ret(WINDOW_RET60, i)

            # Realised volatility and its ratio. The 20-bar guard is stricter
            # than the 5-bar one it depends on, so vol_5 is already populated.
            if i >= WINDOW_VOL5:
                row["vol_5"] = s.vol(WINDOW_VOL5, i)
            if i >= WINDOW_VOL20:
                vol20 = s.vol(WINDOW_VOL20, i)
                row["vol_20"] = vol20
                row["vol_ratio"] = _div(row["vol_5"], vol20)

            # Candle geometry. Each reads only the bar it belongs to, so all of
            # them are defined from bar 0.
            row["range_1"] = s.range1(i)
            row["body_1"] = s.body1(i)
            row["upper_wick_1"] = s.upper_wick1(i)
            row["lower_wick_1"] = s.lower_wick1(i)
            row["spread_proxy"] = s.spread_proxy(i)

            # Volume. The 20-bar window counts the current bar, the same window
            # volume_ratio's moving average does.
            if i >= WINDOW_VOL_Z - 1:
                row["volume_z_20"] = s.volume_z20(i)
                row["volume_ratio"] = s.volume_ratio(i)

            if i >= WINDOW_RSI:
                row["momentum_rsi_14"] = s.rsi14(i)
            if i >= WINDOW_SMA - 1:
                row["sma_gap_10"] = s.sma_gap10(i)

            stamp = candle.timestamp.astimezone(loc)
            row["minute_of_session"] = float(
                stamp.hour * 60 + stamp.minute - SESSION_OPEN_MINUTE
            )
            row["hour_of_day"] = float(stamp.hour)

            for name in FEATURE_COLUMNS:
                cols[name].append(_sanitize(row.get(name, 0.0)))

        frame = pd.DataFrame(cols, columns=list(FEATURE_COLUMNS), dtype="float64")
        return FeatureTable(
            columns=FEATURE_COLUMNS, frame=frame, as_of=candles[-1].timestamp
        )


class _Series:
    """Columnar view of a candle window, mirroring `core/features/columns.go`.

    The OHLCV values are copied into parallel lists once per build because a
    feature reads them by offset rather than by value. Index 0 of `log_ret` and
    `change` is not meaningful — neither exists without a previous bar — and
    every read of those two is guarded by a window starting at index 1.
    """

    __slots__ = ("opens", "highs", "lows", "closes", "volumes", "log_ret", "change")

    def __init__(self, candles: Sequence[Candle]) -> None:
        n = len(candles)
        self.opens = [0.0] * n
        self.highs = [0.0] * n
        self.lows = [0.0] * n
        self.closes = [0.0] * n
        self.volumes = [0.0] * n
        self.log_ret = [0.0] * n
        self.change = [0.0] * n
        for i, c in enumerate(candles):
            self.opens[i] = c.open
            self.highs[i] = c.high
            self.lows[i] = c.low
            self.closes[i] = c.close
            self.volumes[i] = float(c.volume)
            if i > 0:
                self.log_ret[i] = math.log(c.close / candles[i - 1].close)
                self.change[i] = c.close - candles[i - 1].close

    def ret(self, k: int, i: int) -> float:
        """log(close[i] / close[i-k]). Requires i >= k."""
        return math.log(self.closes[i] / self.closes[i - k])

    def vol(self, n: int, i: int) -> float:
        """Sample stdev of log_ret[i-n+1..i]. Requires i >= n."""
        return _stdev(self.log_ret, i - n + 1, i)

    def range1(self, i: int) -> float:
        """(high - low) / close."""
        return _div(self.highs[i] - self.lows[i], self.closes[i])

    def body1(self, i: int) -> float:
        """(close - open) / open, signed."""
        return _div(self.closes[i] - self.opens[i], self.opens[i])

    def upper_wick1(self, i: int) -> float:
        """(high - max(open, close)) / (high - low).

        Measured against the range, not the close, because range_1 already
        carries the range in price-relative form: the three geometry features
        then sum to 1 for any well-formed bar.
        """
        body = max(self.opens[i], self.closes[i])
        return _div(self.highs[i] - body, self.highs[i] - self.lows[i])

    def lower_wick1(self, i: int) -> float:
        """(min(open, close) - low) / (high - low)."""
        body = min(self.opens[i], self.closes[i])
        return _div(body - self.lows[i], self.highs[i] - self.lows[i])

    def volume_z20(self, i: int) -> float:
        """Volume against its own trailing 20-bar window, current bar included."""
        return _zscore(self.volumes[i], self.volumes, i - WINDOW_VOL_Z + 1, i)

    def volume_ratio(self, i: int) -> float:
        """volume[i] / mean(volume[i-19..i]). 1.0 is a typical bar."""
        return _div(self.volumes[i], _mean(self.volumes, i - WINDOW_VOL_Z + 1, i))

    def rsi14(self, i: int) -> float:
        """RSI(14) over the last 14 simple changes, mapped onto [-1, 1].

        Simple means, not Wilder's smoothing: Wilder's seeds a running average
        at bar 14 and carries it, so its value at bar i would depend on where
        the caller's query started. A 100-candle pull and a 100000-bar backfill
        would then disagree about the same market state, which is exactly the
        drift the pull architecture cannot tolerate.
        """
        gains = 0.0
        losses = 0.0
        for j in range(i - WINDOW_RSI + 1, i + 1):
            delta = self.change[j]
            if delta > 0.0:
                gains += delta
            elif delta < 0.0:
                losses -= delta

        n = float(WINDOW_RSI)
        avg_gain, avg_loss = gains / n, losses / n

        if avg_gain == 0.0 and avg_loss == 0.0:
            rsi = 50.0  # no movement at all: neutral, not undefined
        elif avg_loss == 0.0:
            rsi = 100.0  # every change was up
        else:
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        return 2.0 * rsi / 100.0 - 1.0

    def sma_gap10(self, i: int) -> float:
        """(close[i] - mean(close[i-9..i])) / mean(close[i-9..i])."""
        sma = _mean(self.closes, i - WINDOW_SMA + 1, i)
        return _div(self.closes[i] - sma, sma)

    def spread_proxy(self, i: int) -> float:
        """abs(close - open) / volume: rupees of body per share traded."""
        return _div(abs(self.closes[i] - self.opens[i]), self.volumes[i])


def _series(candles: Sequence[Candle]) -> _Series:
    return _Series(candles)


def _sanitize(value: float) -> float:
    """The numeric policy: a non-finite feature is 0, never NaN.

    A zero denominator is the normal case in this domain, not an exceptional
    one — a flat window has zero stdev, a bar that never traded has zero
    volume, a doji has zero range. "No information" is precisely what a zero
    column says to the model, and a NaN would poison an entire context table
    rather than merely degrade one row.
    """
    return value if math.isfinite(value) else 0.0


def _div(num: float, den: float) -> float:
    """num/den, or 0 when den is 0 or the quotient is not finite."""
    if den == 0.0:
        return 0.0
    return _sanitize(num / den)


def _mean(values: Sequence[float], lo: int, hi: int) -> float:
    """Arithmetic mean of values[lo:hi+1] inclusive, or 0 for an empty range.

    Windows are summed per row rather than prefix-summed. A 60-bar mean is 60
    additions; a prefix sum would be O(1) but subtracts two large nearly equal
    totals, and the rounding error of that subtraction grows with the length of
    the history. A feature whose value drifts as the history grows is a bug
    that looks like noise, so the cost is paid and the precision kept.
    """
    if hi < lo:
        return 0.0
    total = 0.0
    for i in range(lo, hi + 1):
        total += values[i]
    return total / float(hi - lo + 1)


def _stdev(values: Sequence[float], lo: int, hi: int) -> float:
    """Sample stdev of values[lo:hi+1], with the n-1 denominator."""
    n = hi - lo + 1
    if n < 2:
        return 0.0
    mean = _mean(values, lo, hi)
    total = 0.0
    for i in range(lo, hi + 1):
        delta = values[i] - mean
        total += delta * delta
    return math.sqrt(total / float(n - 1))


def _zscore(x: float, values: Sequence[float], lo: int, hi: int) -> float:
    """(x - mean) / stdev over the window, or 0 when it has no dispersion."""
    return _div(x - _mean(values, lo, hi), _stdev(values, lo, hi))


def _validate(candles: Sequence[Candle]) -> None:
    """Reject input that would make the table confidently wrong.

    Every check here fails invisibly downstream: a window mixing two symbols
    averages two unrelated price series into a plausible return, a window that
    rewinds double-counts a bar, an inconsistent bar yields a wick measured
    from a high that never happened. All of them produce finite numbers.

    Gaps are not a failure. A missing minute is a fact about the market, and it
    leaves the trailing windows shorter than the clock would suggest — which is
    what a fixed-window feature should do.
    """
    symbol = candles[0].symbol
    previous: datetime | None = None
    for i, c in enumerate(candles):
        if c.symbol != symbol:
            raise FeatureError(
                f"candle {i} is {c.symbol!r} but the window opened with {symbol!r}: "
                "a feature window must hold one symbol"
            )
        if c.timestamp.tzinfo is None:
            raise FeatureError(
                f"candle {i} at {c.timestamp} has no timezone: a naive timestamp "
                "would make minute_of_session depend on the host's local zone"
            )
        if previous is not None and c.timestamp <= previous:
            raise FeatureError(
                f"candle {i} at {_rfc3339(c.timestamp)} does not follow "
                f"{_rfc3339(previous)}: candles must ascend strictly in time"
            )
        if not all(
            math.isfinite(v) for v in (c.open, c.high, c.low, c.close)
        ):
            raise FeatureError(f"candle {i} at {_rfc3339(c.timestamp)} has a non-finite price")
        if min(c.open, c.high, c.low, c.close) <= 0.0:
            raise FeatureError(
                f"candle {i} at {_rfc3339(c.timestamp)} has a non-positive price: "
                f"open={c.open!r} high={c.high!r} low={c.low!r} close={c.close!r}"
            )
        if c.high < c.low or c.high < c.open or c.high < c.close or (
            c.low > c.open or c.low > c.close
        ):
            raise FeatureError(
                f"candle {i} at {_rfc3339(c.timestamp)} is not a well-formed OHLC bar: "
                f"open={c.open!r} high={c.high!r} low={c.low!r} close={c.close!r}"
            )
        if c.volume < 0:
            raise FeatureError(
                f"candle {i} at {_rfc3339(c.timestamp)} has negative volume {c.volume}"
            )
        previous = c.timestamp


def _rfc3339(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "FeatureBuilder",
    "FeatureError",
    "FeatureTable",
    "IST_FIXED_OFFSET",
    "SESSION_OPEN_MINUTE",
    "WARMUP_ROWS",
    "WINDOW_RET1",
    "WINDOW_RET15",
    "WINDOW_RET5",
    "WINDOW_RET60",
    "WINDOW_RSI",
    "WINDOW_SMA",
    "WINDOW_VOL5",
    "WINDOW_VOL20",
    "WINDOW_VOL_Z",
    "trading_zone",
]
