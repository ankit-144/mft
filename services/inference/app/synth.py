"""A deterministic synthetic candle series, so the backtest can run today.

`data/candles/` is empty, there are no broker credentials, and `make backtest`
has nothing to read. A harness that only works once somebody has run the
ingestion service for a month is not a harness, it is a rumour. So this module
generates the input.

# What this is for, and what it is not

It proves the **plumbing**: that the feature path builds a table, that the model
scores it, that the threshold and the risk gate produce decisions, that the
accounting adds up. It proves **nothing at all** about whether the strategy has
an edge. A geometric-Brownian-motion series has no momentum, no mean reversion,
no volatility clustering and no news, so a model that appears to forecast it is
forecasting a random number generator with parameters we chose. Every artifact
the backtest writes says so, in the summary and in the JSON.

# The shape is realistic where it is cheap to be

A run of 2000 bars has to look enough like a NSE session that the two calendar
features are not nonsense and the volume features are not degenerate:

* 09:15–15:30 IST, one bar per minute, 375 bars a session, weekdays only.
  `minute_of_session` (feature 16) is measured from `SESSION_OPEN_MINUTE`, so
  the series has to live inside that window to be worth anything.
* Timestamps are UTC on the wire, as `docs/contracts.md` §1 requires; the
  session grid is computed in IST and converted once.
* Prices are rounded to paise and volume is a whole number of shares, because
  real bars are and the feature table divides by them.
* Volatility and volume are U-shaped across the session: heavy at the open,
  quiet at lunch, heavy again into the close. A flat profile would make
  `volume_ratio` and `vol_ratio` behave nothing like a live session.

# The shape is not realistic, and here is the list

This is the part that matters, because each item below is a way real data
punishes a backtest that synthetic data rewards:

* **No overnight gaps.** Prices are continuous, so `ret_1` never gaps. Real
  series open 1–5% away from the previous close and a 1-minute strategy lives
  or dies on those bars.
* **Symbols are independent.** No cross-sectional correlation, so a portfolio
  of three looks like three independent bets and the diversification in the
  equity curve is an artefact.
* **No limit-up, limit-down, halts or circuit filters.** Real bars get skipped
  or frozen; here the tape is always tradable.
* **Constant spread proxy.** `spread_proxy` is `|close-open|/volume`, which on
  a continuous GBM is uncorrelated with the real bid-ask, so nothing penalises
  trading into a wide market.
* **No intraday drift seasonality.** Real indices drift with the hour. A drift
  of exactly zero is the honest null, and it is also the easiest case for a
  mean-reverting model to look good on.
* **Volume is independent of the price move.** Real volume confirms moves;
  here it does not, so `volume_z_20` carries no information and a model that
  uses it is being tested on noise.

# Determinism

Seeded from `--seed` and the symbol, through `random.Random(str)`, whose
seeding is a SHA-512 of the string and does not depend on `PYTHONHASHSEED`.
The same `--seed` gives byte-identical candles on any machine, which is what
makes a failing assertion reproducible without recording a fixture.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo

from .candles import Candle
from .features import SESSION_OPEN_MINUTE, trading_zone

logger = logging.getLogger("mft.inference.synth")

#: One-minute bars in a 09:15–15:30 IST session. The last bar is labelled
#: 15:29 and closes at 15:30, so the session is 375 bars and not 376.
SESSION_BARS_PER_DAY = 375

#: NSE trading days a year. 252, the long-run mean, and an assumption like
#: every other number in this module.
TRADING_DAYS_PER_YEAR = 252

#: Bars a year, used to turn an annualised volatility into a per-bar one and
#: to annualise a Sharpe ratio. Stated rather than buried, because an
#: annualisation constant is a choice and a reader cannot check it otherwise.
BARS_PER_YEAR = SESSION_BARS_PER_DAY * TRADING_DAYS_PER_YEAR

#: First calendar day the generator will place bars on. A Monday, so no run
#: silently starts mid-week and a short `--bars` still looks like a session.
DEFAULT_START_DATE = date(2026, 1, 5)

#: Starting price. A liquid large-cap level, so the notional of a default
#: `inference.order_quantity: 10` is in the range the brokerage floor bites on.
DEFAULT_START_PRICE = 1400.0

#: Annualised volatility of the walk. Roughly a quiet large-cap year; it is an
#: assumption, and the Sharpe it produces is a property of this number.
DEFAULT_ANNUAL_VOLATILITY = 0.30

#: Annualised drift. Zero by default and on purpose: a series with a positive
#: drift would flatter any long-only strategy for reasons that have nothing to
#: do with the model.
DEFAULT_ANNUAL_DRIFT = 0.0

#: Shares per minute before the session shape is applied. Sized so the median
#: minute is worth a few lakh of rupees, which is what a mega-cap's tape looks
#: like, and so `order_quantity: 10` is a small fraction of a minute's volume
#: rather than a large one.
DEFAULT_BASE_VOLUME = 12000.0

#: Log-normal dispersion of volume around its session-shaped mean. Real minute
#: volume is heavily right-skewed; 0.45 is a rough fit.
VOLUME_LOG_DISPERSION = 0.45

#: Intraday volatility multipliers at the open/close edges and at midday. A
#: U, not a V: the open auction and the closing print are both heavy, and the
#: lunch hour is not.
VOL_SHAPE_EDGE = 1.55
VOL_SHAPE_MIDDAY = 0.62

#: The matching volume multipliers. Inverted relative to volatility, because a
#: quiet midday bar is a thin one.
VOLUME_SHAPE_EDGE = 1.60
VOLUME_SHAPE_MIDDAY = 0.70

#: Half the high-low range as a fraction of the body, per unit of that bar's
#: sigma. 0.5 puts the wick at roughly one sigma on each side, which is what
#: `range_1` and the two wick features expect to see.
RANGE_SIGMA_FACTOR = 0.5

#: Prices are rounded to paise. Indian equities quote to 0.05 but a generated
#: series does not need that much resolution, and 2dp keeps the stored candles
#: the same shape as the Parquet ones.
PRICE_DECIMALS = 2


@dataclass(frozen=True, slots=True)
class SynthSpec:
    """Everything about a synthetic run except the seed, which is separate.

    Args:
        bars: How many 1-minute bars to generate, across as many sessions as
            that takes.
        seed: The generator seed. Same seed, same series, on any machine.
        start_price: Price of the first bar's open.
        annual_volatility: Annualised per-bar volatility before the intraday
            shape is applied.
        annual_drift: Annualised drift. Zero by default and deliberately so.
        base_volume: Shares per minute at the midday quiet point.
        start: First calendar day, IST. Weekends are skipped forward.
        zone: Trading location. Only affects which days are "weekday" and the
            session grid, which is always the IST one.
    """

    bars: int = 2000
    seed: int = 20260929
    start_price: float = DEFAULT_START_PRICE
    annual_volatility: float = DEFAULT_ANNUAL_VOLATILITY
    annual_drift: float = DEFAULT_ANNUAL_DRIFT
    base_volume: float = DEFAULT_BASE_VOLUME
    start: date = DEFAULT_START_DATE
    zone: tzinfo | None = None

    def __post_init__(self) -> None:
        if self.bars < 0:
            raise ValueError(f"bars must not be negative, got {self.bars}")
        if self.start_price <= 0.0:
            raise ValueError(f"start_price must be positive, got {self.start_price}")
        if self.annual_volatility <= 0.0:
            raise ValueError(
                f"annual_volatility must be positive, got {self.annual_volatility}"
            )
        if self.base_volume <= 0.0:
            raise ValueError(f"base_volume must be positive, got {self.base_volume}")


def session_grid(count: int, *, start: date, zone: tzinfo) -> list[datetime]:
    """`count` 1-minute bar timestamps on the NSE session grid, in UTC.

    Starts at the open of `start` and rolls to the next weekday when a session
    ends. Weekends are skipped rather than filled, so the series has the same
    `minute_of_session` value at the same point in every session — which is
    what makes feature 16 a calendar feature and not a row counter.
    """
    open_local = time(SESSION_OPEN_MINUTE // 60, SESSION_OPEN_MINUTE % 60)
    stamps: list[datetime] = []
    day = start
    while len(stamps) < count:
        if day.weekday() < 5:
            session_open = datetime.combine(day, open_local, tzinfo=zone)
            for minute in range(SESSION_BARS_PER_DAY):
                if len(stamps) == count:
                    break
                stamps.append((session_open + timedelta(minutes=minute)).astimezone(timezone.utc))
        day += timedelta(days=1)
    return stamps


def generate(symbol: str, spec: SynthSpec, *, zone_name: str = "Asia/Kolkata") -> list[Candle]:
    """Generate one symbol's synthetic candle series.

    Args:
        symbol: The instrument, used for the per-symbol seed stream and set on
            every returned candle. `FeatureBuilder` rejects a window that mixes
            symbols, so this is load-bearing.
        spec: The run's parameters.
        zone_name: Ignored when `spec.zone` is set; otherwise resolved through
            `features.trading_zone`, so the fallback is the same fixed +05:30
            the feature builder would use.

    Returns:
        `spec.bars` well-formed candles, oldest first, timestamps in UTC and
        strictly ascending.
    """
    if spec.bars == 0:
        return []

    zone = spec.zone if spec.zone is not None else trading_zone(zone_name)
    # A string seed is hashed with SHA-512 inside `random.Random`, so this does
    # not depend on PYTHONHASHSEED and the series is reproducible across runs
    # and machines. Mixing the symbol in keeps three instruments from being
    # three copies of one series.
    rng = random.Random(f"{spec.seed}:{symbol.strip().upper()}")

    stamps = session_grid(spec.bars, start=spec.start, zone=zone)
    sigma = spec.annual_volatility / math.sqrt(BARS_PER_YEAR)
    drift = spec.annual_drift / BARS_PER_YEAR

    candles: list[Candle] = []
    price = spec.start_price
    for index, stamp in enumerate(stamps):
        shape = _volatility_shape(index % SESSION_BARS_PER_DAY)
        shock = rng.gauss(0.0, 1.0)
        # Itô-corrected log drift. With drift 0 the walk has no tendency at all,
        # which is the honest null for a backtest.
        log_move = drift - 0.5 * sigma * sigma + sigma * shape * shock
        open_ = price
        close = _round(open_ * math.exp(log_move))
        if close <= 0.0:  # pragma: no cover - a sub-paise price, not reachable
            close = _round(open_)

        half_range = RANGE_SIGMA_FACTOR * sigma * shape * abs(rng.gauss(0.0, 1.0))
        high = _round(max(open_, close) * (1.0 + half_range))
        low = _round(min(open_, close) * (1.0 - half_range))
        # Rounding to 2dp can never move a value below an already-rounded
        # neighbour, but a generated bar that fails `FeatureBuilder._validate`
        # would abort the whole run, so the invariant is enforced here rather
        # than assumed.
        high = max(high, open_, close)
        low = min(low, open_, close)

        volume = max(1, int(spec.base_volume * _volume_shape(index % SESSION_BARS_PER_DAY) *
                            math.exp(rng.gauss(0.0, VOLUME_LOG_DISPERSION))))
        candles.append(
            Candle(
                symbol=symbol.strip().upper(),
                timestamp=stamp,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close

    logger.info(
        "synthetic %s: %d bars, seed=%d, %s..%s, price %.2f..%.2f, "
        "vol(annual)=%.2f drift(annual)=%.3f",
        symbol,
        len(candles),
        spec.seed,
        _iso(candles[0].timestamp),
        _iso(candles[-1].timestamp),
        candles[0].open,
        candles[-1].close,
        spec.annual_volatility,
        spec.annual_drift,
    )
    return candles


def generate_many(
    symbols: Sequence[str], spec: SynthSpec, *, zone_name: str = "Asia/Kolkata"
) -> dict[str, list[Candle]]:
    """One independent series per symbol, keyed by upper-case symbol."""
    return {
        symbol.strip().upper(): generate(symbol, spec, zone_name=zone_name)
        for symbol in symbols
    }


def _volatility_shape(minute_of_session: int) -> float:
    """Volatility multiplier for a bar, U-shaped across the session."""
    half = (SESSION_BARS_PER_DAY - 1) / 2.0
    distance = abs(minute_of_session - half) / half  # 0 at midday, 1 at the edges
    return VOL_SHAPE_MIDDAY + (VOL_SHAPE_EDGE - VOL_SHAPE_MIDDAY) * distance**2


def _volume_shape(minute_of_session: int) -> float:
    """Volume multiplier for a bar, high at the edges and low at midday."""
    half = (SESSION_BARS_PER_DAY - 1) / 2.0
    distance = abs(minute_of_session - half) / half
    return VOLUME_SHAPE_MIDDAY + (VOLUME_SHAPE_EDGE - VOLUME_SHAPE_MIDDAY) * distance**2


def _round(value: float) -> float:
    """Round to paise, and never to zero or below: a zero price is rejected."""
    return max(round(value, PRICE_DECIMALS), 0.01)


def _iso(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "BARS_PER_YEAR",
    "DEFAULT_ANNUAL_DRIFT",
    "DEFAULT_ANNUAL_VOLATILITY",
    "DEFAULT_BASE_VOLUME",
    "DEFAULT_START_DATE",
    "DEFAULT_START_PRICE",
    "SESSION_BARS_PER_DAY",
    "TRADING_DAYS_PER_YEAR",
    "SynthSpec",
    "generate",
    "generate_many",
    "session_grid",
]
