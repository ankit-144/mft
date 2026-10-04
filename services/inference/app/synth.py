"""A deterministic synthetic candle series, so the backtest can run today."""

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


SESSION_BARS_PER_DAY = 375


TRADING_DAYS_PER_YEAR = 252


BARS_PER_YEAR = SESSION_BARS_PER_DAY * TRADING_DAYS_PER_YEAR


DEFAULT_START_DATE = date(2026, 1, 5)


DEFAULT_START_PRICE = 1400.0


DEFAULT_ANNUAL_VOLATILITY = 0.30


DEFAULT_ANNUAL_DRIFT = 0.0


DEFAULT_BASE_VOLUME = 12000.0


VOLUME_LOG_DISPERSION = 0.45


VOL_SHAPE_EDGE = 1.55
VOL_SHAPE_MIDDAY = 0.62


VOLUME_SHAPE_EDGE = 1.60
VOLUME_SHAPE_MIDDAY = 0.70


RANGE_SIGMA_FACTOR = 0.5


PRICE_DECIMALS = 2


@dataclass(frozen=True, slots=True)
class SynthSpec:
    """Everything about a synthetic run except the seed, which is separate."""

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
    """`count` 1-minute bar timestamps on the NSE session grid, in UTC."""
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
    """Generate one symbol's synthetic candle series."""
    if spec.bars == 0:
        return []

    zone = spec.zone if spec.zone is not None else trading_zone(zone_name)


    rng = random.Random(f"{spec.seed}:{symbol.strip().upper()}")

    stamps = session_grid(spec.bars, start=spec.start, zone=zone)
    sigma = spec.annual_volatility / math.sqrt(BARS_PER_YEAR)
    drift = spec.annual_drift / BARS_PER_YEAR

    candles: list[Candle] = []
    price = spec.start_price
    for index, stamp in enumerate(stamps):
        shape = _volatility_shape(index % SESSION_BARS_PER_DAY)
        shock = rng.gauss(0.0, 1.0)


        log_move = drift - 0.5 * sigma * sigma + sigma * shape * shock
        open_ = price
        close = _round(open_ * math.exp(log_move))
        if close <= 0.0:  # pragma: no cover
            close = _round(open_)

        half_range = RANGE_SIGMA_FACTOR * sigma * shape * abs(rng.gauss(0.0, 1.0))
        high = _round(max(open_, close) * (1.0 + half_range))
        low = _round(min(open_, close) * (1.0 - half_range))


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
    distance = abs(minute_of_session - half) / half
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
