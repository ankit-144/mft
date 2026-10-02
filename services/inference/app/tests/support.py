"""Reusable builders and fakes for the inference service tests.

Separate from `conftest.py` so that test modules can import them by a name no
other conftest in this service owns. C5's `model/tests/conftest.py` is also
called `conftest`, and a run covering both trees would resolve
`from conftest import ...` to whichever pytest imported first.

Nothing here may load the TabFM checkpoint, open a socket, or write outside a
`tmp_path`. A test that can do any of those is a bug in the test, not a
demonstration.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.candles import Candle
from app.config import Config, InferenceConfig
from app.execution import SignalAck
from app.signals import Signal

#: The minute every synthetic candle window is anchored to: 2026-08-03 07:50
#: UTC, which is 13:20 IST. The same anchor the Go golden fixture uses, so a
#: golden test and a loop test agree about what minute they are looking at.
ANCHOR = datetime(2026, 8, 3, 7, 50, tzinfo=timezone.utc)

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "feature_golden.json"


def load_golden() -> dict[str, Any]:
    """The shared candles-to-features fixture, transcribed from C3's Go test."""
    return json.loads(GOLDEN_PATH.read_text())


def golden_candles(symbol: str = "RELIANCE") -> list[Candle]:
    """The golden fixture as a candle window, anchored at its own start time."""
    document = load_golden()
    start = datetime.fromisoformat(document["start_utc"].replace("Z", "+00:00"))
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(minutes=bar["minute"]),
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
        )
        for bar in document["bars"]
    ]


def make_candles(
    rows: int = 160,
    *,
    symbol: str = "RELIANCE",
    start: datetime | None = None,
    drift: float = 0.0,
    volatility: float = 0.0012,
    seed: int = 7,
) -> list[Candle]:
    """A synthetic, well-formed 1-minute window.

    Deterministic: the walk comes from a seeded generator, so a failing
    assertion is reproducible without recording a fixture. OHLC is built
    around the close so every bar passes the builder's well-formedness check —
    a real window that the feature builder rejects is a different test.
    """
    rng = random.Random(seed)
    anchor = start if start is not None else ANCHOR
    out: list[Candle] = []
    price = 2900.0
    for index in range(rows):
        step = drift + rng.gauss(0.0, volatility)
        open_ = price
        close = price * (1.0 + step)
        high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, 0.0004)))
        low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, 0.0004)))
        volume = int(rng.lognormvariate(9.0, 0.4))
        out.append(
            Candle(
                symbol=symbol,
                timestamp=anchor + timedelta(minutes=index),
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=volume,
            )
        )
        price = close
    return out


class FakeStore:
    """An in-process `CandleStore`.

    Records every read so a test can assert how much history the loop pulled,
    which is a real question: pulling the whole store every minute would work
    and be absurd.
    """

    def __init__(self, candles: dict[str, list[Candle]] | None = None) -> None:
        self._candles: dict[str, list[Candle]] = dict(candles or {})
        self.reads: list[tuple[str, int]] = []
        self.fail_with: Exception | None = None

    def add(self, symbol: str, candles: list[Candle]) -> None:
        """Register a window for a symbol."""
        self._candles[symbol] = list(candles)

    async def tail(self, symbol: str, limit: int) -> list[Candle]:
        """Return the newest `limit` candles for `symbol`, oldest first."""
        self.reads.append((symbol, limit))
        if self.fail_with is not None:
            raise self.fail_with
        rows = self._candles.get(symbol, [])
        return rows[-limit:] if limit > 0 else list(rows)

    async def aclose(self) -> None:
        return None


class FakeSender:
    """An execution service that records signals instead of trading them.

    `raise_with` makes every submit fail, which is how the "a throwing POST
    must skip rather than crash" case is exercised. Every other field is what a
    caller would want to assert on afterwards.
    """

    def __init__(self, *, raise_with: Exception | None = None, order_id: str = "4412") -> None:
        self.sent: list[Signal] = []
        self.raise_with = raise_with
        self.order_id = order_id
        self.closed = False

    async def submit(self, signal: Signal) -> SignalAck:
        """Record the signal, or raise what the test asked for."""
        self.sent.append(signal)
        if self.raise_with is not None:
            raise self.raise_with
        return SignalAck(
            order_id=self.order_id, status="FILLED", score=signal.score, http_status=202
        )

    async def aclose(self) -> None:
        self.closed = True

    @property
    def count(self) -> int:
        return len(self.sent)


class ConstantModel:
    """An `InferenceModel` whose score is a constant, for threshold tests.

    A fixed score is what makes a threshold test a threshold test: a real
    model produces a different number on every window, so "below threshold
    sends nothing" would pass or fail by luck.
    """

    def __init__(self, score: float = 0.9, *, loaded: bool = True) -> None:
        self.score = score
        self._loaded = loaded
        self.calls = 0

    @property
    def name(self) -> str:
        return "constant-v1"

    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        self._loaded = True

    def predict(self, context: Any, horizon: int) -> float:
        self.calls += 1
        return self.score


def app_config(inference: InferenceConfig) -> Config:
    """A whole config wrapping one `inference` section."""
    return Config(inference=inference)
