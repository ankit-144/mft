"""`contracts.Signal` and the idempotency key that makes a retry safe.

`docs/contracts.md` §1 defines the Go struct; this is the same wire shape in
Python. The field names, the JSON keys and the timestamp format are the
contract — C7 decodes this body with `encoding/json` into `contracts.Signal`
and rejects a missing or empty `idempotency_key` with a 400 before the risk
gate ever runs.

# The idempotency key

**Format: `<SYMBOL>:<SIDE>:<YYYYMMDDTHHMM>`**, all three fields in uppercase
and the minute rendered from the *UTC* `as_of`, zero-padded, with no
separators inside the timestamp. It is the example key in `docs/contracts.md`
§7 — `RELIANCE:BUY:20260929T1031` — so the format is the documented one rather
than a private invention.

It is derived from exactly the three fields the contract names: symbol, side,
and the minute close being predicted from. That is a deliberate, narrow choice:

* **Same inputs, same key.** Inference retries. A timeout, a 502, or a
  process restart mid-minute must produce the *same* key, or the retry is a
  second order and the position doubles. Nothing time-varying, nothing
  random, nothing per-attempt goes into it.
* **Different minute, different key.** Otherwise C7's 409 would swallow every
  signal after the first and the strategy would place exactly one order.
* **The side is in the key.** A model that flips from BUY to SELL at the same
  minute close is a different decision, and a 409 on it would suppress a
  genuine reversal.

The minute is rendered from UTC because that is what `as_of` is on the wire
and because it is the only rendering that cannot be shifted by a local
timezone. Two processes with different `TZ` settings would still agree.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

logger = logging.getLogger("mft.inference.signals")

SIDE_BUY: Final[str] = "BUY"
SIDE_SELL: Final[str] = "SELL"

#: `strftime` pattern for the minute component of an idempotency key.
KEY_MINUTE_FORMAT: Final[str] = "%Y%m%dT%H%M"

Side = Literal["BUY", "SELL"]


def side_for_score(score: float) -> Side:
    """Derive the trade direction from the sign of the conviction score.

    A score of exactly zero has no direction. It is a legitimate model output
    — `scale_to_unit` maps a non-finite prediction to 0.0 rather than to
    maximum conviction — so it is refused here instead of being coerced into a
    buy, where it would trade the neutral answer.

    Raises:
        ValueError: If `score` is zero, non-finite, or outside [-1, 1].
    """
    if not math.isfinite(score):
        raise ValueError(f"score {score!r} is not finite")
    if not -1.0 <= score <= 1.0:
        raise ValueError(f"score {score!r} is outside [-1, 1]")
    if score == 0.0:
        raise ValueError("score 0.0 has no direction; refusing to pick a side")
    return SIDE_BUY if score > 0.0 else SIDE_SELL


def idempotency_key(symbol: str, side: str, as_of: datetime) -> str:
    """Build the key for one (symbol, side, minute close) decision.

    Deterministic in all three arguments and in nothing else. See the module
    docstring for why those three and not others.
    """
    minute = as_of.astimezone(timezone.utc).strftime(KEY_MINUTE_FORMAT)
    return f"{symbol.strip().upper()}:{side.strip().upper()}:{minute}"


def rfc3339(stamp: datetime) -> str:
    """Render a UTC instant the way Go's `time.Time` marshals it."""
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Signal(BaseModel):
    """A model conviction for one symbol at one minute close.

    Mirrors `contracts.Signal` (`docs/contracts.md` §1). Field names are the
    JSON keys, so `model_dump()` is already the request body C7 expects.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: Side
    quantity: int
    price: float
    score: float
    model: str
    as_of: datetime
    idempotency_key: str = ""

    @field_validator("symbol")
    @classmethod
    def _normalise_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("symbol is required")
        return symbol

    @field_validator("side")
    @classmethod
    def _normalise_side(cls, value: str) -> str:
        side = value.strip().upper()
        if side not in (SIDE_BUY, SIDE_SELL):
            raise ValueError(f"side {value!r} must be {SIDE_BUY} or {SIDE_SELL}")
        return side  # type: ignore[return-value]

    @field_validator("price")
    @classmethod
    def _positive_price(cls, value: float) -> float:
        # The risk gate sizes a position as quantity*price, so a zero price is
        # not a small order, it is an unmeasurable one. C7 rejects it with a
        # 400; refusing it here means the loop never builds such a signal.
        if not value > 0.0:
            raise ValueError(f"price {value} must be positive")
        return value

    @field_validator("score")
    @classmethod
    def _unit_score(cls, value: float) -> float:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"score {value} is outside [-1, 1]")
        return value

    @field_validator("quantity")
    @classmethod
    def _positive_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"quantity {value} must be positive")
        return value

    @field_validator("as_of")
    @classmethod
    def _require_zone(cls, value: datetime) -> datetime:
        # A naive timestamp has no single meaning and would be serialised with
        # an invented offset. Every timestamp in the platform is UTC.
        if value.tzinfo is None:
            raise ValueError("as_of must carry a timezone offset")
        return value.astimezone(timezone.utc)

    @field_serializer("as_of")
    def _serialise_as_of(self, value: datetime) -> str:
        return rfc3339(value)

    @classmethod
    def from_score(
        cls,
        *,
        symbol: str,
        score: float,
        price: float,
        quantity: int,
        model: str,
        as_of: datetime,
    ) -> Signal:
        """Derive a complete signal from a model score.

        `price` is the close of the last candle in the context, not the
        predicted value: it is the *reference* price the risk gate sizes
        against and the execution service reports, and a model output in
        [-1, 1] is not a price at all.
        """
        side = side_for_score(score)
        return cls(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            score=score,
            model=model,
            as_of=as_of,
            idempotency_key=idempotency_key(symbol, side, as_of),
        )

    def to_wire(self) -> dict[str, object]:
        """The exact JSON body for `POST /v1/signals`."""
        return self.model_dump(mode="json")


__all__ = [
    "KEY_MINUTE_FORMAT",
    "SIDE_BUY",
    "SIDE_SELL",
    "Side",
    "Signal",
    "idempotency_key",
    "rfc3339",
    "side_for_score",
]
