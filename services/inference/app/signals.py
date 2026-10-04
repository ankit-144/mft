"""`contracts.Signal` and the idempotency key that makes a retry safe."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

logger = logging.getLogger("mft.inference.signals")

SIDE_BUY: Final[str] = "BUY"
SIDE_SELL: Final[str] = "SELL"


KEY_MINUTE_FORMAT: Final[str] = "%Y%m%dT%H%M"

Side = Literal["BUY", "SELL"]


def side_for_score(score: float) -> Side:
    """Derive the trade direction from the sign of the conviction score."""
    if not math.isfinite(score):
        raise ValueError(f"score {score!r} is not finite")
    if not -1.0 <= score <= 1.0:
        raise ValueError(f"score {score!r} is outside [-1, 1]")
    if score == 0.0:
        raise ValueError("score 0.0 has no direction; refusing to pick a side")
    return SIDE_BUY if score > 0.0 else SIDE_SELL


def idempotency_key(symbol: str, side: str, as_of: datetime) -> str:
    """Build the key for one (symbol, side, minute close) decision."""
    minute = as_of.astimezone(timezone.utc).strftime(KEY_MINUTE_FORMAT)
    return f"{symbol.strip().upper()}:{side.strip().upper()}:{minute}"


def rfc3339(stamp: datetime) -> str:
    """Render a UTC instant the way Go's `time.Time` marshals it."""
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Signal(BaseModel):
    """A model conviction for one symbol at one minute close."""

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
        """Derive a complete signal from a model score."""
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
