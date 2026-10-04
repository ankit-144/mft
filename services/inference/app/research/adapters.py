"""Typed boundaries between research algorithms, predictors, and strategies."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Protocol

import pandas as pd


class Predictor(Protocol):
    """Minimal synchronous prediction contract used by a walk-forward run."""

    @property
    def name(self) -> str:
        """Stable model identifier."""
        ...

    async def load(self) -> None:
        """Prepare the model before a run."""
        ...

    def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Score the latest context row."""
        ...


class Strategy(Protocol):
    """Map a bounded predictor score onto an allowed position side."""

    def side(self, score: float) -> int:
        """Return 1 for long, -1 for short, or 0 for no position."""
        ...


@dataclass(frozen=True, slots=True)
class ModelPredictorAdapter:
    """Adapt an InferenceModel implementation to the research Predictor API."""

    model: Predictor

    @property
    def name(self) -> str:
        """Return the wrapped model's identifier."""
        return self.model.name

    async def load(self) -> None:
        """Load the wrapped model once through its standard protocol."""
        await self.model.load()

    def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Delegate a prediction without changing the supplied context."""
        return self.model.predict(context, horizon)


@dataclass(frozen=True, slots=True)
class ThresholdStrategy:
    """Score-threshold position adapter with a long-only safe default."""

    threshold: float = 0.25
    long_only: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    def side(self, score: float) -> int:
        """Apply threshold and optional short-side support."""
        if score >= self.threshold and score > 0.0:
            return 1
        if not self.long_only and score <= -self.threshold and score < 0.0:
            return -1
        return 0


_STRATEGIES: dict[str, Callable[..., Strategy]] = {"threshold": ThresholdStrategy}


def register_strategy(name: str, factory: Callable[..., Strategy]) -> None:
    """Register an extensible decision strategy factory by stable name."""
    if not name.strip() or name in _STRATEGIES:
        raise ValueError(f"strategy {name!r} is empty or already registered")
    _STRATEGIES[name] = factory


def create_strategy(name: str, options: Mapping[str, object] | None = None) -> Strategy:
    """Construct a strategy from its registered factory and JSON options."""
    try:
        factory = _STRATEGIES[name]
    except KeyError as err:
        raise ValueError(f"unknown strategy {name!r}; available: {', '.join(available_strategies())}") from err
    try:
        return factory(**dict(options or {}))
    except TypeError as err:
        raise ValueError(f"invalid options for strategy {name!r}: {err}") from err


def available_strategies() -> tuple[str, ...]:
    """Return stable names of registered strategy factories."""
    return tuple(sorted(_STRATEGIES))


__all__ = [
    "ModelPredictorAdapter", "Predictor", "Strategy", "ThresholdStrategy",
    "available_strategies", "create_strategy", "register_strategy",
]
