"""The `InferenceModel` protocol and the registry that backs model selection."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger("mft.inference.model")


FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "vol_5",
    "vol_20",
    "vol_ratio",
    "range_1",
    "body_1",
    "upper_wick_1",
    "lower_wick_1",
    "volume_z_20",
    "volume_ratio",
    "momentum_rsi_14",
    "sma_gap_10",
    "minute_of_session",
    "hour_of_day",
    "spread_proxy",
)


TARGET_COLUMN = "ret_1"


MIN_CONTEXT_ROWS = 60


MAX_CONTEXT_ROWS = 100


MAX_NUM_FEATURES = 500


MIN_RETURN_VOLATILITY = 1e-6


class InferenceModelError(RuntimeError):
    """Base class for every error raised by a model implementation."""


class ModelNotLoadedError(InferenceModelError):
    """Raised when `predict` is called before `load` has completed."""


class ModelUnavailableError(InferenceModelError):
    """Raised when a model cannot be loaded, e.g."""


class InvalidContextError(InferenceModelError):
    """Raised when the supplied context cannot be used as a prompt table."""


@runtime_checkable
class InferenceModel(Protocol):
    """What the inference service needs from any model implementation."""

    @property
    def name(self) -> str:
        """Identifier of the loaded weights, e.g."""
        ...

    async def load(self) -> None:
        """Acquire weights and make the model ready for `predict`."""
        ...

    def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Return the expected return for the bar `horizon` bars ahead."""
        ...

    def is_loaded(self) -> bool:
        """Whether `predict` can be called."""
        ...


class BaseInferenceModel(ABC):
    """Abstract base sharing the contract every implementation must honour."""

    @property
    def name(self) -> str:
        """Identifier of the loaded weights."""
        return self._name

    def is_loaded(self) -> bool:
        """Whether `predict` can be called."""
        return self._loaded

    async def load(self) -> None:
        """Load the model."""
        if self._loaded:
            logger.debug("%s already loaded, skipping", self._name)
            return
        await self._load()
        self._loaded = True
        logger.info("model %s loaded", self._name)

    def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Validate, predict, and clamp the result into [-1, 1]."""
        if not self._loaded:
            raise ModelNotLoadedError(f"{self._name} is not loaded; call load() first")
        table = validate_context(context, horizon=horizon)
        raw = self._predict_raw(table, horizon)
        return scale_to_unit(raw)

    @property
    @abstractmethod
    def _name(self) -> str:
        """Identifier of the weights this instance serves."""

    @abstractmethod
    async def _load(self) -> None:
        """Acquire weights."""

    @abstractmethod
    def _predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        """Return the expected return, pre-scaling, in natural return units."""


def validate_context(context: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Check a context table and return it in newest-last form."""
    if horizon < 1:
        raise InvalidContextError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(context, pd.DataFrame):
        raise InvalidContextError(f"context must be a DataFrame, got {type(context).__name__}")

    missing = [c for c in FEATURE_COLUMNS if c not in context.columns]
    if missing:
        raise InvalidContextError(
            f"context is missing {len(missing)} frozen feature column(s): {missing}"
        )

    if tuple(context.columns) != FEATURE_COLUMNS:
        raise InvalidContextError("context must contain exactly the ordered 18 feature columns")


    usable = len(context) - horizon
    if usable < MIN_CONTEXT_ROWS:
        raise InvalidContextError(
            f"need at least {MIN_CONTEXT_ROWS + horizon} rows to predict {horizon} bar(s) "
            f"ahead, got {len(context)}"
        )

    try:
        values = context.to_numpy(dtype=float)
    except (ValueError, TypeError) as err:
        raise InvalidContextError("context features must be numeric") from err
    if not np.isfinite(values).all():
        raise InvalidContextError("context contains NaN or infinity in the feature columns")

    return context.reset_index(drop=True)


def take_recent(context: pd.DataFrame, limit: int = MAX_CONTEXT_ROWS) -> pd.DataFrame:
    """Truncate a context table to its `limit` most recent rows."""
    if limit <= 0:
        raise InvalidContextError(f"limit must be > 0, got {limit}")
    if len(context) <= limit:
        return context
    return context.iloc[-limit:].reset_index(drop=True)


def returns_volatility(targets: np.ndarray) -> float:
    """Sample stdev of the target column, floored away from zero."""
    vol = float(np.std(targets)) if targets.size > 1 else 0.0
    return max(vol, MIN_RETURN_VOLATILITY)


def scale_to_unit(value: float, softness: float = 1.0) -> float:
    """Map a volatility-normalised return onto [-1, 1]."""
    if not np.isfinite(value):
        logger.warning("non-finite model output %r, returning 0.0", value)
        return 0.0
    if not np.isfinite(softness) or softness <= 0.0:
        raise InvalidContextError(f"softness must be finite and > 0, got {softness}")
    scaled = float(np.tanh(value / softness))
    return max(-1.0, min(1.0, scaled))


ModelFactory = Callable[[], InferenceModel]

_REGISTRY: dict[str, ModelFactory] = {}


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory under `name` for `config.InferenceConfig.Model`."""
    if not name:
        raise ValueError("model name must be non-empty")

    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = factory
        logger.debug("registered inference model %r -> %r", name, factory)
        return factory

    return decorator


def unregister(name: str) -> None:
    """Remove a registration."""
    _REGISTRY.pop(name, None)


def create_model(name: str) -> InferenceModel:
    """Instantiate the model registered under `name`."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(available_models()) or "<none>"
        raise ValueError(
            f"unknown inference model {name!r}; registered models: {known}"
        ) from None
    model = factory()
    logger.debug("created inference model %r (%s)", name, model.name)
    return model


def available_models() -> tuple[str, ...]:
    """Registered model names, sorted."""
    return tuple(sorted(_REGISTRY))


def _register_builtin_models() -> None:
    """Register the two implementations shipped with the service."""
    from .heuristic_model import HeuristicModel

    _REGISTRY.setdefault("heuristic", HeuristicModel)

    try:
        from .tabfm_model import TabFMModel

        _REGISTRY.setdefault("tabfm", TabFMModel)
    except ImportError:  # pragma: no cover
        logger.warning("tabfm backend not importable; 'tabfm' is not selectable")


_register_builtin_models()
