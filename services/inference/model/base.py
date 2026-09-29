"""The `InferenceModel` protocol and the registry that backs model selection.

The protocol exists so TabFM can be replaced without touching the service.
TabFM's pretrained weights are non-commercial (see `Plan.md` §5), so the
ability to swap the implementation is a hard requirement, not a nicety.

Nothing in this module imports torch or tabfm. That is deliberate: the
heuristic implementation and the test suite must be usable on a machine where
the model backend failed to install.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger("mft.inference.model")

#: The frozen 18-column feature schema, in order. TabFM treats the column set
#: as part of the table contract; changing it invalidates historical context
#: rows. Mirrors `docs/contracts.md` §4. Owned by C3, mirrored here so the
#: model can validate its input.
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

#: The column the models regress on: the realised 1-bar log return.
TARGET_COLUMN = "ret_1"

#: Rows below this are warm-up only and are never valid prediction targets.
MIN_CONTEXT_ROWS = 60

#: TabFM's practical in-context row budget. Attention is roughly linear in
#: table size, so feeding more than this costs CPU time for no extra signal.
MAX_CONTEXT_ROWS = 100

#: TabFM's feature budget. The frozen schema is 18 columns, far below this.
MAX_NUM_FEATURES = 500

#: Smallest volatility we will normalise by, so a dead-flat tape cannot
#: produce a division by zero or an unbounded score.
MIN_RETURN_VOLATILITY = 1e-6


class InferenceModelError(RuntimeError):
    """Base class for every error raised by a model implementation."""


class ModelNotLoadedError(InferenceModelError):
    """Raised when `predict` is called before `load` has completed."""


class ModelUnavailableError(InferenceModelError):
    """Raised when a model cannot be loaded, e.g. missing weights.

    The service treats this as "fall back to the heuristic model" rather than
    as a crash: a research-grade backend going missing must not take the
    process down.
    """


class InvalidContextError(InferenceModelError):
    """Raised when the supplied context cannot be used as a prompt table."""


@runtime_checkable
class InferenceModel(Protocol):
    """What the inference service needs from any model implementation.

    Matches `docs/contracts.md` §5 exactly.
    """

    @property
    def name(self) -> str:
        """Identifier of the loaded weights, e.g. ``"tabfm-v1.0.0"``.

        Reported in every signal, so it must distinguish weight versions.
        """
        ...

    async def load(self) -> None:
        """Acquire weights and make the model ready for `predict`.

        Must be safe to call more than once. Must raise `ModelUnavailableError`
        rather than panicking when the backend cannot be loaded.
        """
        ...

    def predict(self, context: pd.DataFrame, horizon: int) -> float:
        """Return the expected return for the bar `horizon` bars ahead.

        Args:
            context: Rows x 18 features, oldest first, newest last.
            horizon: Bars ahead to predict, 1 for minute-close inference.

        Returns:
            Expected return, scaled to [-1, 1].

        Raises:
            ModelNotLoadedError: If `load` has not completed.
            InvalidContextError: If `context` is not a usable prompt table.
        """
        ...

    def is_loaded(self) -> bool:
        """Whether `predict` can be called."""
        ...


class BaseInferenceModel(ABC):
    """Abstract base sharing the contract every implementation must honour.

    Subclasses implement `_load`, `_predict_raw` and `_name`. The public
    methods here add the parts that must behave identically everywhere: the
    loaded-state guard, the second-call-safe `load`, and the range clamp.
    """

    @property
    def name(self) -> str:
        """Identifier of the loaded weights."""
        return self._name

    def is_loaded(self) -> bool:
        """Whether `predict` can be called."""
        return self._loaded

    async def load(self) -> None:
        """Load the model. Safe to call repeatedly; loads at most once."""
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
        """Acquire weights. Must raise `ModelUnavailableError` on failure."""

    @abstractmethod
    def _predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        """Return the expected return, pre-scaling, in natural return units."""


def validate_context(context: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Check a context table and return it in newest-last form.

    Every model shares this so that a malformed prompt fails the same way
    regardless of backend.

    A context containing NaN in the frozen feature columns is rejected rather
    than repaired. A raw `features.Builder` table does have NaN in its
    rolling columns across the warm-up window, so the service is expected to
    hand the model already-warmed rows; silently dropping rows here would
    change what the model sees in a way nothing downstream could observe.

    Raises:
        InvalidContextError: If the table is not a usable prompt.
    """
    if horizon < 1:
        raise InvalidContextError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(context, pd.DataFrame):
        raise InvalidContextError(f"context must be a DataFrame, got {type(context).__name__}")

    missing = [c for c in FEATURE_COLUMNS if c not in context.columns]
    if missing:
        raise InvalidContextError(
            f"context is missing {len(missing)} frozen feature column(s): {missing}"
        )

    # A model predicting h bars ahead needs h realised rows after the query
    # row, so the usable prompt is one bar shorter than the table.
    usable = len(context) - horizon
    if usable < MIN_CONTEXT_ROWS:
        raise InvalidContextError(
            f"need at least {MIN_CONTEXT_ROWS + horizon} rows to predict {horizon} bar(s) "
            f"ahead, got {len(context)}"
        )

    if context.loc[:, FEATURE_COLUMNS].isna().to_numpy().any():
        raise InvalidContextError("context contains NaN in the frozen feature columns")

    return context.reset_index(drop=True)


def take_recent(context: pd.DataFrame, limit: int = MAX_CONTEXT_ROWS) -> pd.DataFrame:
    """Truncate a context table to its `limit` most recent rows.

    Deliberate truncation rather than sampling: in-context learning attends
    over the whole prompt, and a stale regime at the head of the table is
    worse than no data at all. The caller has already validated the table.
    """
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
    """Map a volatility-normalised return onto [-1, 1].

    `tanh` rather than a hard clip: a clip produces a flat region at the
    extremes, which destroys the ranking the execution service relies on
    when it compares several instruments against `score_threshold`.

    A non-finite input returns 0.0. It has to be checked before the transform,
    not after: `tanh(inf)` is 1.0, so a diverged model would otherwise be
    reported as maximum conviction and traded.
    """
    if not np.isfinite(value):
        logger.warning("non-finite model output %r, returning 0.0", value)
        return 0.0
    if not np.isfinite(softness) or softness <= 0.0:
        raise InvalidContextError(f"softness must be finite and > 0, got {softness}")
    scaled = float(np.tanh(value / softness))
    return max(-1.0, min(1.0, scaled))


#: A factory builds a fresh, unloaded model. Stored in the registry.
ModelFactory = Callable[[], InferenceModel]

_REGISTRY: dict[str, ModelFactory] = {}


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory under `name` for `config.InferenceConfig.Model`.

    Args:
        name: Selector value, e.g. ``"tabfm"`` or ``"heuristic"``.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered.
    """
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
    """Remove a registration. For tests that register throwaway models."""
    _REGISTRY.pop(name, None)


def create_model(name: str) -> InferenceModel:
    """Instantiate the model registered under `name`.

    Raises:
        ValueError: If `name` is not registered.
    """
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
    """Register the two implementations shipped with the service.

    Imported here rather than at module scope so that importing
    `services.inference.model.base` never drags in torch. A model whose
    backend failed to install must still be selectable by name so the
    service can fall back.
    """
    from .heuristic_model import HeuristicModel

    _REGISTRY.setdefault("heuristic", HeuristicModel)

    try:
        from .tabfm_model import TabFMModel

        _REGISTRY.setdefault("tabfm", TabFMModel)
    except ImportError:  # pragma: no cover - depends on the local install
        logger.warning("tabfm backend not importable; 'tabfm' is not selectable")


_register_builtin_models()
