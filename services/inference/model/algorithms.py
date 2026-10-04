"""Classical, weight-free models for baseline research and live comparison."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from .base import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    BaseInferenceModel,
    InvalidContextError,
    returns_volatility,
    validate_context,
)


class Algorithm(Protocol):
    """A raw-return baseline that consumes the frozen feature context."""

    def predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        """Estimate a horizon return in units of recent target volatility."""
        ...


class AlgorithmModel(BaseInferenceModel):
    """InferenceModel adapter for classical, deterministic algorithms."""

    algorithm_name = "algorithm"

    def __init__(self, algorithm: Algorithm | None = None) -> None:
        self._algorithm = algorithm or self.make_algorithm()
        self._loaded = False

    @classmethod
    def make_algorithm(cls) -> Algorithm:
        """Create this model's stateless algorithm implementation."""
        raise NotImplementedError

    @property
    def _name(self) -> str:
        return f"{self.algorithm_name}-v1"

    async def _load(self) -> None:
        return None

    def _predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        table = validate_context(context, horizon)
        return float(self._algorithm.predict_raw(table, horizon))


class _Momentum:
    def predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        volatility = returns_volatility(context[TARGET_COLUMN].to_numpy(dtype=float))
        drift = 0.65 * float(context["ret_5"].iloc[-1]) / 5.0
        drift += 0.35 * float(context["ret_15"].iloc[-1]) / 15.0
        return drift * horizon / (volatility * np.sqrt(horizon))


class MovingAverageModel(AlgorithmModel):
    """Trend baseline blending short and medium moving-return horizons."""

    algorithm_name = "baseline_momentum"

    @classmethod
    def make_algorithm(cls) -> Algorithm:
        return _Momentum()


class _RSIContrarian:
    def predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        rsi = float(context["momentum_rsi_14"].iloc[-1])
        return -rsi * np.sqrt(horizon)


class RSIContrarianModel(AlgorithmModel):
    """Contrarian RSI baseline using the normalized RSI feature."""

    algorithm_name = "baseline_rsi"

    @classmethod
    def make_algorithm(cls) -> Algorithm:
        return _RSIContrarian()


class _Donchian:
    def __init__(self, window: int = 20) -> None:
        self.window = window

    def predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        if len(context) < self.window + 1:
            return 0.0


        returns = context[TARGET_COLUMN].to_numpy(dtype=float)[-(self.window + 1):]
        relative_closes = np.exp(np.cumsum(returns))
        prior = relative_closes[:-1]
        current = relative_closes[-1]
        upper, lower = float(np.max(prior)), float(np.min(prior))
        volatility = returns_volatility(returns)
        if current > upper:
            return (current / upper - 1.0) / volatility
        if current < lower:
            return (current / lower - 1.0) / volatility
        return 0.0


class DonchianBreakoutModel(AlgorithmModel):
    """Twenty-bar close-channel breakout baseline."""

    algorithm_name = "baseline_donchian"

    @classmethod
    def make_algorithm(cls) -> Algorithm:
        return _Donchian()


class _Ridge:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        if horizon < 1:
            raise InvalidContextError(f"horizon must be >= 1, got {horizon}")
        values = context.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        returns = context[TARGET_COLUMN].to_numpy(dtype=float)
        training_rows = len(context) - horizon
        if training_rows < 2:
            return 0.0
        x_train = values[:training_rows]
        y_train = np.array(
            [float(np.sum(returns[i + 1 : i + horizon + 1])) for i in range(training_rows)],
            dtype=float,
        )
        mean = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0)
        scale[scale < 1e-12] = 1.0
        x = (x_train - mean) / scale
        query = (values[-1] - mean) / scale
        x_mean, y_mean = np.mean(x, axis=0), float(np.mean(y_train))
        x_centered, y_centered = x - x_mean, y_train - y_mean
        gram = x_centered.T @ x_centered
        gram.flat[:: gram.shape[0] + 1] += self.alpha
        coefficients = np.linalg.solve(gram, x_centered.T @ y_centered)
        estimate = float((query - x_mean) @ coefficients + y_mean)
        target_vol = returns_volatility(y_train)
        return estimate / target_vol


class RidgeReturnModel(AlgorithmModel):
    """Regularized least-squares forecast of the next horizon log return."""

    algorithm_name = "baseline_ridge"

    @classmethod
    def make_algorithm(cls) -> Algorithm:
        return _Ridge()


ALGORITHMS: dict[str, Callable[[], AlgorithmModel]] = {}


def register_algorithm(name: str) -> Callable[[Callable[[], AlgorithmModel]], Callable[[], AlgorithmModel]]:
    """Register a named predictor for research comparisons and model selection."""
    if not name.strip() or name in ALGORITHMS:
        raise ValueError(f"algorithm {name!r} is empty or already registered")

    def decorate(factory: Callable[[], AlgorithmModel]) -> Callable[[], AlgorithmModel]:
        ALGORITHMS[name] = factory
        return factory

    return decorate


def unregister_algorithm(name: str) -> None:
    """Remove an algorithm registration, primarily for isolated tests."""
    ALGORITHMS.pop(name, None)


for _model_class in (MovingAverageModel, RSIContrarianModel, DonchianBreakoutModel, RidgeReturnModel):
    register_algorithm(_model_class.algorithm_name)(_model_class)


def register_algorithms(register_model: Callable[[str], Callable[[Callable[[], object]], object]]) -> None:
    """Register baseline model factories with the shared InferenceModel registry."""
    for name, factory in ALGORITHMS.items():
        try:
            register_model(name)(factory)
        except ValueError as err:
            if "already registered" not in str(err):
                raise


def create_algorithm(name: str) -> AlgorithmModel:
    """Create a named classical baseline without importing a model backend."""
    try:
        return ALGORITHMS[name]()
    except KeyError as err:
        raise ValueError(f"unknown algorithm {name!r}; available: {', '.join(sorted(ALGORITHMS))}") from err


def available_algorithms() -> tuple[str, ...]:
    """Return the registered, deterministic baseline names."""
    return tuple(sorted(ALGORITHMS))


__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "AlgorithmModel",
    "DonchianBreakoutModel",
    "MovingAverageModel",
    "RSIContrarianModel",
    "RidgeReturnModel",
    "available_algorithms",
    "create_algorithm",
    "register_algorithms",
    "register_algorithm",
    "unregister_algorithm",
]
