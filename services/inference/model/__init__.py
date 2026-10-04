"""Inference model implementations behind the `InferenceModel` protocol."""

from __future__ import annotations

from .base import (
    FEATURE_COLUMNS,
    MAX_CONTEXT_ROWS,
    MAX_NUM_FEATURES,
    MIN_CONTEXT_ROWS,
    TARGET_COLUMN,
    BaseInferenceModel,
    InferenceModel,
    InferenceModelError,
    InvalidContextError,
    ModelNotLoadedError,
    ModelUnavailableError,
    available_models,
    create_model,
    register,
    unregister,
    validate_context,
)
from .heuristic_model import HeuristicModel
from .algorithms import register_algorithms

register_algorithms(register)

__all__ = [
    "FEATURE_COLUMNS",
    "MAX_CONTEXT_ROWS",
    "MAX_NUM_FEATURES",
    "MIN_CONTEXT_ROWS",
    "TARGET_COLUMN",
    "BaseInferenceModel",
    "HeuristicModel",
    "InferenceModel",
    "InferenceModelError",
    "InvalidContextError",
    "ModelNotLoadedError",
    "ModelUnavailableError",
    "available_models",
    "create_model",
    "register",
    "unregister",
    "validate_context",
]
