"""Chronological, cost-aware model and strategy research tools."""

from .config import CostAssumptions, ExperimentConfig
from .adapters import (
    ModelPredictorAdapter,
    Predictor,
    Strategy,
    ThresholdStrategy,
    available_strategies,
    create_strategy,
    register_strategy,
)
from .engine import CandidateConfig, ExperimentResult, ResearchError, run_experiment, run_experiment_from_store
from .evaluators import MetricRule, available_evaluators, register_evaluator
from .splits import GlobalSplitPlan, SplitPlan, plan_global_splits, plan_splits

__all__ = [
    "CostAssumptions",
    "ExperimentConfig",
    "ModelPredictorAdapter",
    "Predictor",
    "Strategy",
    "ThresholdStrategy",
    "available_strategies",
    "create_strategy",
    "register_strategy",
    "CandidateConfig",
    "ExperimentResult",
    "ResearchError",
    "SplitPlan",
    "GlobalSplitPlan",
    "MetricRule",
    "available_evaluators",
    "plan_splits",
    "plan_global_splits",
    "register_evaluator",
    "run_experiment",
    "run_experiment_from_store",
]
