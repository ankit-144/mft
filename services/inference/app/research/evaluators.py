"""Pluggable evaluators for cost-adjusted portfolio result series."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """Portfolio observations supplied to an evaluator."""

    initial_capital: float
    equity: tuple[float, ...]
    trade_pnls: tuple[float, ...]
    trade_returns: tuple[float, ...]
    turnover: float
    fees: float
    gross_pnl: float
    gross_trade_pnls: tuple[float, ...] = ()
    equity_timestamps: tuple[datetime, ...] = ()
    timezone_name: str = "Asia/Kolkata"


@dataclass(frozen=True, slots=True)
class MetricRule:
    """Selection direction and minimum scored observations for one metric."""

    direction: str = "max"
    min_observations: int = 0
    observation_key: str = "observations"

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"} or self.min_observations < 0 or not self.observation_key:
            raise ValueError("metric rule direction must be max|min and minimum nonnegative")


class MetricEvaluator:
    """Protocol base for named metric evaluators."""

    name = "metric"

    def evaluate(self, series: MetricSeries, annualization_bars: int) -> Mapping[str, float | None]:
        """Calculate metrics from completed simulation observations."""
        raise NotImplementedError


class StandardEvaluator(MetricEvaluator):
    """Common net return, risk, downside, hit-rate, and turnover metrics."""

    name = "standard"

    def evaluate(self, series: MetricSeries, annualization_bars: int) -> Mapping[str, float | None]:
        equity = np.asarray(series.equity, dtype=float)
        returns = np.diff(equity) / equity[:-1] if equity.size > 1 else np.array([], dtype=float)
        daily_returns = _daily_returns(series, equity)
        trade_pnls = np.asarray(series.trade_pnls, dtype=float)
        trade_returns = np.asarray(series.trade_returns, dtype=float)
        gross_trade_pnls = np.asarray(series.gross_trade_pnls, dtype=float)
        positive = float(trade_pnls[trade_pnls > 0].sum()) if trade_pnls.size else 0.0
        negative = float(-trade_pnls[trade_pnls < 0].sum()) if trade_pnls.size else 0.0
        gross_positive = float(gross_trade_pnls[gross_trade_pnls > 0].sum()) if gross_trade_pnls.size else 0.0
        gross_negative = float(-gross_trade_pnls[gross_trade_pnls < 0].sum()) if gross_trade_pnls.size else 0.0
        peak = np.maximum.accumulate(equity) if equity.size else np.array([series.initial_capital])
        drawdowns = (peak - equity) / np.maximum(peak, np.finfo(float).tiny) if equity.size else np.array([0.0])
        downside = np.minimum(daily_returns, 0.0)
        net_return = float(equity[-1] / series.initial_capital - 1.0) if equity.size else 0.0
        return {
            "observations": float(max(0, equity.size - 1)),
            "daily_observations": float(daily_returns.size),
            "trades": float(trade_pnls.size),
            "wins": float((trade_pnls > 0).sum()),
            "losses": float((trade_pnls < 0).sum()),
            "hit_rate": float((trade_pnls > 0).mean()) if trade_pnls.size else 0.0,
            "gross_return": series.gross_pnl / series.initial_capital,
            "net_return": net_return,
            "mean_trade_return": float(trade_returns.mean()) if trade_returns.size else 0.0,
            "median_trade_return": float(np.median(trade_returns)) if trade_returns.size else 0.0,
            "sharpe_per_bar": _sharpe(returns),
            "sharpe_per_observation": _sharpe(returns),
            "net_sharpe": _sharpe(daily_returns) * math.sqrt(252.0),
            "downside_sharpe": (
                float(daily_returns.mean() / np.sqrt(np.mean(np.square(downside)))) * math.sqrt(252.0)
                if downside.size and np.mean(np.square(downside)) > 0 else 0.0
            ),
            "max_drawdown": float(drawdowns.max(initial=0.0)),
            "profit_factor": positive / negative if negative > 0 else None,
            "gross_profit_factor": gross_positive / gross_negative if gross_negative > 0 else None,
            "turnover": series.turnover / series.initial_capital,
            "fees": series.fees,
            "gross_pnl": series.gross_pnl,
            "net_pnl": float(equity[-1] - series.initial_capital) if equity.size else 0.0,
        }


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    return float(np.mean(returns) / deviation) if deviation > 0.0 else 0.0


def _daily_returns(series: MetricSeries, equity: np.ndarray) -> np.ndarray:
    """Return close-to-close daily equity changes, using a zero risk-free rate."""
    if not equity.size:
        return np.array([], dtype=float)
    if len(series.equity_timestamps) != equity.size:


        return np.array([], dtype=float)
    zone = ZoneInfo(series.timezone_name)
    closes: dict[object, float] = {}
    for timestamp, value in zip(series.equity_timestamps, equity):
        closes[timestamp.astimezone(zone).date()] = float(value)
    daily_equity = [series.initial_capital, *(closes[key] for key in sorted(closes))]
    values = np.asarray(daily_equity, dtype=float)
    if values.size < 2 or (values[:-1] <= 0.0).any():
        return np.array([], dtype=float)
    return np.diff(values) / values[:-1]


_EVALUATORS: dict[str, Callable[[], MetricEvaluator]] = {"standard": StandardEvaluator}
_SELECTION_RULES: dict[str, dict[str, MetricRule]] = {
    "standard": {
        "net_sharpe": MetricRule("max", 2, "daily_observations"),
        "net_return": MetricRule("max", 1, "daily_observations"),
        "downside_sharpe": MetricRule("max", 2, "daily_observations"),
        "profit_factor": MetricRule("max", 1, "trades"),
        "gross_profit_factor": MetricRule("max", 1, "trades"),
        "hit_rate": MetricRule("max", 1, "trades"),
    }
}


def register_evaluator(
    name: str,
    *,
    selection_metrics: Mapping[str, MetricRule] | None = None,
) -> Callable[[Callable[[], MetricEvaluator]], Callable[[], MetricEvaluator]]:
    """Register a metric evaluator factory; duplicate names are rejected."""
    if not name or name in _EVALUATORS:
        raise ValueError(f"metric evaluator {name!r} is empty or already registered")

    def decorate(factory: Callable[[], MetricEvaluator]) -> Callable[[], MetricEvaluator]:
        _EVALUATORS[name] = factory
        _SELECTION_RULES[name] = dict(selection_metrics or {})
        return factory

    return decorate


def create_evaluator(name: str) -> MetricEvaluator:
    """Construct a registered evaluator by stable name."""
    try:
        return _EVALUATORS[name]()
    except KeyError as err:
        raise ValueError(f"unknown evaluator {name!r}; available: {', '.join(available_evaluators())}") from err


def available_evaluators() -> tuple[str, ...]:
    """Return registered evaluator names in stable order."""
    return tuple(sorted(_EVALUATORS))


def selection_rule(evaluator: str, metric: str) -> MetricRule:
    """Return the declared ranking direction for a registered metric."""
    try:
        return _SELECTION_RULES[evaluator][metric]
    except KeyError as err:
        known = ", ".join(sorted(_SELECTION_RULES.get(evaluator, {}))) or "none"
        raise ValueError(f"metric {metric!r} is not selectable for {evaluator!r}; registered: {known}") from err


__all__ = [
    "MetricEvaluator",
    "MetricSeries",
    "MetricRule",
    "StandardEvaluator",
    "available_evaluators",
    "create_evaluator",
    "register_evaluator",
    "selection_rule",
]
