"""Chronological, deterministic model comparison and cost-aware simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time as clock_time, timezone
import hashlib
import json
import logging
import math
from datetime import timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from model.algorithms import available_algorithms, create_algorithm

from ..candles import Candle
from ..features import FeatureBuilder, WARMUP_ROWS
from .adapters import ModelPredictorAdapter, Strategy, create_strategy
from .config import ExperimentConfig
from .evaluators import MetricRule, MetricSeries, create_evaluator, selection_rule
from .splits import GlobalSplitPlan, SplitPlan, plan_global_splits

logger = logging.getLogger("mft.inference.research")
MIN_CONTEXT_ROWS = 60
DISCLAIMER = "Research estimates only. Historical or synthetic performance does not guarantee future results."


class CandleSource(Protocol):
    """Read an ascending tail of candles for one symbol."""

    async def tail(self, symbol: str, limit: int) -> list[Candle]:
        """Return up to limit candles, oldest first."""
        ...


class ResearchError(RuntimeError):
    """Raised when an experiment cannot be run without violating its design."""


@dataclass(frozen=True, slots=True)
class TradeResult:
    """One simulated round trip, including stated costs."""

    symbol: str
    entry_as_of: datetime
    exit_as_of: datetime
    side: str
    score: float
    entry_price: float
    exit_price: float
    notional: float
    gross_pnl: float
    fees: float
    net_pnl: float

    def to_json(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "entry_as_of": _utc(self.entry_as_of),
            "exit_as_of": _utc(self.exit_as_of),
            "side": self.side,
            "score": _round(self.score),
            "entry_price": _round(self.entry_price),
            "exit_price": _round(self.exit_price),
            "notional": _round(self.notional),
            "gross_pnl": _round(self.gross_pnl),
            "fees": _round(self.fees),
            "net_pnl": _round(self.net_pnl),
        }


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Metrics and trades for one algorithm in one chronological split."""

    split: str
    metrics: Mapping[str, float | None]
    trades: tuple[TradeResult, ...]
    equity: tuple[tuple[datetime, float], ...]
    scored_bars: int
    rejections: Mapping[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "split": self.split,
            "metrics": {k: _round(v) for k, v in sorted(self.metrics.items())},
            "trades": [t.to_json() for t in self.trades],
            "equity": [{"as_of": _utc(t), "value": _round(v)} for t, v in self.equity],
            "scored_bars": self.scored_bars,
            "rejections": dict(sorted(self.rejections.items())),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmAssessment:
    """Training and validation scores for a candidate algorithm."""

    algorithm: str
    train: SplitEvaluation
    validation: SplitEvaluation
    candidate: "CandidateConfig | None" = None

    def to_json(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "parameters": self.candidate.to_json() if self.candidate else {},
            "train": self.train.to_json(),
            "validation": self.validation.to_json(),
        }


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """One model/strategy parameter tuple compared using validation data."""

    algorithm: str
    threshold: float | None = None
    strategy_options: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise ValueError("candidate algorithm must be nonempty")
        if self.threshold is not None and (not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0):
            raise ValueError("candidate threshold must be finite and in [0, 1]")
        for key, value in self.strategy_options:
            if not key.strip() or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError("candidate strategy option names must be nonempty and numeric values finite")

    def to_json(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "strategy_options": dict(self.strategy_options),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Reproducible candidate comparison and one held-out evaluation."""

    config: ExperimentConfig
    data_fingerprint: str
    configuration_fingerprint: str
    split_plans: Mapping[str, SplitPlan]
    assessments: tuple[AlgorithmAssessment, ...]
    selected_algorithm: str
    selection_metric: str
    test: SplitEvaluation
    selected_candidate: CandidateConfig | None = None
    source: str = "candles"
    global_train_cut: datetime | None = None
    global_validation_cut: datetime | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "mft.research.v1",
            "disclaimer": DISCLAIMER,
            "source": self.source,
            "holding_policy": (
                "Intraday candidates whose label bar crosses the session/date boundary are rejected."
                if self.config.holding_policy == "intraday"
                else "Delivery candidates may remain open across session/date boundaries."
            ),
            "metric_conventions": {
                "net_sharpe": "daily close-to-close net returns, annualized by sqrt(252), risk-free rate zero",
                "downside_sharpe": "daily downside net-return ratio, annualized by sqrt(252), risk-free rate zero",
                "sharpe_per_bar": "unannualized Sharpe per observed equity step; timestamps may be irregular",
                "profit_factor": "undefined (JSON null) when there are no losing trades",
            },
            "config": self.config.to_json(),
            "configuration_fingerprint": self.configuration_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "splits": {
                symbol: {
                    "first_decision": p.first_decision,
                    "stop": p.stop,
                    "train": list(p.train),
                    "validation": list(p.validation),
                    "test": list(p.test),
                    "train_cut": p.train_cut,
                    "validation_cut": p.validation_cut,
                    "purge_bars": p.purge_bars,
                    "embargo_bars": p.embargo_bars,
                }
                for symbol, p in sorted(self.split_plans.items())
            },
            "global_boundaries": {
                "train_cut": _utc(self.global_train_cut) if self.global_train_cut else None,
                "validation_cut": _utc(self.global_validation_cut) if self.global_validation_cut else None,
            },
            "selection": {
                "algorithm": self.selected_algorithm,
                "parameters": self.selected_candidate.to_json() if self.selected_candidate else {},
                "metric": self.selection_metric,
                "rule": {
                    "direction": selection_rule(self.config.evaluator, self.selection_metric).direction,
                    "min_observations": selection_rule(self.config.evaluator, self.selection_metric).min_observations,
                },
                "basis": (
                    "validation only; candidates with undefined metrics or too few observations were ineligible; "
                    "the held-out test split was evaluated once for the selected candidate"
                ),
            },
            "candidates": [a.to_json() for a in self.assessments],
            "test": self.test.to_json(),
        }


@dataclass(frozen=True, slots=True)
class _Opportunity:
    symbol: str
    index: int
    as_of: datetime
    exit_as_of: datetime
    entry_price: float
    exit_price: float
    score: float
    side: int


@dataclass(slots=True)
class _Position:
    opportunity: _Opportunity
    notional: float
    entry_fee: float


async def run_experiment_from_store(
    source: CandleSource,
    symbols: Sequence[str],
    *,
    config: ExperimentConfig = ExperimentConfig(),
    algorithms: Sequence[str] | None = None,
    candidates: Sequence[CandidateConfig] | None = None,
    builder: FeatureBuilder | None = None,
) -> ExperimentResult:
    """Load bounded candle tails from a storage adapter and run research."""
    normalized = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not normalized:
        raise ResearchError("at least one nonempty symbol is required")
    candle_map: dict[str, list[Candle]] = {}
    for symbol in normalized:
        candles = await source.tail(symbol, config.max_bars_per_symbol + 1)
        if len(candles) > config.max_bars_per_symbol:
            raise ResearchError(f"{symbol} exceeds max_bars_per_symbol={config.max_bars_per_symbol}")
        if candles:
            candle_map[symbol] = candles
    return await run_experiment(
        candle_map, config=config, algorithms=algorithms, candidates=candidates,
        builder=builder, source="parquet",
    )


async def run_experiment(
    symbol_candles: Mapping[str, Sequence[Candle]],
    *,
    config: ExperimentConfig = ExperimentConfig(),
    algorithms: Sequence[str] | None = None,
    candidates: Sequence[CandidateConfig] | None = None,
    builder: FeatureBuilder | None = None,
    source: str = "provided",
) -> ExperimentResult:
    """Compare algorithms chronologically and test only the validation winner."""
    if not symbol_candles:
        raise ResearchError("no candles supplied")
    if algorithms is not None and candidates is not None:
        raise ResearchError("provide algorithms or candidate configurations, not both")
    specs = tuple(candidates or (CandidateConfig(name) for name in (algorithms or available_algorithms())))
    if not specs:
        raise ResearchError("no algorithms selected")
    specs = tuple(sorted(specs, key=lambda x: json.dumps(x.to_json(), sort_keys=True, separators=(",", ":"))))
    unknown = sorted({candidate.algorithm for candidate in specs} - set(available_algorithms()))
    if unknown:
        raise ResearchError(f"unknown algorithm(s): {', '.join(unknown)}")
    if config.strategy != "threshold" and any(candidate.threshold is not None for candidate in specs):
        raise ResearchError("candidate threshold grids require the built-in threshold strategy")
    try:
        rank_rule = selection_rule(config.evaluator, config.selection_metric)
    except ValueError as err:
        raise ResearchError(str(err)) from err

    feature_builder = builder or FeatureBuilder()
    candles = {s: list(rows) for s, rows in sorted(symbol_candles.items()) if rows}
    if not candles:
        raise ResearchError("no nonempty candle series supplied")
    if any(len(rows) > config.max_bars_per_symbol for rows in candles.values()):
        raise ResearchError(f"a symbol exceeds max_bars_per_symbol={config.max_bars_per_symbol}")
    frames: dict[str, pd.DataFrame] = {}
    timestamp_map = {symbol: [c.timestamp for c in rows] for symbol, rows in candles.items()}
    for symbol, rows in candles.items():
        try:
            frames[symbol] = feature_builder.build(rows).frame
        except (ValueError, RuntimeError) as err:
            raise ResearchError(f"cannot prepare {symbol}: {err}") from err
    try:
        global_plan: GlobalSplitPlan = plan_global_splits(
            timestamp_map,
            horizon=config.horizon,
            first_decision=WARMUP_ROWS + MIN_CONTEXT_ROWS + config.horizon - 1,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            purge_bars=config.purge,
            embargo_bars=config.embargo,
        )
    except ValueError as err:
        raise ResearchError(f"cannot construct global chronological splits: {err}") from err
    plans: dict[str, SplitPlan] = dict(global_plan.plans)

    assessments: list[AlgorithmAssessment] = []
    for spec in specs:
        strategy = _strategy_for(spec, config)
        model = ModelPredictorAdapter(create_algorithm(spec.algorithm))
        await model.load()
        train_opps, train_scored = _score_split("train", model, strategy, candles, frames, plans, config)
        val_opps, val_scored = _score_split("validation", model, strategy, candles, frames, plans, config)
        assessments.append(
            AlgorithmAssessment(
                algorithm=spec.algorithm,
                train=_simulate("train", train_opps, train_scored, candles, plans, config),
                validation=_simulate("validation", val_opps, val_scored, candles, plans, config),
                candidate=spec,
            )
        )
    eligible = [
        item for item in assessments
        if item.validation.metrics.get(rank_rule.observation_key) is not None
        and float(item.validation.metrics[rank_rule.observation_key]) >= rank_rule.min_observations
        and item.validation.metrics.get(config.selection_metric) is not None
        and math.isfinite(float(item.validation.metrics[config.selection_metric]))
    ]
    if not eligible:
        raise ResearchError(
            f"no candidate has a defined {config.selection_metric!r} with at least "
            f"{rank_rule.min_observations} {rank_rule.observation_key}"
        )
    assessments.sort(
        key=lambda x: (
            -_metric_value(x.validation.metrics, config.selection_metric, rank_rule)
            if rank_rule.direction == "max"
            else _metric_value(x.validation.metrics, config.selection_metric, rank_rule),
            json.dumps(x.candidate.to_json() if x.candidate else {}, sort_keys=True),
        )
    )
    winner = assessments[0]

    selected_spec = winner.candidate or CandidateConfig(winner.algorithm)
    final_model = ModelPredictorAdapter(create_algorithm(winner.algorithm))
    await final_model.load()
    final_strategy = _strategy_for(selected_spec, config)
    test_opps, test_scored = _score_split("test", final_model, final_strategy, candles, frames, plans, config)
    test_result = _simulate("test", test_opps, test_scored, candles, plans, config)
    config_bytes = json.dumps(
        {"config": config.to_json(), "candidates": [x.to_json() for x in specs]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    data_bytes = _canonical_data(candles)
    return ExperimentResult(
        config=config,
        data_fingerprint=hashlib.sha256(data_bytes).hexdigest(),
        configuration_fingerprint=hashlib.sha256(config_bytes).hexdigest(),
        split_plans=plans,
        assessments=tuple(assessments),
        selected_algorithm=winner.algorithm,
        selection_metric=config.selection_metric,
        test=test_result,
        selected_candidate=selected_spec,
        source=source,
        global_train_cut=global_plan.train_cut,
        global_validation_cut=global_plan.validation_cut,
    )


def _strategy_for(spec: CandidateConfig, config: ExperimentConfig) -> Strategy:
    options: dict[str, object] = dict(config.strategy_options)
    options.update(dict(spec.strategy_options))
    if config.strategy == "threshold":
        options.setdefault("threshold", config.threshold)
        if spec.threshold is not None:
            options["threshold"] = spec.threshold
        options.setdefault("long_only", config.long_only)
    try:
        return create_strategy(config.strategy, options)
    except ValueError as err:
        raise ResearchError(str(err)) from err


def _score_split(
    split: str,
    model: ModelPredictorAdapter,
    strategy: Strategy,
    candle_map: Mapping[str, Sequence[Candle]],
    frames: Mapping[str, pd.DataFrame],
    plans: Mapping[str, SplitPlan],
    config: ExperimentConfig,
) -> tuple[list[_Opportunity], int]:
    opportunities: list[_Opportunity] = []
    scored = 0
    for symbol in sorted(candle_map):
        rows, frame, plan = candle_map[symbol], frames[symbol], plans[symbol]
        for i in plan.indices(split):
            context = frame.iloc[: i + 1].iloc[WARMUP_ROWS:]
            if len(context) > config.context_rows:
                context = context.iloc[-config.context_rows:]
            context = context.reset_index(drop=True)
            score = float(model.predict(context, config.horizon))
            scored += 1
            if not math.isfinite(score):
                continue
            side = strategy.side(score)
            if side == 0:
                continue
            opportunities.append(
                _Opportunity(
                    symbol=symbol,
                    index=i,
                    as_of=rows[i].timestamp,
                    exit_as_of=rows[i + config.horizon].timestamp,
                    entry_price=rows[i].close,
                    exit_price=rows[i + config.horizon].close,
                    score=score,
                    side=side,
                )
            )
    return opportunities, scored


def _simulate(
    split: str,
    opportunities: Sequence[_Opportunity],
    scored: int,
    candle_map: Mapping[str, Sequence[Candle]],
    plans: Mapping[str, SplitPlan],
    config: ExperimentConfig,
) -> SplitEvaluation:
    by_time: dict[datetime, list[_Opportunity]] = {}
    cross_session = 0
    for item in opportunities:
        if config.holding_policy == "intraday":
            entry_date = item.as_of.astimezone(ZoneInfo(config.timezone_name)).date()
            exit_local = item.exit_as_of.astimezone(ZoneInfo(config.timezone_name))
            if exit_local.date() != entry_date or not _market_open(item.exit_as_of, config):
                cross_session += 1
                continue
        by_time.setdefault(item.as_of, []).append(item)
    timeline: dict[datetime, list[tuple[str, float]]] = {}
    for symbol, rows in candle_map.items():
        lo, hi = getattr(plans[symbol], split)
        final_idx = min(len(rows) - 1, hi + config.horizon - 1)
        for row in rows[lo : final_idx + 1]:
            timeline.setdefault(row.timestamp, []).append((symbol, row.close))
    times = sorted(set(timeline) | set(by_time))
    cash = config.initial_capital
    marks: dict[str, float] = {}
    active: list[_Position] = []
    trades: list[TradeResult] = []
    equity: list[tuple[datetime, float]] = []
    fees_total = gross_total = turnover = 0.0
    dp_debits: set[tuple[str, date]] = set()
    rejections: Counter[str] = Counter()
    if cross_session:
        rejections["RISK_SESSION_CROSSING"] = cross_session
    debounce: dict[tuple[str, int], datetime] = {}
    current_day: date | None = None
    realised_today = 0.0
    peak_equity = config.initial_capital
    if times:
        equity.append((times[0] - timedelta(microseconds=1), config.initial_capital))
    notional = config.initial_capital * config.position_fraction

    for moment in times:
        local = moment.astimezone(ZoneInfo(config.timezone_name))
        if current_day != local.date():
            current_day = local.date()
            realised_today = 0.0
        for symbol, price in sorted(timeline.get(moment, [])):
            marks[symbol] = price
        matured = [p for p in active if p.opportunity.exit_as_of <= moment]
        for position in sorted(matured, key=lambda p: (p.opportunity.exit_as_of, p.opportunity.symbol)):
            op = position.opportunity
            exit_notional = position.notional * op.exit_price / op.entry_price
            gross = position.notional * op.side * (op.exit_price / op.entry_price - 1.0)
            is_sell = op.side > 0
            trade_date = op.exit_as_of.astimezone(ZoneInfo(config.timezone_name)).date()
            dp_key = (op.symbol, trade_date)
            include_dp = is_sell and dp_key not in dp_debits
            exit_fee = config.costs.leg(exit_notional, is_buy=not is_sell, apply_dp_fee=include_dp)
            if is_sell and config.costs.dp_sell_fee > 0:
                dp_debits.add(dp_key)
            total_fees = position.entry_fee + exit_fee
            cash += gross - exit_fee
            realised_today += gross - exit_fee
            fees_total += exit_fee
            gross_total += gross
            turnover += position.notional + exit_notional
            trades.append(TradeResult(op.symbol, op.as_of, op.exit_as_of, "BUY" if op.side > 0 else "SELL", op.score, op.entry_price, op.exit_price, position.notional, gross, total_fees, gross - total_fees))
            active.remove(position)

        occupied = {p.opportunity.symbol for p in active}
        active_notional = sum(p.notional for p in active)
        mark_equity = _marked_equity(cash, active, marks, config)
        peak_equity = max(peak_equity, mark_equity)
        for op in sorted(by_time.get(moment, []), key=lambda x: x.symbol):
            drawdown_pct = (peak_equity - mark_equity) / peak_equity * 100.0 if peak_equity else 0.0
            if not _market_open(moment, config):
                rejections["RISK_MARKET_CLOSED"] += 1
                continue
            if op.symbol in occupied or len(active) >= config.max_positions:
                rejections["RISK_MAX_POSITIONS"] += 1
                continue
            if drawdown_pct > config.max_drawdown_pct:
                rejections["RISK_MAX_DRAWDOWN"] += 1
                continue
            if realised_today < -config.daily_loss_limit:
                rejections["RISK_DAILY_LOSS"] += 1
                continue
            debounce_key = (op.symbol, op.side)
            prior = debounce.get(debounce_key)
            if prior is not None and (moment - prior).total_seconds() < config.debounce_ttl_seconds:
                rejections["RISK_DEBOUNCED"] += 1
                continue
            if mark_equity > 0.0 and notional / mark_equity * 100.0 > config.max_position_pct:
                rejections["RISK_MAX_POSITION"] += 1
                continue
            shares = math.floor(notional / op.entry_price)
            if shares < 1:
                rejections["RISK_BAD_QUANTITY"] += 1
                continue
            allocated = shares * op.entry_price
            if active_notional + allocated > max(0.0, cash) + 1e-9:
                rejections["RISK_MAX_POSITION"] += 1
                continue
            notional_for_leg = allocated
            entry_fee = config.costs.leg(notional_for_leg, is_buy=op.side > 0)
            cash -= entry_fee
            realised_today -= entry_fee
            fees_total += entry_fee
            active.append(_Position(op, notional_for_leg, entry_fee))
            occupied.add(op.symbol)
            active_notional += notional_for_leg
            debounce[debounce_key] = moment


            mark_equity = _marked_equity(cash, active, marks, config)
            peak_equity = max(peak_equity, mark_equity)

        equity.append((moment, _marked_equity(cash, active, marks, config)))

    series = MetricSeries(
        initial_capital=config.initial_capital,
        equity=tuple(value for _, value in equity),
        trade_pnls=tuple(t.net_pnl for t in trades),
        trade_returns=tuple(t.net_pnl / t.notional for t in trades if t.notional),
        turnover=turnover,
        fees=fees_total,
        gross_pnl=gross_total,
        gross_trade_pnls=tuple(t.gross_pnl for t in trades),
        equity_timestamps=tuple(moment for moment, _ in equity),
        timezone_name=config.timezone_name,
    )
    metrics = create_evaluator(config.evaluator).evaluate(series, config.annualization_bars)
    return SplitEvaluation(split, dict(metrics), tuple(trades), tuple(equity), scored, dict(rejections))


def _marked_equity(
    cash: float,
    active: Sequence[_Position],
    marks: Mapping[str, float],
    config: ExperimentConfig,
) -> float:
    """Return cash plus open P&L after one estimated liquidation leg per position."""
    value = cash
    for position in active:
        op = position.opportunity
        current = marks.get(op.symbol, op.entry_price)
        mark_notional = position.notional * current / op.entry_price
        value += position.notional * op.side * (current / op.entry_price - 1.0)
        value -= config.costs.leg(mark_notional, is_buy=op.side < 0, apply_dp_fee=False)
    return value


def _market_open(moment: datetime, config: ExperimentConfig) -> bool:
    local = moment.astimezone(ZoneInfo(config.timezone_name))
    if local.weekday() >= 5 or local.date().isoformat() in config.market_holidays:
        return False
    return clock_time(9, 15) <= local.time().replace(tzinfo=None) < clock_time(15, 30)


def _metric_value(metrics: Mapping[str, float | None], name: str, rule: MetricRule) -> float:
    observations = float(metrics.get(rule.observation_key, 0.0) or 0.0)
    raw = metrics.get(name)
    invalid = float("-inf") if rule.direction == "max" else float("inf")
    if observations < rule.min_observations or raw is None:
        return invalid
    value = float(raw)
    return value if math.isfinite(value) else invalid


def _canonical_data(candles: Mapping[str, Sequence[Candle]]) -> bytes:
    digest = hashlib.sha256()
    for symbol, rows in sorted(candles.items()):
        digest.update(symbol.encode())
        digest.update(b"\0")
        for candle in rows:
            record = (
                _utc(candle.timestamp), repr(candle.open), repr(candle.high),
                repr(candle.low), repr(candle.close), str(candle.volume),
            )
            digest.update("|".join(record).encode())
            digest.update(b"\n")
    return digest.digest()


def _utc(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 10) if math.isfinite(float(value)) else 0.0


__all__ = [
    "AlgorithmAssessment",
    "CandleSource",
    "CandidateConfig",
    "ExperimentResult",
    "ResearchError",
    "SplitEvaluation",
    "TradeResult",
    "run_experiment",
    "run_experiment_from_store",
]
