from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.candles import Candle
from app.research import (
    CandidateConfig,
    CostAssumptions,
    ExperimentConfig,
    MetricRule,
    run_experiment,
    run_experiment_from_store,
)
from app.research.evaluators import MetricSeries, StandardEvaluator, available_evaluators, register_evaluator
from app.research.engine import _Opportunity, _simulate
from app.research.splits import plan_global_splits, plan_splits


def candles(count: int = 1200) -> list[Candle]:
    start = datetime(2026, 1, 5, 3, 45, tzinfo=timezone.utc)
    rows = []
    previous = 100.0
    for i in range(count):
        close = 100.0 * math.exp(0.0002 * i + 0.003 * math.sin(i / 7.0))
        open_ = previous
        # Keep fixture candles within exchange hours so the market-hours risk
        # gate exercises trades across more than one session.
        session, session_bar = divmod(i, 375)
        stamp = start + timedelta(days=session, minutes=session_bar)
        rows.append(Candle("TEST", stamp, open_, max(open_, close) * 1.001,
                           min(open_, close) * 0.999, close, 1000 + i % 31))
        previous = close
    return rows


def test_split_boundaries_purge_labels_and_embargo_following_context() -> None:
    horizon = 5
    plan = plan_splits(1000, horizon=horizon, first_decision=120)
    assert plan.train[0] == 120
    assert plan.train[1] + horizon <= plan.train_cut
    assert plan.validation[0] >= plan.train_cut + horizon
    assert plan.validation[1] + horizon <= plan.validation_cut
    assert plan.test[0] >= plan.validation_cut + horizon
    assert plan.test[1] == 995
    assert not (set(plan.indices("train")) & set(plan.indices("validation")))
    assert not (set(plan.indices("validation")) & set(plan.indices("test")))
    with pytest.raises(ValueError, match="purge_bars must be at least horizon"):
        plan_splits(1000, horizon=horizon, first_decision=120, purge_bars=0)


def test_global_split_cuts_keep_unequal_symbol_histories_chronological() -> None:
    start = datetime(2026, 1, 5, 3, 45, tzinfo=timezone.utc)
    stamps = {
        "LONG": [start + timedelta(minutes=i) for i in range(1000)],
        "SHORT": [start + timedelta(minutes=i) for i in range(100, 900)],
    }
    plan = plan_global_splits(stamps, horizon=2, first_decision=10)
    for symbol, split in plan.plans.items():
        assert stamps[symbol][split.test[0]] >= plan.validation_cut
        assert stamps[symbol][split.validation[0]] > plan.train_cut
        assert stamps[symbol][split.train[1] - 1] < plan.train_cut


def test_standard_metrics_match_a_hand_calculated_equity_series() -> None:
    result = StandardEvaluator().evaluate(
        MetricSeries(
            initial_capital=1000,
            equity=(1000, 1010, 990, 1030),
            trade_pnls=(20, -10),
            trade_returns=(0.02, -0.01),
            turnover=2500,
            fees=5,
            gross_pnl=15,
        ),
        annualization_bars=4,
    )
    assert result["net_return"] == pytest.approx(0.03)
    assert result["hit_rate"] == pytest.approx(0.5)
    assert result["max_drawdown"] == pytest.approx(20 / 1010)
    assert result["profit_factor"] == pytest.approx(2.0)
    assert result["turnover"] == pytest.approx(2.5)
    assert result["fees"] == 5


def test_metric_evaluators_are_pluggable() -> None:
    @register_evaluator("test_total_return")
    class TotalReturn:
        def evaluate(self, series: MetricSeries, annualization_bars: int) -> dict[str, float]:
            return {"net_return": series.equity[-1] / series.initial_capital - 1.0}

    assert "test_total_return" in available_evaluators()


@pytest.mark.asyncio
async def test_candidate_grid_and_custom_minimization_metric_select_on_validation_only() -> None:
    evaluator_name = "test_minimum_candidate_trades"

    @register_evaluator(
        evaluator_name,
        selection_metrics={"candidate_trades": MetricRule(direction="min", min_observations=2)},
    )
    class MinimumTrades:
        def evaluate(self, series: MetricSeries, annualization_bars: int) -> dict[str, float]:
            return {
                "observations": float(max(0, len(series.equity) - 1)),
                "candidate_trades": float(len(series.trade_pnls)),
            }

    cfg = ExperimentConfig(
        horizon=1,
        context_rows=80,
        threshold=0.0,
        evaluator=evaluator_name,
        selection_metric="candidate_trades",
        max_positions=1,
    )
    result = await run_experiment(
        {"TEST": candles()},
        config=cfg,
        candidates=[
            CandidateConfig("baseline_momentum", threshold=0.0),
            CandidateConfig("baseline_momentum", threshold=1.0),
        ],
    )
    assert result.selected_candidate is not None
    assert result.selected_candidate.threshold == 1.0
    assert result.test.metrics["candidate_trades"] == 0.0
    assert result.to_json()["selection"]["rule"] == {"direction": "min", "min_observations": 2}


def test_zerodha_mis_and_cnc_assumptions_are_dated_and_not_mixed() -> None:
    mis = CostAssumptions.zerodha_equity_mis(spread_bps=0.0, slippage_bps=0.0)
    cnc = CostAssumptions.zerodha_equity_cnc(spread_bps=0.0, slippage_bps=0.0)
    assert mis.effective_date == cnc.effective_date == "2026-10-04"
    assert mis.source_url == cnc.source_url == "https://zerodha.com/charges"
    assert mis.stt_buy_bps == 0 and mis.stt_sell_bps == 2.5 and mis.dp_sell_fee == 0
    assert cnc.stt_buy_bps == cnc.stt_sell_bps == 10 and cnc.dp_sell_fee == 15.34
    assert cnc.brokerage_bps == cnc.brokerage_cap == 0.0
    assert mis.leg(100_000, is_buy=True) != mis.leg(100_000, is_buy=False)
    assert cnc.leg(100_000, is_buy=False) > cnc.leg(100_000, is_buy=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_cost_and_experiment_assumptions_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        CostAssumptions(spread_bps=value)
    with pytest.raises(ValueError, match="finite"):
        ExperimentConfig(threshold=value)
    with pytest.raises(ValueError, match="finite integers"):
        ExperimentConfig(max_positions=value)
    with pytest.raises(ValueError, match="finite"):
        CostAssumptions().leg(value, is_buy=True)


def test_profit_factor_is_undefined_when_there_are_no_losses() -> None:
    metrics = StandardEvaluator().evaluate(
        MetricSeries(
            initial_capital=1000,
            equity=(1000, 1010),
            trade_pnls=(10,),
            trade_returns=(0.01,),
            turnover=100,
            fees=0,
            gross_pnl=10,
            gross_trade_pnls=(10,),
        ),
        annualization_bars=1,
    )
    assert metrics["profit_factor"] is None
    assert metrics["gross_profit_factor"] is None


def test_intraday_simulation_refuses_a_label_on_the_next_session() -> None:
    rows = candles(600)
    config = ExperimentConfig(horizon=1, context_rows=80)
    plan = plan_splits(len(rows), horizon=1, first_decision=120)
    opportunity = _Opportunity(
        "TEST", 374, rows[374].timestamp, rows[375].timestamp,
        rows[374].close, rows[375].close, 0.8, 1,
    )
    result = _simulate("test", [opportunity], 1, {"TEST": rows}, {"TEST": plan}, config)
    assert result.trades == ()
    assert result.rejections["RISK_SESSION_CROSSING"] == 1


def test_realized_loss_reduces_available_position_capital() -> None:
    rows = candles(600)
    plan = plan_splits(len(rows), horizon=1, first_decision=120)
    costs = CostAssumptions(
        brokerage_bps=0, stt_buy_bps=0, stt_sell_bps=0, spread_bps=0, slippage_bps=0,
    )
    config = ExperimentConfig(
        horizon=1, context_rows=80, initial_capital=100_000, position_fraction=0.5,
        max_position_pct=100, max_drawdown_pct=1000, daily_loss_limit=1_000_000,
        debounce_ttl_seconds=0, max_positions=1, costs=costs,
    )
    first = _Opportunity("TEST", 120, rows[120].timestamp, rows[121].timestamp, 100, 500, -0.9, -1)
    second = _Opportunity("TEST", 121, rows[121].timestamp, rows[122].timestamp, 100, 100, -0.9, -1)
    result = _simulate("test", [first, second], 2, {"TEST": rows}, {"TEST": plan}, config)
    assert len(result.trades) == 1
    assert result.trades[0].net_pnl == pytest.approx(-200_000)
    assert result.rejections["RISK_MAX_POSITION"] == 1


def test_sharpe_uses_gapped_session_closes_and_unannualized_observation_returns() -> None:
    start = datetime(2026, 1, 5, 3, 45, tzinfo=timezone.utc)
    stamps = (
        start,
        start + timedelta(minutes=1),
        start + timedelta(days=2),
        start + timedelta(days=2, minutes=1),
    )
    metrics = StandardEvaluator().evaluate(
        MetricSeries(
            initial_capital=1000.0,
            equity=(1000.0, 1050.0, 1050.0, 1000.0),
            trade_pnls=(),
            trade_returns=(),
            turnover=0.0,
            fees=0.0,
            gross_pnl=0.0,
            equity_timestamps=stamps,
        ),
        annualization_bars=94_500,
    )
    daily_returns = [0.05, (1000.0 / 1050.0) - 1.0]
    mean = sum(daily_returns) / 2
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in daily_returns))
    expected_daily = mean / sample_std * math.sqrt(252)
    assert metrics["daily_observations"] == 2
    assert metrics["net_sharpe"] == pytest.approx(expected_daily)
    assert metrics["sharpe_per_bar"] == pytest.approx(StandardEvaluator().evaluate(
        MetricSeries(1000.0, (1000.0, 1050.0, 1050.0, 1000.0), (), (), 0.0, 0.0, 0.0),
        94_500,
    )["sharpe_per_bar"])


def test_daily_sharpe_is_undefined_when_equity_timestamps_do_not_align() -> None:
    metrics = StandardEvaluator().evaluate(
        MetricSeries(
            initial_capital=1000.0,
            equity=(1000.0, 1050.0, 1100.0),
            trade_pnls=(),
            trade_returns=(),
            turnover=0.0,
            fees=0.0,
            gross_pnl=0.0,
            equity_timestamps=(datetime(2026, 1, 5, tzinfo=timezone.utc),),
        ),
        annualization_bars=94_500,
    )
    assert metrics["daily_observations"] == 0
    assert metrics["net_sharpe"] == 0.0


def test_simultaneous_entries_recheck_drawdown_after_entry_costs() -> None:
    original = candles(600)
    rows_a = original
    rows_b = [
        Candle("OTHER", row.timestamp, row.open, row.high, row.low, row.close, row.volume)
        for row in original
    ]
    plans = {
        "TEST": plan_splits(len(rows_a), horizon=1, first_decision=120),
        "OTHER": plan_splits(len(rows_b), horizon=1, first_decision=120),
    }
    costs = CostAssumptions(
        brokerage_bps=0,
        stt_buy_bps=0,
        stt_sell_bps=0,
        spread_bps=0,
        slippage_bps=0,
        fixed_per_order=15,
    )
    config = ExperimentConfig(
        horizon=1,
        context_rows=80,
        initial_capital=100_000,
        position_fraction=0.1,
        max_position_pct=100,
        max_drawdown_pct=0.02,
        max_positions=2,
        costs=costs,
    )
    stamp = rows_a[120].timestamp
    exit_stamp = rows_a[121].timestamp
    opportunities = [
        _Opportunity("TEST", 120, stamp, exit_stamp, rows_a[120].close, rows_a[121].close, 0.8, 1),
        _Opportunity("OTHER", 120, stamp, exit_stamp, rows_b[120].close, rows_b[121].close, 0.8, 1),
    ]
    result = _simulate("test", opportunities, 2, {"TEST": rows_a, "OTHER": rows_b}, plans, config)
    assert result.rejections["RISK_MAX_DRAWDOWN"] == 1


@pytest.mark.asyncio
async def test_feature_to_algorithm_to_split_portfolio_and_report_is_reproducible() -> None:
    config = ExperimentConfig(
        horizon=2,
        context_rows=80,
        threshold=0.0,
        train_fraction=0.5,
        validation_fraction=0.35,
        position_fraction=0.1,
        max_positions=1,
    )
    data = {"TEST": candles()}
    first = await run_experiment(data, config=config, algorithms=["baseline_momentum", "baseline_rsi"])
    second = await run_experiment(data, config=config, algorithms=["baseline_rsi", "baseline_momentum"])
    document = first.to_json()
    assert document["schema"] == "mft.research.v1"
    assert document["disclaimer"].startswith("Research estimates")
    assert document["data_fingerprint"] == second.data_fingerprint
    assert document["configuration_fingerprint"] == second.configuration_fingerprint
    assert first.selected_algorithm == second.selected_algorithm
    assert document["selection"]["basis"].startswith("validation only")
    assert len(document["candidates"]) == 2
    assert document["test"]["split"] == "test"
    assert document["test"]["scored_bars"] > 0
    assert document["test"]["metrics"]["trades"] > 0
    assert document["test"]["metrics"]["fees"] > 0
    final_timestamp = document["test"]["equity"][-1]["as_of"]
    assert final_timestamp >= max(trade["exit_as_of"] for trade in document["test"]["trades"])


@pytest.mark.asyncio
async def test_heldout_mutation_does_not_change_validation_or_model_selection() -> None:
    source = candles()
    base_config = ExperimentConfig(
        horizon=3, context_rows=80, threshold=0.0, max_positions=1,
        train_fraction=0.5, validation_fraction=0.35,
    )
    plan = plan_splits(
        len(source), horizon=base_config.horizon,
        first_decision=120, train_fraction=base_config.train_fraction,
        validation_fraction=base_config.validation_fraction,
        purge_bars=base_config.purge, embargo_bars=base_config.embargo,
    )
    a = await run_experiment({"TEST": source}, config=base_config, algorithms=["baseline_momentum", "baseline_ridge"])
    dirty = list(source)
    factor = 1.8
    for i in range(plan.test[0], len(dirty)):
        c = dirty[i]
        dirty[i] = Candle(c.symbol, c.timestamp, c.open * factor, c.high * factor,
                          c.low * factor, c.close * factor, c.volume)
    b = await run_experiment({"TEST": dirty}, config=base_config, algorithms=["baseline_momentum", "baseline_ridge"])
    assert a.selected_algorithm == b.selected_algorithm
    assert [x.validation.metrics for x in a.assessments] == [x.validation.metrics for x in b.assessments]
    assert a.test.metrics != b.test.metrics


@pytest.mark.asyncio
async def test_store_adapter_uses_bounded_tail_protocol() -> None:
    class Store:
        limit = 0

        async def tail(self, symbol: str, limit: int) -> list[Candle]:
            self.limit = limit
            return candles(600)

    store = Store()
    cfg = ExperimentConfig(max_bars_per_symbol=700, train_fraction=0.5, validation_fraction=0.35)
    result = await run_experiment_from_store(store, ["test"], config=cfg, algorithms=["baseline_momentum"])
    assert store.limit == 701
    assert result.test.scored_bars > 0
