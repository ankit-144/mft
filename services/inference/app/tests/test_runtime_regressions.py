"""Regression coverage for shared data, replay cursors and model boundaries."""

import asyncio
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from app.candles import DuckDBCandleStore
from app.config import Config, ConfigError, load_config
from app.cursors import DecisionCursors
from app.loop import MinuteScheduler, Outcome
from app.predictor import Predictor, PredictorBusyError
from model import InvalidContextError, validate_context
from support import ConstantModel, FakeSender, FakeStore, make_candles
from test_candles import write_part
from test_loop import live_config


def test_deduplicate_before_limit_with_many_overlapping_files(tmp_path):
    root = tmp_path / "candles"
    candles = make_candles(rows=30)
    write_part(root, "RELIANCE", candles, part="part-001")
    for index in range(10):
        write_part(root, "RELIANCE", [replace(candles[-1], close=3000 + index)], part=f"part-1{index:02}")
    async def run():
        store = DuckDBCandleStore(root)
        try:
            rows = await store.tail("RELIANCE", 10)
            assert len(rows) == 10
            assert rows[-1].close == 3009
            assert len({row.timestamp for row in rows}) == 10
        finally:
            await store.aclose()
    asyncio.run(run())


def test_historical_rows_warm_up_live_and_live_overrides_overlap(tmp_path):
    root = tmp_path / "candles"
    candles = make_candles(rows=150)
    source = write_part(root, "RELIANCE", candles, part="part-source")
    historical = tmp_path / "historical" / "symbol=RELIANCE" / "from=2026-08-01" / "to=2026-08-04"
    historical.mkdir(parents=True)
    source.rename(historical / "candles.parquet")
    write_part(root, "RELIANCE", [replace(candles[-1], close=9999)], part="part-live")
    async def run():
        store = DuckDBCandleStore(root)
        try:
            rows = await store.tail("RELIANCE", 200)
            assert len(rows) == 150
            assert rows[-1].close == 9999
            assert await store.tail("RELIANCE", 10) == rows[-10:]
        finally:
            await store.aclose()
    asyncio.run(run())


def test_cursors_survive_restart_and_do_not_move_backwards(tmp_path):
    path = tmp_path / "cursors.json"
    stamp = make_candles(rows=1)[0].timestamp
    cursors = DecisionCursors(str(path))
    cursors.advance("RELIANCE", stamp)
    cursors.advance("RELIANCE", stamp - timedelta(minutes=1))
    restored = DecisionCursors(str(path))
    assert restored.contains("RELIANCE", stamp)
    assert not restored.contains("RELIANCE", stamp + timedelta(minutes=1))
    assert path.stat().st_mode & 0o777 == 0o600


def test_corrupt_cursor_fails_startup(tmp_path):
    path = tmp_path / "cursors.json"
    path.write_text('{"version":2}')
    with pytest.raises(ValueError, match="version"):
        DecisionCursors(str(path))


def test_cursor_owner_excludes_second_runtime_and_reloads_after_handoff(tmp_path):
    path = str(tmp_path / "cursor.json")
    first, second = DecisionCursors(path), DecisionCursors(path)
    stamp = make_candles(rows=1)[0].timestamp
    first.claim_owner()
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            second.claim_owner()
        first.advance("RELIANCE", stamp)
    finally:
        first.close()
    second.claim_owner()
    try:
        assert second.contains("RELIANCE", stamp)
    finally:
        second.close()


def test_prediction_overload_is_rejected_and_close_cannot_admit_work():
    async def run():
        predictor = Predictor(ConstantModel(0.9), model_name="constant")
        await predictor.load()
        predictor._pending = asyncio.Semaphore(0)
        with pytest.raises(PredictorBusyError, match="capacity"):
            await predictor.predict(None, 1)
        await predictor.aclose()
        predictor._pending.release()
        with pytest.raises(RuntimeError, match="closed"):
            await predictor.predict(None, 1)
        assert predictor._pending._value == 1
    asyncio.run(run())


def test_overlapping_ticks_and_restart_score_one_bar_once(tmp_path):
    config = live_config(cursor_path=str(tmp_path / "cursor.json"))
    store = FakeStore({"RELIANCE": make_candles(rows=200)})
    sender = FakeSender()
    async def run():
        predictor = Predictor(ConstantModel(0.9), model_name="constant")
        await predictor.load()
        try:
            scheduler = MinuteScheduler(config, store, predictor, sender, max_context_age=None)
            results = await asyncio.gather(scheduler.tick(), scheduler.tick())
            assert {row[0].outcome for row in results} == {Outcome.SENT, Outcome.UNCHANGED}
            restored = MinuteScheduler(config, store, predictor, sender, max_context_age=None)
            assert (await restored.tick())[0].outcome is Outcome.UNCHANGED
            assert sender.count == 1
        finally:
            await predictor.aclose()
    asyncio.run(run())


def test_config_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text("storage:\n  data_dir: data\n")
    first = load_config(path)
    monkeypatch.chdir(directory)
    second = load_config(path)
    assert first.storage.data_dir == second.storage.data_dir == str(tmp_path / "data")
    assert first.inference.dry_run and first.execution.paper_trading


def test_credentials_are_excluded_from_mapping():
    config = Config.model_validate({"execution": {"api_token": "secret"}})
    assert "secret" not in str(config.model_dump())


def test_model_rejects_infinite_extra_or_reordered_features():
    from app.features import FeatureBuilder
    context = FeatureBuilder().build(make_candles(rows=200)).context(100)
    for corrupted in (context.assign(extra=1), context.iloc[:, ::-1], context.replace(context.iloc[0, 0], np.inf)):
        with pytest.raises(InvalidContextError):
            validate_context(corrupted, 1)


def test_horizon_cannot_exceed_context(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("inference:\n  horizon_bars: 100\n")
    with pytest.raises(ConfigError, match="horizon"):
        load_config(path)


def test_tabfm_adapter_uses_cumulative_horizon_targets_without_backend():
    from app.features import FeatureBuilder
    from model.tabfm_model import TabFMModel
    context = FeatureBuilder().build(make_candles(rows=200)).context(100)
    captured = {}
    class FakeRegressor:
        def fit(self, prompt, targets):
            captured["prompt"] = prompt
            captured["targets"] = targets.to_numpy()
        def predict(self, query):
            return np.array([0.001])
    model = TabFMModel()
    model._loaded = True
    model._regressor = lambda: FakeRegressor()
    model.predict(context, 3)
    returns = context["ret_1"].to_numpy()
    expected = np.array([sum(returns[i + 1:i + 4]) for i in range(97)])
    assert captured["targets"] == pytest.approx(expected)
    assert len(captured["prompt"]) == 97


@pytest.mark.parametrize("configured,reader_capacity,symbol_count,expected_peak", [
    (3, None, 7, 3), (64, 6, 20, 6), (64, None, 20, 8),
])
def test_bounded_symbol_reads_overlap_without_concurrent_model_calls(
    configured, reader_capacity, symbol_count, expected_peak,
):
    class SlowStore(FakeStore):
        active = 0
        peak = 0
        async def tail(self, symbol, limit):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.01)
                return await super().tail(symbol, limit)
            finally:
                self.active -= 1
    symbols = [f"S{index}" for index in range(symbol_count)]
    store = SlowStore({symbol: make_candles(rows=200, symbol=symbol) for symbol in symbols})
    if reader_capacity is not None:
        store.capacity = reader_capacity
    async def run():
        predictor = Predictor(ConstantModel(0.9), model_name="constant")
        await predictor.load()
        try:
            scheduler = MinuteScheduler(live_config(instruments=symbols, max_concurrency=configured), store, predictor, FakeSender(), max_context_age=None)
            rows = await scheduler.tick()
            assert len(rows) == symbol_count and all(row.outcome is Outcome.SENT for row in rows)
            assert store.peak == expected_peak
        finally:
            await predictor.aclose()
    asyncio.run(run())
