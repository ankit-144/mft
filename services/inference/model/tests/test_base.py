"""Contract tests for the protocol, the registry, and the shared helpers."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
import pytest

from conftest import make_context

from model.base import (
    FEATURE_COLUMNS,
    MAX_CONTEXT_ROWS,
    MIN_CONTEXT_ROWS,
    InferenceModel,
    InvalidContextError,
    ModelNotLoadedError,
    available_models,
    create_model,
    register,
    scale_to_unit,
    take_recent,
    unregister,
    validate_context,
)
from model.heuristic_model import HeuristicModel
from model.tabfm_model import TabFMModel


class FakeModel:
    """A minimal model that satisfies the protocol structurally.

    Deliberately does not subclass anything: if this passes the protocol
    checks, a third-party model can drop in without importing our code. That
    is the whole reason the protocol exists, given the TabFM licence.
    """

    def __init__(self) -> None:
        self._loaded = False
        self.load_calls = 0

    @property
    def name(self) -> str:
        return "fake-v0"

    async def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def predict(self, context: pd.DataFrame, horizon: int) -> float:
        return 0.25

    def is_loaded(self) -> bool:
        return self._loaded


def test_frozen_feature_schema_has_eighteen_columns() -> None:
    """The schema is frozen; changing it invalidates historical context rows."""
    assert len(FEATURE_COLUMNS) == 18
    assert len(set(FEATURE_COLUMNS)) == 18
    assert FEATURE_COLUMNS[0] == "ret_1"


def test_fake_model_satisfies_the_protocol() -> None:
    assert isinstance(FakeModel(), InferenceModel)


def test_builtin_models_satisfy_the_protocol() -> None:
    assert isinstance(HeuristicModel(), InferenceModel)
    assert isinstance(TabFMModel(), InferenceModel)


def test_base_implementations_satisfy_the_protocol() -> None:
    """Every concrete base subclass is a protocol implementation."""
    for model in (HeuristicModel(), TabFMModel()):
        assert isinstance(model, InferenceModel)


def test_predict_before_load_is_refused() -> None:
    """A model that has not loaded must not be asked for a number."""
    model = HeuristicModel()
    assert model.is_loaded() is False
    with pytest.raises(ModelNotLoadedError, match="not loaded"):
        model.predict(make_context(), 1)


def test_load_is_idempotent() -> None:
    """Startup may call load more than once; weights must not reload."""
    model = HeuristicModel()
    asyncio.run(model.load())
    asyncio.run(model.load())
    assert model.is_loaded() is True


def test_name_reports_a_weight_version() -> None:
    """The name lands in every signal, so it must be populated before load."""
    assert HeuristicModel().name
    assert TabFMModel().name.startswith("tabfm-")


# --- registry --------------------------------------------------------------


def test_factory_selects_by_name() -> None:
    assert isinstance(create_model("heuristic"), HeuristicModel)
    assert isinstance(create_model("tabfm"), TabFMModel)


def test_factory_returns_a_fresh_unloaded_instance() -> None:
    first = create_model("heuristic")
    second = create_model("heuristic")
    assert first is not second
    assert first.is_loaded() is False


def test_unknown_name_is_reported_with_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown inference model 'nope'"):
        create_model("nope")


def test_available_models_includes_the_config_values() -> None:
    """`config.InferenceConfig.Model` accepts exactly these two."""
    assert "tabfm" in available_models()
    assert "heuristic" in available_models()


def test_a_registered_model_is_selectable_and_released() -> None:
    register("test-fake")(FakeModel)
    try:
        assert isinstance(create_model("test-fake"), FakeModel)
        assert "test-fake" in available_models()
    finally:
        unregister("test-fake")
    assert "test-fake" not in available_models()
    with pytest.raises(ValueError):
        create_model("test-fake")


def test_duplicate_registration_is_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("heuristic")(FakeModel)


def test_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register("")(FakeModel)


# --- context validation ----------------------------------------------------


def test_context_below_the_warmup_floor_is_refused() -> None:
    """The first 60 rows are context-only and never prediction targets."""
    with pytest.raises(InvalidContextError, match="at least"):
        validate_context(make_context(rows=MIN_CONTEXT_ROWS), horizon=1)


def test_horizon_consumes_its_own_row() -> None:
    """Predicting h bars ahead needs h realised rows after the query row."""
    with pytest.raises(InvalidContextError, match="at least 62 rows"):
        validate_context(make_context(rows=MIN_CONTEXT_ROWS + 1), horizon=2)


def test_missing_feature_column_is_refused() -> None:
    table = make_context().drop(columns=["sma_gap_10"])
    with pytest.raises(InvalidContextError, match="missing 1 frozen feature column"):
        validate_context(table, horizon=1)


def test_nan_is_refused() -> None:
    table = make_context()
    table.loc[10, "vol_20"] = np.nan
    with pytest.raises(InvalidContextError, match="NaN"):
        validate_context(table, horizon=1)


def test_non_dataframe_is_refused() -> None:
    with pytest.raises(InvalidContextError, match="must be a DataFrame"):
        validate_context([[0.0] * 18], horizon=1)  # type: ignore[arg-type]


def test_zero_horizon_is_refused() -> None:
    with pytest.raises(InvalidContextError, match="horizon must be >= 1"):
        validate_context(make_context(), horizon=0)


def test_context_index_is_reset() -> None:
    table = make_context()
    shuffled = table.iloc[::-1].set_index(np.arange(len(table))[::-1])
    assert validate_context(shuffled, horizon=1).index.tolist() == list(range(len(table)))


# --- prompt truncation -----------------------------------------------------


def test_take_recent_keeps_the_newest_rows() -> None:
    table = make_context(rows=MAX_CONTEXT_ROWS + 40)
    trimmed = take_recent(table, limit=MAX_CONTEXT_ROWS)
    assert len(trimmed) == MAX_CONTEXT_ROWS
    assert trimmed["ret_1"].iloc[-1] == table["ret_1"].iloc[-1]
    assert trimmed.index.tolist() == list(range(MAX_CONTEXT_ROWS))


def test_take_recent_is_a_noop_when_short_enough() -> None:
    table = make_context(rows=MAX_CONTEXT_ROWS)
    assert take_recent(table, limit=MAX_CONTEXT_ROWS) is table


def test_take_recent_refuses_a_nonpositive_limit() -> None:
    with pytest.raises(InvalidContextError, match="limit must be > 0"):
        take_recent(make_context(), limit=0)


# --- scaling ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-100.0, -1.0), (100.0, 1.0), (0.0, 0.0)],
)
def test_scaling_saturates_at_the_boundaries(value: float, expected: float) -> None:
    assert scale_to_unit(value) == pytest.approx(expected)


def test_scaling_is_monotonic() -> None:
    """The service ranks instruments against a threshold, so order must hold."""
    values = [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0]
    scaled = [scale_to_unit(v) for v in values]
    assert scaled == sorted(scaled)


def test_scaling_degrades_gracefully_on_nan() -> None:
    assert scale_to_unit(float("nan")) == 0.0
    assert scale_to_unit(float("inf")) == 0.0


def test_scaling_never_leaves_the_unit_interval() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 50.0, size=2000)
    scaled = [scale_to_unit(float(v)) for v in values]
    assert all(-1.0 <= v <= 1.0 for v in scaled)
    assert any(v > 0.0 for v in scaled)
    assert any(v < 0.0 for v in scaled)
