"""Tests for the heuristic fallback.

The point of this model is that it is *useful*: if it always returned 0.0 the
pipeline would still pass every integration test while learning nothing. So
these tests assert directional behaviour, not just a range check.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from conftest import make_context

from model.base import MIN_CONTEXT_ROWS, InvalidContextError, ModelNotLoadedError
from model.heuristic_model import HeuristicModel


def test_scores_stay_inside_the_unit_interval(loaded_heuristic: HeuristicModel) -> None:
    rng = np.random.default_rng(5)
    for seed in range(12):
        table = make_context(rows=80, seed=seed, drift=rng.normal(0.0, 2e-4))
        value = loaded_heuristic.predict(table, 1)
        assert -1.0 <= value <= 1.0


def test_is_deterministic(loaded_heuristic: HeuristicModel) -> None:
    """A backtest that cannot be reproduced is not a backtest."""
    table = make_context()
    scores = {loaded_heuristic.predict(table, 1) for _ in range(5)}
    assert len(scores) == 1


def test_a_fresh_instance_agrees_with_a_loaded_one(
    loaded_heuristic: HeuristicModel,
) -> None:
    table = make_context()
    other = HeuristicModel()
    asyncio.run(other.load())
    assert other.predict(table, 1) == loaded_heuristic.predict(table, 1)


def test_is_not_a_no_op(loaded_heuristic: HeuristicModel) -> None:
    """Scores must actually vary with the tape, and not only by rounding."""
    rng = np.random.default_rng(9)
    scores = [
        loaded_heuristic.predict(
            make_context(rows=80, seed=seed, drift=rng.normal(0.0, 3e-4)), 1
        )
        for seed in range(25)
    ]
    assert len(set(scores)) > 20
    assert min(scores) < -0.02
    assert max(scores) > 0.02


def test_a_rising_tape_scores_above_a_falling_one(
    loaded_heuristic: HeuristicModel,
) -> None:
    """Momentum dominates at the horizon we actually trade."""
    up = loaded_heuristic.predict(make_context(drift=3e-4, volume_ratio=1.8), 1)
    down = loaded_heuristic.predict(make_context(drift=-3e-4, volume_ratio=1.8), 1)
    assert up > down


def test_volume_confirmation_moves_the_score_toward_the_trend(
    loaded_heuristic: HeuristicModel,
) -> None:
    """The same drift on heavy volume is a better bet than on thin volume."""
    heavy = loaded_heuristic.predict(make_context(drift=4e-4, volume_ratio=2.2), 1)
    thin = loaded_heuristic.predict(make_context(drift=4e-4, volume_ratio=0.3), 1)
    assert heavy > thin


def _flat_tape(rows: int = 80) -> "object":
    """A context with no displacement anywhere, and no NaN.

    A dead-flat tape is the degenerate case the volatility floor exists for.
    Every return-derived column is zeroed, including `vol_ratio`: on a real
    flat tape `vol_5 / vol_20` is 0/0, so the builder has to define it
    explicitly or the table comes out as NaN and is rejected.
    """
    table = make_context(rows=rows)
    for column in ("ret_1", "ret_5", "ret_15", "ret_60", "range_1", "body_1", "sma_gap_10"):
        table[column] = 0.0
    table["vol_5"] = 0.0
    table["vol_20"] = 0.0
    table["vol_ratio"] = 1.0
    table["momentum_rsi_14"] = 0.0
    return table


def test_flat_tape_produces_no_conviction(loaded_heuristic: HeuristicModel) -> None:
    """A tape with no displacement must not manufacture a trade."""
    assert abs(loaded_heuristic.predict(_flat_tape(), 1)) < 1e-9


def test_zero_volatility_does_not_divide_by_zero() -> None:
    """Zero volatility is the degenerate case the floors exist for."""
    model = HeuristicModel()
    asyncio.run(model.load())
    value = model.predict(_flat_tape(), 1)
    assert value == pytest.approx(0.0)
    assert np.isfinite(value)


def test_a_longer_horizon_does_not_shrink_the_signal(
    loaded_heuristic: HeuristicModel,
) -> None:
    """A multi-bar move is more dispersed, so the score should grow, not fade."""
    table = make_context(rows=90, drift=3e-4, volume_ratio=1.8)
    one_bar = abs(loaded_heuristic.predict(table, 1))
    two_bars = abs(loaded_heuristic.predict(table, 2))
    assert two_bars >= one_bar


def test_more_history_does_not_break_it(loaded_heuristic: HeuristicModel) -> None:
    """The service may send up to inference.context_rows, not exactly 60."""
    for rows in (MIN_CONTEXT_ROWS + 1, 100, 250, 500):
        value = loaded_heuristic.predict(make_context(rows=rows), 1)
        assert -1.0 <= value <= 1.0


def test_short_context_is_refused() -> None:
    model = HeuristicModel()
    asyncio.run(model.load())
    with pytest.raises(InvalidContextError):
        model.predict(make_context(rows=30), 1)


def test_predict_before_load_is_refused() -> None:
    with pytest.raises(ModelNotLoadedError):
        HeuristicModel().predict(make_context(), 1)


def test_load_is_idempotent() -> None:
    model = HeuristicModel()
    asyncio.run(model.load())
    asyncio.run(model.load())
    assert model.is_loaded() is True


def test_name_reports_its_version() -> None:
    assert HeuristicModel().name == "heuristic-v1"


def test_weights_are_reproducible_for_a_fixed_context() -> None:
    """Same input, same output, across instances and across runs."""
    table = make_context(seed=3)
    model = HeuristicModel()
    asyncio.run(model.load())
    assert model.predict(table, 1) == model.predict(table.copy(), 1)


def test_horizon_beyond_the_prompt_is_refused(loaded_heuristic: HeuristicModel) -> None:
    with pytest.raises(InvalidContextError, match="at least"):
        loaded_heuristic.predict(make_context(rows=MIN_CONTEXT_ROWS), horizon=5)
