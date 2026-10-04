from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from model.algorithms import available_algorithms, create_algorithm
from model.base import FEATURE_COLUMNS


def context(rows: int = 70) -> pd.DataFrame:
    x = np.linspace(-0.003, 0.004, rows)
    data = {name: np.zeros(rows, dtype=float) for name in FEATURE_COLUMNS}
    data["ret_1"] = x
    data["ret_5"] = np.linspace(-0.01, 0.02, rows)
    data["ret_15"] = np.linspace(-0.02, 0.04, rows)
    data["ret_60"] = np.linspace(-0.04, 0.08, rows)
    data["vol_20"] = np.full(rows, 0.002)
    data["momentum_rsi_14"] = np.linspace(-0.8, 0.8, rows)
    data["volume_ratio"] = np.ones(rows)
    data["vol_ratio"] = np.ones(rows)
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


@pytest.mark.parametrize("name", available_algorithms())
def test_every_baseline_implements_loaded_inference_model(name: str) -> None:
    import asyncio

    model = create_algorithm(name)
    assert not model.is_loaded()
    asyncio.run(model.load())
    score = model.predict(context(), horizon=1)
    assert model.is_loaded()
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


def test_moving_average_and_rsi_baselines_have_expected_direction() -> None:
    import asyncio

    up = context()
    down = up.copy()
    down.loc[:, "ret_5"] *= -1
    down.loc[:, "ret_15"] *= -1
    down.loc[:, "momentum_rsi_14"] *= -1
    momentum = create_algorithm("baseline_momentum")
    rsi = create_algorithm("baseline_rsi")
    asyncio.run(momentum.load())
    asyncio.run(rsi.load())
    assert momentum.predict(up, 1) > 0
    assert momentum.predict(down, 1) < 0
    assert rsi.predict(up, 1) < 0
    assert rsi.predict(down, 1) > 0


def test_ridge_uses_only_realized_forward_horizon_labels() -> None:
    import asyncio

    table = context()
    model = create_algorithm("baseline_ridge")
    asyncio.run(model.load())
    first = model.predict(table, horizon=2)
    changed = table.copy()
    # This is the query row's observed return and is known at query time; the
    # unobserved future is never in the context at all.
    changed.loc[changed.index[-1], "ret_1"] = 0.25
    second = model.predict(changed, horizon=2)
    assert np.isfinite(first) and np.isfinite(second)
    assert first != second


def test_algorithm_registry_names_are_stable() -> None:
    assert available_algorithms() == (
        "baseline_donchian",
        "baseline_momentum",
        "baseline_ridge",
        "baseline_rsi",
    )
