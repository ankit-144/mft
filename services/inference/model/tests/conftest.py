"""Shared fixtures.

Tests import the package as `model`, so `services/inference` goes on the path
rather than the repository root. The service itself is a package; running the
tests from a bare checkout should not need the Go module to be resolvable.

Nothing here may load the TabFM checkpoint. It is ~6.6 GB and the suite is
expected to run on a 14 GB machine with the OOM killer active; the real
inference tests are opt-in via `MFT_TABFM_WEIGHTS_TESTS=1` and are guarded
inside `test_tabfm_model.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _pytest.config import Config

#: Opt-in for the real-inference tests. See the `no_accidental_weights` fixture.
WEIGHTS_TESTS_ENV_VAR = "MFT_TABFM_WEIGHTS_TESTS"

INFERENCE_ROOT = Path(__file__).resolve().parents[2]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from model.base import FEATURE_COLUMNS, MIN_CONTEXT_ROWS  # noqa: E402



def make_context(
    rows: int = MIN_CONTEXT_ROWS + 5,
    seed: int = 11,
    drift: float = 0.0,
    volatility: float = 0.0015,
    volume_ratio: float = 1.2,
) -> pd.DataFrame:
    """Build a synthetic context table that satisfies the frozen schema.

    The generator returns a realistic mixture of a slowly varying level, a
    noise process, and per-row microstructure terms, so a model has something
    to actually respond to. `drift` tilts the walk so tests can check that a
    rising tape scores above a falling one.

    No column contains NaN. `validate_context` rejects NaN, and a real
    `features.Builder` table does have NaN in its rolling columns for the
    first rows of the warm-up window, so the service has to hand the model
    warmed rows rather than the raw table.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=drift, scale=volatility, size=rows)
    returns = pd.Series(steps)
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)))
    high = close.to_numpy() * (1.0 + np.abs(rng.normal(0.0, 0.0006, rows)))
    low = close.to_numpy() * (1.0 - np.abs(rng.normal(0.0, 0.0006, rows)))
    open_ = np.concatenate([[close.iloc[0]], close.to_numpy()[:-1]])
    volume = rng.lognormal(mean=12.0, sigma=0.3, size=rows)
    volume = volume / volume.mean() * volume_ratio

    table = pd.DataFrame(
        {
            "ret_1": returns,
            "ret_5": returns.rolling(5, min_periods=1).sum(),
            "ret_15": returns.rolling(15, min_periods=1).sum(),
            "ret_60": returns.rolling(60, min_periods=1).sum(),
            "vol_5": returns.rolling(5, min_periods=2).std().fillna(volatility),
            "vol_20": returns.rolling(20, min_periods=2).std().fillna(volatility),
            "range_1": (high - low) / close,
            "body_1": (close - open_) / open_,
            "upper_wick_1": np.abs(rng.normal(0.0, 0.0004, rows)),
            "lower_wick_1": np.abs(rng.normal(0.0, 0.0004, rows)),
            "volume_z_20": rng.normal(0.0, 1.0, rows),
            "volume_ratio": volume,
            "momentum_rsi_14": np.tanh(returns / volatility * 2.0),
            "sma_gap_10": (
                close - close.rolling(10, min_periods=1).mean()
            )
            / close.rolling(10, min_periods=1).mean(),
            "minute_of_session": np.arange(rows) % 375,
            "hour_of_day": 9 + (np.arange(rows) // 60) % 7,
            "spread_proxy": np.abs(close - open_) / (volume + 1.0),
        }
    )
    table["vol_ratio"] = table["vol_5"] / table["vol_20"]
    return table[list(FEATURE_COLUMNS)]


def pytest_configure(config: Config) -> None:
    """Register marks here rather than in an ini file outside this directory."""
    config.addinivalue_line("markers", "slow: exercises a full TabFM forward pass")
    config.addinivalue_line(
        "markers",
        "refusal: drives the real backend loader, which must refuse before "
        "importing torch",
    )


@pytest.fixture
def context() -> pd.DataFrame:
    """A valid 65-row context table."""
    return make_context()


@pytest.fixture
def loaded_heuristic() -> object:
    """A `HeuristicModel` that has had `load` awaited."""
    import asyncio

    from model.heuristic_model import HeuristicModel

    model = HeuristicModel()
    asyncio.run(model.load())
    return model


@pytest.fixture(autouse=True)
def no_accidental_weights(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make the TabFM checkpoint unreachable unless the run opted in.

    Autouse across the whole directory rather than declared in
    `test_tabfm_model.py`, because `test_base.py` also constructs `TabFMModel`
    and calls `create_model("tabfm")`. The per-test markers are a convention;
    this is a wall. A test that forgets `@needs_weights` fails loudly instead
    of quietly mapping ~6.6 GB on a machine that cannot hold it.

    `MFT_TABFM_WEIGHTS_TESTS=1` lifts the wall, and `@refusal` lifts it for a
    single test that deliberately drives the real loader to prove it refuses
    before it imports torch.
    """
    if os.environ.get(WEIGHTS_TESTS_ENV_VAR) == "1":
        return
    if request.node.get_closest_marker("refusal") is not None:
        return

    from model import tabfm_model

    def refuse(device: str, dtype: object) -> object:
        raise AssertionError(
            f"{request.node.name} tried to load the TabFM checkpoint without "
            f"{WEIGHTS_TESTS_ENV_VAR}=1. Mark it @needs_weights, or fix the "
            "refusal path it meant to exercise."
        )

    monkeypatch.setattr(tabfm_model, "_make_backend", refuse)
