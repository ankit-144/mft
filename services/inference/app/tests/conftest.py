"""Pytest fixtures for the inference service tests.

The reusable builders and fakes live in `support.py` rather than here, because
C5's `model/tests/conftest.py` is also a module named `conftest`: a
`from conftest import FakeStore` in a test module resolves to whichever of the
two pytest imported first, so a run of the whole `services/inference` tree
broke the service tests. `support` is a name nothing else in the tree uses.

Two rules govern everything in this directory.

**Nothing here may load the TabFM checkpoint.** It is ~6.6 GB and this machine
has 14 GB with the OOM killer active. Every fixture that needs a model builds
the `heuristic` one, which C5 measured at ~137 MB, and `pytest_configure`
below turns any attempt to build the other into a hard failure rather than an
OOM kill. The autouse guard is the same shape as C5's: a convention is not a
wall, and a test that forgets the convention must fail loudly.

**Nothing here may reach a real service.** The execution receiver is a fake
that records what it was sent, and the store is either an in-process list or a
temporary Parquet tree. A test that places a real order is a bug in the test.

Tests import the service as `app`, which puts `services/inference` on the path
rather than the repository root: the service is run from that directory by
`make run-inference`, and the tests should exercise the same import graph.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from _pytest.config import Config as PytestConfig

INFERENCE_ROOT = Path(__file__).resolve().parents[2]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from app.config import Config, InferenceConfig  # noqa: E402
from app.predictor import Predictor  # noqa: E402
from support import FakeSender, FakeStore  # noqa: E402

#: Opt-in for anything that would load real weights. Nothing in this directory
#: sets it, and nothing in this directory should.
WEIGHTS_TESTS_ENV_VAR = "MFT_TABFM_WEIGHTS_TESTS"


def pytest_configure(config: PytestConfig) -> None:
    config.addinivalue_line("markers", "slow: takes more than a moment")


@pytest.fixture(autouse=True)
def no_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse to build a heavyweight model, whatever a test asks for.

    `create_model` is the only way this service instantiates a model, so
    refusing every name but `heuristic` here is a complete wall: no test in
    this directory can reach torch, the checkpoint, or the Hugging Face cache,
    whatever it names.
    """
    if WEIGHTS_TESTS_ENV_VAR in os.environ:
        pytest.fail(
            f"{WEIGHTS_TESTS_ENV_VAR} is set. The inference service tests are "
            "written to run without weights and must never load them; unset it."
        )

    from model import base

    def refuse(name: str) -> Any:
        raise AssertionError(
            f"a test asked for model {name!r}. This suite runs on "
            "`inference.model: heuristic` only — the TabFM checkpoint is ~6.6 GB "
            "and must never be loaded here."
        )

    monkeypatch.setattr(base, "create_model", refuse)


@pytest.fixture
def inference_config() -> InferenceConfig:
    """A config that is safe by construction: dry run on, threshold 0.5.

    `dry_run: false` appears in exactly one test, and it is a test with a
    `FakeSender` behind it. Nothing in this suite can place an order.
    """
    return InferenceConfig(
        model="heuristic",
        execution_url="http://execution.invalid:8080",
        context_rows=100,
        horizon_bars=1,
        score_threshold=0.5,
        order_quantity=10,
        instruments=["RELIANCE"],
        dry_run=True,
    )


@pytest.fixture
def loaded_predictor() -> Iterator[Predictor]:
    """A `Predictor` around the heuristic model, already loaded.

    The heuristic is the whole reason a test can run here: no checkpoint, no
    torch, no network, a deterministic score.
    """
    from model import create_model

    predictor = Predictor(create_model("heuristic"), model_name="heuristic")
    asyncio.run(predictor.load())
    yield predictor
    asyncio.run(predictor.aclose())


@pytest.fixture
def runtime_factory() -> Callable[..., Any]:
    """Build a `Runtime` from a config and fakes, for the HTTP tests."""
    from app.runtime import Runtime

    def build(config: Config, **kwargs: Any) -> Any:
        store = kwargs.pop("store", None) or FakeStore()
        sender = kwargs.pop("sender", None) or FakeSender()
        predictor = kwargs.pop("predictor", None)
        if predictor is None:
            from model import create_model

            predictor = Predictor(create_model("heuristic"), model_name="heuristic")
        from app.loop import MinuteScheduler

        # The fixture windows are anchored in the past, so the staleness guard
        # is off unless a test asks for it. `test_loop.py` covers it directly.
        kwargs.setdefault("max_context_age", None)
        scheduler = MinuteScheduler(
            config.inference,
            store,
            predictor,
            sender,
            timezone_name=config.app.timezone,
            **kwargs,
        )
        return Runtime(
            config=config,
            store=store,
            predictor=predictor,
            sender=sender,
            scheduler=scheduler,
        )

    return build

