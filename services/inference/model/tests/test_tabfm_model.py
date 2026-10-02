"""Tests for the TabFM wrapper.

Three groups, only the first of which runs unattended.

1. Licence and construction. Cheap, no backend, no weights. These are the
   reason this module exists.
2. Refusal paths. They drive `_make_backend` with a deliberately impossible
   checkpoint path or a patched loader, so they exercise the error contract
   without importing torch and without touching the Hugging Face cache.
3. Real inference. Gated behind `MFT_TABFM_WEIGHTS_TESTS=1`, never on cache
   presence alone. A populated cache is not consent: the checkpoint is ~6.6 GB
   and the development machine has 14 GB of RAM with the OOM killer active, so
   a test that silently materialises it because someone ran `make tabfm-weights`
   once is a landmine for whoever runs the suite next. Verification of real
   inference is a deliberate manual step, on a machine with every service
   stopped.

The weights are licensed `tabfm-non-commercial-v1.0`. See `Plan.md` §5.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from conftest import make_context

from model import tabfm_model
from model.base import (
    MAX_CONTEXT_ROWS,
    MIN_CONTEXT_ROWS,
    ModelNotLoadedError,
    ModelUnavailableError,
)
from model.tabfm_model import (
    ACK_ENV_VAR,
    CHECKPOINT_ENV_VAR,
    HF_REPO_ID,
    LICENCE_WARNING,
    NON_COMMERCIAL,
    WEIGHTS_FOOTPRINT_BYTES,
    WEIGHTS_LICENCE,
    WEIGHTS_VERSION,
    TabFMModel,
    weights_available,
)

#: Set to 1 by a human who has confirmed the machine can hold the checkpoint.
WEIGHTS_TESTS_ENV_VAR = "MFT_TABFM_WEIGHTS_TESTS"

OPT_IN = os.environ.get(WEIGHTS_TESTS_ENV_VAR) == "1"

#: Marks a test that drives the real `_make_backend` to prove it refuses
#: before it imports torch. Safe precisely because it never gets that far.
drives_real_refusal = pytest.mark.refusal

NOT_OPTED_IN = (
    f"real-inference tests need {WEIGHTS_TESTS_ENV_VAR}=1. The checkpoint is "
    "~6.6 GB and this suite is expected to run on a memory-constrained "
    "machine, so loading it is a deliberate manual step, never a default. "
    "See Plan.md section 5 for the licence."
)

WEIGHTS_MISSING = (
    f"TabFM weights ({WEIGHTS_LICENCE}, ~6.6 GB) are not in the local "
    "Hugging Face cache. Load them once with network access: "
    f"`make tabfm-weights`, or run services/inference/venv.sh then start the "
    "service once. Do not run this to find out."
)

needs_weights = pytest.mark.skipif(
    not OPT_IN, reason=f"{NOT_OPTED_IN} (set {WEIGHTS_TESTS_ENV_VAR}=1 to allow)"
)


def _backend_modules() -> set[str]:
    """Heavy modules whose import is what this suite is avoiding."""
    import sys

    return {name for name in ("torch", "tabfm", "safetensors") if name in sys.modules}


# --- licence ---------------------------------------------------------------


def test_weights_are_declared_non_commercial() -> None:
    """The restriction is a runtime fact, not a note in a README."""
    assert NON_COMMERCIAL is True
    assert WEIGHTS_LICENCE == "tabfm-non-commercial-v1.0"


def test_the_model_never_claims_commercial_licence() -> None:
    assert TabFMModel().is_commercially_licensed is False
    assert TabFMModel().weights_licence == WEIGHTS_LICENCE


def test_the_warning_names_the_licence_and_the_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Loading must be impossible to do without seeing the restriction.

    `_make_backend` is patched to refuse. The real one is the only thing in
    this component that touches the Hugging Face cache, and this test must not
    be a second way to reach it.
    """

    def refuse(device: str, dtype: object) -> object:
        raise ModelUnavailableError("patched: no weights in tests")

    monkeypatch.setattr(tabfm_model, "_make_backend", refuse)

    model = TabFMModel()
    with caplog.at_level(logging.WARNING, logger="mft.inference.model.tabfm"):
        with pytest.raises(ModelUnavailableError):
            asyncio.run(model.load())
    assert model.is_loaded() is False
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert WEIGHTS_LICENCE in messages
    assert ACK_ENV_VAR in messages
    assert "NON-COMMERCIAL" in messages


def test_acknowledgement_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TabFMModel()
    assert model.is_licence_acknowledged() is False
    monkeypatch.setenv(ACK_ENV_VAR, "1")
    assert model.is_licence_acknowledged() is True
    monkeypatch.setenv(ACK_ENV_VAR, "yes")
    assert model.is_licence_acknowledged() is False


def test_acknowledgement_is_not_required_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is auditable, not obstructive: research use must still work."""
    monkeypatch.delenv(ACK_ENV_VAR, raising=False)
    assert TabFMModel().is_licence_acknowledged() is False


def test_the_warning_template_is_substitutable() -> None:
    rendered = LICENCE_WARNING % (WEIGHTS_LICENCE, ACK_ENV_VAR)
    assert "%s" not in rendered
    assert WEIGHTS_LICENCE in rendered


# --- construction ----------------------------------------------------------


def test_name_reports_the_weights_version() -> None:
    """The name lands in every signal, so it must identify the checkpoint.

    `name` must not import the backend: it is read by the service, by the log
    on every signal, and by test collection on a machine where torch may not
    even be installed.
    """
    model = TabFMModel()
    assert model.name == f"tabfm-{WEIGHTS_VERSION}"
    assert model.name != "tabfm-unknown"
    before = _backend_modules()
    assert model.name
    assert _backend_modules() - before == set(), "reading .name pulled in the backend"


def test_defaults_target_cpu() -> None:
    """No GPU on the target machine, and the tables do not justify one."""
    model = TabFMModel()
    assert model._device == "cpu"
    assert model._max_context_rows == MAX_CONTEXT_ROWS
    assert model._n_estimators >= 1


def test_prompt_budget_is_bounded() -> None:
    """In-context learning costs roughly linear attention in prompt rows."""
    assert TabFMModel()._max_context_rows == MAX_CONTEXT_ROWS <= 100
    assert MIN_CONTEXT_ROWS < MAX_CONTEXT_ROWS


def test_random_state_is_fixed() -> None:
    """An ensemble with a moving seed makes every backtest irreproducible."""
    assert TabFMModel()._random_state == 42


def test_rejects_nonsense_construction() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        TabFMModel(n_estimators=0)
    with pytest.raises(ValueError, match="max_context_rows"):
        TabFMModel(max_context_rows=0)


def test_predict_before_load_is_refused() -> None:
    with pytest.raises(ModelNotLoadedError):
        TabFMModel().predict(make_context(), 1)


# --- refusal paths: never reach the cache, never import torch --------------


@drives_real_refusal
def test_a_missing_checkpoint_is_refused_before_anything_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """A mistyped checkpoint path is an error, not a 6.6 GB download.

    The guard has to fire before the torch import, or "we have the weights
    but no memory" stops being the failure this reports.
    """
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(tmp_path) + "/does-not-exist")
    with pytest.raises(ModelUnavailableError, match="not a directory"):
        asyncio.run(TabFMModel().load())


@drives_real_refusal
def test_the_checkpoint_refusal_names_the_env_var_and_the_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """The operator needs to know what to fix without reading the source."""
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(tmp_path) + "/does-not-exist")
    with pytest.raises(ModelUnavailableError) as caught:
        asyncio.run(TabFMModel().load())
    message = str(caught.value)
    assert CHECKPOINT_ENV_VAR in message
    assert HF_REPO_ID in message
    assert "heuristic" in message


@drives_real_refusal
def test_a_short_memory_machine_is_refused_before_the_weights_are_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under-resourced means typed error, not the OOM killer winning."""
    monkeypatch.delenv(CHECKPOINT_ENV_VAR, raising=False)
    monkeypatch.setattr(
        tabfm_model, "_free_memory_bytes", lambda: WEIGHTS_FOOTPRINT_BYTES // 4
    )
    with pytest.raises(ModelUnavailableError, match="free memory"):
        asyncio.run(TabFMModel().load())


def test_an_unknown_memory_answer_does_not_block_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None means "cannot check", which must not mean "refuse"."""
    monkeypatch.setattr(tabfm_model, "_free_memory_bytes", lambda: None)
    calls: list[str] = []

    def fake_backend(device: str, dtype: object) -> object:
        calls.append(device)
        return object()

    monkeypatch.setattr(tabfm_model, "_make_backend", fake_backend)
    model = TabFMModel()
    asyncio.run(model.load())
    assert calls == ["cpu"]
    assert model.is_loaded() is True


def test_weights_available_does_not_raise_without_a_cache() -> None:
    """The probe is allowed to stat; it must not import the backend."""
    before = _backend_modules()
    assert isinstance(weights_available(), bool)
    assert weights_available(checkpoint_dir="/nonexistent/path") is False
    assert _backend_modules() - before == set()


# --- real inference: manual, opt-in, never automatic -----------------------


@needs_weights
def test_the_weights_are_actually_there() -> None:
    """Opting in on a machine without the checkpoint is a configuration error."""
    assert weights_available(), WEIGHTS_MISSING


@needs_weights
def test_load_succeeds_and_is_idempotent() -> None:
    model = TabFMModel()
    asyncio.run(model.load())
    assert model.is_loaded() is True
    asyncio.run(model.load())
    assert model.is_loaded() is True


@needs_weights
def test_predict_returns_a_bounded_score() -> None:
    model = TabFMModel()
    asyncio.run(model.load())
    value = model.predict(make_context(rows=MIN_CONTEXT_ROWS + 5), 1)
    assert -1.0 <= value <= 1.0


@needs_weights
def test_predict_is_deterministic() -> None:
    model = TabFMModel()
    asyncio.run(model.load())
    table = make_context()
    scores = {model.predict(table, 1) for _ in range(3)}
    assert len(scores) == 1


@needs_weights
def test_predict_handles_a_full_prompt_window() -> None:
    """inference.context_rows is 100; the prompt must survive at that size."""
    model = TabFMModel()
    asyncio.run(model.load())
    value = model.predict(make_context(rows=MAX_CONTEXT_ROWS + 1), 1)
    assert -1.0 <= value <= 1.0


@needs_weights
@pytest.mark.slow
def test_predict_handles_a_long_history() -> None:
    """More history than the prompt budget must be truncated, not rejected."""
    model = TabFMModel()
    asyncio.run(model.load())
    value = model.predict(make_context(rows=400), 1)
    assert -1.0 <= value <= 1.0
