"""The HTTP surface: `/healthz`, `/v1/context` and `/v1/predict`.

`docs/contracts.md` §7 fixes the paths and the response shapes, and the
platform's error body. Everything here is served against a `FakeStore` and a
`FakeSender`, so the tests prove the endpoints work and prove — by the count
of signals at the end — that neither of them can place an order.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import InferenceConfig
from app.main import create_app
from app.predictor import Predictor
from support import FakeSender, FakeStore, app_config, make_candles


def live_config(**overrides: Any) -> InferenceConfig:
    base = {
        "model": "heuristic",
        "execution_url": "http://execution.invalid:8080",
        "context_rows": 100,
        "horizon_bars": 1,
        "score_threshold": 0.5,
        "order_quantity": 10,
        "instruments": ["RELIANCE"],
        "dry_run": True,
    }
    base.update(overrides)
    return InferenceConfig(**base)


@pytest.fixture
def loaded_runtime(runtime_factory: Any) -> Iterator[Any]:
    """A runtime with a real heuristic model behind it and fakes either side.

    `run_scheduler=False` throughout: an endpoint test must not start a
    background loop that posts orders on a timer.
    """
    candles = make_candles(rows=200)
    store = FakeStore({"RELIANCE": candles})
    sender = FakeSender()
    predictor = Predictor.from_name("heuristic")
    runtime = runtime_factory(
        app_config(live_config()), store=store, sender=sender, predictor=predictor
    )
    app = create_app(runtime, run_scheduler=False)
    with TestClient(app) as client:
        client.runtime = runtime  # type: ignore[attr-defined]
        client.store = store  # type: ignore[attr-defined]
        client.sender = sender  # type: ignore[attr-defined]
        yield client
    asyncio.run(predictor.aclose())


# --- /healthz ------------------------------------------------------------


def test_healthz_reports_liveness_and_the_model_state(loaded_runtime: Any) -> None:
    body = loaded_runtime.get("/healthz")

    assert body.status_code == 200
    document = body.json()
    assert document["status"] == "ok"
    assert document["model_loaded"] is True
    assert document["model"] == "heuristic-v1"
    assert document["dry_run"] is True
    assert document["instruments"] == ["RELIANCE"]


def test_healthz_stays_200_while_the_model_is_loading(loaded_runtime: Any) -> None:
    """A probe that failed on an unready model would restart a young service.

    Liveness and readiness are different questions. The flag is there so a
    caller can ask the second one.
    """
    loaded_runtime.runtime.predictor._loaded = False  # noqa: SLF001 - pinned state

    body = loaded_runtime.get("/healthz")

    assert body.status_code == 200
    assert body.json()["model_loaded"] is False


# --- /v1/context ---------------------------------------------------------


def test_context_returns_the_frozen_columns_and_the_warmest_rows(loaded_runtime: Any) -> None:
    from model import FEATURE_COLUMNS

    body = loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE", "rows": 5})

    assert body.status_code == 200
    document = body.json()
    assert document["symbol"] == "RELIANCE"
    assert document["columns"] == list(FEATURE_COLUMNS)
    assert len(document["rows"]) == 5
    assert document["warmup_rows"] == 60
    assert document["as_of"].endswith("Z")
    first = document["rows"][0]
    assert set(first["values"]) == set(FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in first["values"].values())


def test_context_rows_are_ordered_and_carry_their_own_timestamp(loaded_runtime: Any) -> None:
    body = loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE", "rows": 3})
    stamps = [row["as_of"] for row in body.json()["rows"]]

    assert stamps == sorted(stamps), "oldest first, newest last"
    assert body.json()["rows"][-1]["as_of"] == body.json()["as_of"]


def test_context_drops_the_warmup_rows_the_model_would_never_see(loaded_runtime: Any) -> None:
    """What a caller sees here is what the model saw, minus the warm-up.

    200 candles build 200 rows of which the first 60 are context-only zeros.
    Handing those to a debugging eye as if they were features is how a wrong
    feature gets debugged in the wrong place.
    """
    body = loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE", "rows": 10})
    document = body.json()

    assert document["warmup_rows"] == 60
    minute = document["rows"][0]["values"]["minute_of_session"]
    assert minute != 0.0, "row 0 is the 61st bar, not the first"


def test_context_for_an_unknown_symbol_is_a_contract_error(loaded_runtime: Any) -> None:
    body = loaded_runtime.get("/v1/context", params={"symbol": "INFY"})

    assert body.status_code == 404
    document = body.json()
    assert document["error"] == "NO_CONTEXT"
    assert "INFY" in document["message"]


def test_context_refuses_a_row_count_it_cannot_serve(loaded_runtime: Any) -> None:
    assert loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE", "rows": 0}).status_code == 422
    assert (
        loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE", "rows": 5000}).status_code
        == 422
    )


def test_context_reports_a_store_failure_as_upstream(loaded_runtime: Any) -> None:
    loaded_runtime.store.fail_with = OSError("disk gone")

    body = loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE"})

    assert body.status_code == 502
    assert body.json()["error"] == "STORE_UNAVAILABLE"


# --- /v1/predict ---------------------------------------------------------


def test_predict_returns_exactly_the_contract_fields(loaded_runtime: Any) -> None:
    body = loaded_runtime.post("/v1/predict", json={"symbol": "RELIANCE"})

    assert body.status_code == 200
    document = body.json()
    assert set(document) == {"symbol", "score", "model", "as_of"}
    assert document["symbol"] == "RELIANCE"
    assert document["model"] == "heuristic-v1"
    assert document["as_of"].endswith("Z")
    assert -1.0 <= document["score"] <= 1.0


def test_predict_never_places_an_order(loaded_runtime: Any) -> None:
    """A debugging endpoint that could trade would be an order behind a curl."""
    for _ in range(5):
        loaded_runtime.post("/v1/predict", json={"symbol": "RELIANCE"})

    assert loaded_runtime.sender.count == 0


def test_predict_cannot_be_told_to_ignore_dry_run(loaded_runtime: Any) -> None:
    """`dry_run` is not a request parameter, and an extra field is a 422.

    The request model forbids extras precisely so that `{"dry_run": false}`
    cannot become a way to trade from a script.
    """
    body = loaded_runtime.post(
        "/v1/predict", json={"symbol": "RELIANCE", "dry_run": False}
    )
    assert body.status_code == 422


def test_predict_accepts_a_row_count(loaded_runtime: Any) -> None:
    body = loaded_runtime.post(
        "/v1/predict", json={"symbol": "RELIANCE", "context_rows": 80}
    )
    assert body.status_code == 200
    assert body.json()["symbol"] == "RELIANCE"


def test_predict_on_an_unloaded_model_is_503(loaded_runtime: Any) -> None:
    loaded_runtime.runtime.predictor._loaded = False  # noqa: SLF001 - pinned state

    body = loaded_runtime.post("/v1/predict", json={"symbol": "RELIANCE"})

    assert body.status_code == 503
    assert body.json()["error"] == "MODEL_NOT_LOADED"


def test_predict_on_a_warm_but_short_window_is_422(loaded_runtime: Any) -> None:
    """40 candles cannot produce a predictable row, so there is nothing to score."""
    loaded_runtime.store.add("SHORT", make_candles(rows=40, symbol="SHORT"))

    body = loaded_runtime.post("/v1/predict", json={"symbol": "SHORT"})

    assert body.status_code == 422
    assert body.json()["error"] == "BAD_CONTEXT"
    assert "warm-up" in body.json()["message"]


def test_predict_on_a_window_the_builder_rejects_is_422(loaded_runtime: Any) -> None:
    candles = make_candles(rows=200, symbol="BROKEN")
    candles[150] = candles[149]
    loaded_runtime.store.add("BROKEN", candles)

    body = loaded_runtime.post("/v1/predict", json={"symbol": "BROKEN"})

    assert body.status_code == 422
    assert body.json()["error"] == "BAD_CONTEXT"


def test_predict_on_an_unknown_symbol_is_404(loaded_runtime: Any) -> None:
    body = loaded_runtime.post("/v1/predict", json={"symbol": "NOWHERE"})

    assert body.status_code == 404
    assert body.json()["error"] == "NO_CONTEXT"


def test_predict_requires_a_symbol(loaded_runtime: Any) -> None:
    body = loaded_runtime.post("/v1/predict", json={})
    assert body.status_code == 422


def test_predict_uppercases_the_symbol(loaded_runtime: Any) -> None:
    """C7 upper-cases what it receives; the response should say what it scored."""
    body = loaded_runtime.post("/v1/predict", json={"symbol": " reliance "})
    assert body.json()["symbol"] == "RELIANCE"


# --- error shape ---------------------------------------------------------


def test_errors_use_the_platform_body_not_fastapis(loaded_runtime: Any) -> None:
    """`docs/contracts.md` §7: `{"error": code, "message": text}`.

    FastAPI's default wraps a detail in `{"detail": ...}`, so a caller that
    switches between services would have to know which convention each one
    uses. This service does not get that exception.
    """
    body = loaded_runtime.post("/v1/predict", json={"symbol": "NOWHERE"})

    document = body.json()
    assert set(document) == {"error", "message"}
    assert "detail" not in document


def test_the_openapi_document_describes_the_three_endpoints(loaded_runtime: Any) -> None:
    paths = set(loaded_runtime.get("/openapi.json").json()["paths"])
    assert {"/healthz", "/v1/context", "/v1/predict"} <= paths


# --- the production loop is not an endpoint ------------------------------


def test_the_scheduler_does_not_run_under_the_test_client(loaded_runtime: Any) -> None:
    """The production path is the scheduler, and a test must not start it.

    `make run-inference` starts it through the lifespan; these clients are
    built with `run_scheduler=False`, so no background task can fire while an
    assertion is being made.
    """
    from app.runtime import Runtime

    runtime: Runtime = loaded_runtime.runtime
    assert runtime._task is None  # noqa: SLF001 - pinned: no task was created
    assert runtime.scheduler.ticks == 0


def test_the_runtime_closes_what_it_opened(runtime_factory: Any) -> None:
    """Leaving the client runs the lifespan shutdown.

    A DuckDB connection or an httpx pool that outlives a restart is a file
    handle the next process cannot take, and this store is opened by several
    services at once.
    """
    sender = FakeSender()
    runtime = runtime_factory(app_config(live_config()), sender=sender)
    predictor = runtime.predictor

    with TestClient(create_app(runtime, run_scheduler=False)):
        assert sender.closed is False

    assert sender.closed is True
    asyncio.run(predictor.aclose())


def test_a_stale_store_does_not_look_healthy(loaded_runtime: Any) -> None:
    """Health is about this process; a stalled store is the loop's problem.

    `/healthz` reports `model_loaded` and nothing about the store, because a
    liveness probe that fails on a store hiccup restarts a service that was
    working. The loop logs and skips instead, per symbol.
    """
    loaded_runtime.store.fail_with = OSError("disk gone")

    assert loaded_runtime.get("/healthz").status_code == 200
    assert loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE"}).status_code == 502


def test_a_minute_of_scoring_leaves_the_ledger_in_the_logs(caplog, loaded_runtime: Any) -> None:
    """`run` fires on a boundary; `tick` is the same pass, called directly.

    Worth pinning that the two agree, because the tests drive `tick` and
    production drives `run`.
    """
    from app.loop import Outcome

    with caplog.at_level("INFO", logger="mft.inference.loop"):
        decisions = asyncio.run(loaded_runtime.runtime.scheduler.tick())

    assert [d.symbol for d in decisions] == ["RELIANCE"]
    assert decisions[0].outcome in {Outcome.DRY_RUN, Outcome.BELOW_THRESHOLD, Outcome.SENT}
    if decisions[0].outcome is Outcome.DRY_RUN:
        assert "would POST" in caplog.text
    assert loaded_runtime.sender.count == 0, "dry_run is still on after a tick"


def test_the_loop_and_the_endpoint_agree_on_the_context(loaded_runtime: Any) -> None:
    """One pull, one build, one prompt, whichever door it is asked through.

    Both read the same window and both report the minute it ends at, so a
    disagreement here would mean a debugging view that does not show what the
    model saw.
    """
    document = loaded_runtime.get("/v1/context", params={"symbol": "RELIANCE", "rows": 70}).json()
    prediction = loaded_runtime.post(
        "/v1/predict", json={"symbol": "RELIANCE", "context_rows": 70}
    ).json()

    assert prediction["as_of"] == document["as_of"]
    assert document["rows"][-1]["as_of"] == document["as_of"]


def test_predict_with_too_few_rows_is_refused_rather_than_truncated(loaded_runtime: Any) -> None:
    """C5's model floor applies to the endpoint too.

    A model handed fewer rows than it accepts answers from whatever it can
    attend to, which is a confident number computed from too little. The
    endpoint does not pad the prompt to make the request succeed.
    """
    body = loaded_runtime.post("/v1/predict", json={"symbol": "RELIANCE", "context_rows": 5})

    assert body.status_code == 422
    assert body.json()["error"] == "BAD_CONTEXT"


def test_the_staleness_guard_follows_the_flush_interval(tmp_path: Path) -> None:
    """A store that flushes rarely is not skipped for publishing rarely.

    The newest *published* candle lags the clock by up to one
    `storage.flush_interval_seconds`, so a guard tighter than that would skip
    every minute of a healthy system. The wiring derives the guard from the
    configured interval, and never below a five-minute floor; the behaviour is
    covered in `test_loop.py`.
    """
    from app.config import Config, StorageConfig
    from app.loop import DEFAULT_MAX_CONTEXT_AGE, staleness_guard
    from app.runtime import build_runtime

    assert DEFAULT_MAX_CONTEXT_AGE == timedelta(minutes=5)
    assert staleness_guard(300) == timedelta(minutes=6), "one flush plus a minute"
    assert staleness_guard(30) == DEFAULT_MAX_CONTEXT_AGE, "never below the floor"
    assert staleness_guard(0) == DEFAULT_MAX_CONTEXT_AGE

    config = Config(
        storage=StorageConfig(data_dir=str(tmp_path), flush_interval_seconds=600),
        inference=live_config(),
    )
    runtime = build_runtime(config)
    try:
        assert runtime.scheduler._max_context_age == timedelta(seconds=config.inference.max_context_age_seconds)  # noqa: SLF001
    finally:
        asyncio.run(runtime.shutdown())
