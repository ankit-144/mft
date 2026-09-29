"""The execution client: what is retried, what is not, and what is a success.

No test here opens a socket. `httpx.MockTransport` answers every request in
process, and the `FakeSender` behind the loop is a list. A test that can place
a real order is a bug in the test, and `docs/contracts.md` is explicit that
nothing in the suite should.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.execution import (
    SIGNALS_PATH,
    ExecutionClient,
    ExecutionError,
)
from app.signals import Signal

AS_OF = datetime(2026, 9, 29, 10, 31, tzinfo=timezone.utc)


def a_signal(score: float = 0.72) -> Signal:
    return Signal.from_score(
        symbol="RELIANCE",
        score=score,
        price=2934.5,
        quantity=10,
        model="heuristic-v1",
        as_of=AS_OF,
    )


def client_for(
    handler: Any,
    *,
    max_attempts: int = 3,
    backoff: float = 0.0,
    base_url: str = "http://execution.invalid:8080",
) -> tuple[ExecutionClient, list[float]]:
    """An `ExecutionClient` over a mock transport, plus a recorded sleep log."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, timeout=5.0)
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    client = ExecutionClient(
        base_url,
        client=http,
        max_attempts=max_attempts,
        backoff_seconds=backoff,
        sleep=sleep,
    )
    return client, slept


def accepted(order_id: str = "4412", status: str = "FILLED") -> httpx.Response:
    return httpx.Response(
        202, json={"order_id": order_id, "status": status, "score": 0.72}
    )


def test_an_accepted_signal_reports_its_order_id() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return accepted()

    client, _ = client_for(handler)
    ack = asyncio.run(client.submit(a_signal()))

    assert ack.order_id == "4412"
    assert ack.status == "FILLED"
    assert ack.duplicate is False
    assert len(seen) == 1, "an accepted signal is not retried"


def test_the_post_goes_to_the_contract_path_with_the_contract_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return accepted()

    client, _ = client_for(handler)
    asyncio.run(client.submit(a_signal()))

    request = seen[0]
    assert str(request.url) == f"http://execution.invalid:8080{SIGNALS_PATH}"
    assert request.method == "POST"
    import json

    body = json.loads(request.content)
    assert body["idempotency_key"] == "RELIANCE:BUY:20260929T1031"
    assert body["as_of"] == "2026-09-29T10:31:00Z"
    assert body["side"] == "BUY"


def test_a_replayed_key_is_a_success_not_a_failure() -> None:
    """C7 answers 409 with the original order id; that is the retry working.

    Inference cannot tell a lost response from a lost request. Treating the
    409 as an error would log a phantom fault on every slow minute.
    """
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection reset by peer", request=request)
        return httpx.Response(
            409,
            json={
                "error": "RISK_DUPLICATE",
                "message": "key already processed",
                "order_id": "4412",
            },
        )

    client, slept = client_for(handler, backoff=0.01)
    ack = asyncio.run(client.submit(a_signal()))

    assert ack.duplicate is True
    assert ack.order_id == "4412"
    assert len(attempts) == 2
    assert len(slept) == 1


def test_a_transport_failure_is_retried_with_backoff() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("timed out", request=request)
        return accepted("9999")

    client, slept = client_for(handler, backoff=0.5)
    ack = asyncio.run(client.submit(a_signal()))

    assert ack.order_id == "9999"
    assert len(attempts) == 3
    assert len(slept) == 2, "one sleep between each pair of attempts"
    # The first backoff is `backoff_seconds`; the second doubles it. Both are
    # jittered down to zero by full jitter, so only the ceilings are pinned.
    assert slept[0] <= 0.5
    assert slept[1] <= 1.0
    assert all(delay > 0.0 for delay in slept)


def test_retries_are_bounded() -> None:
    """Three attempts, then the loop logs it and moves to the next minute.

    An unbounded retry would hold a minute open and, because the context has
    moved on, eventually place an order decided from a stale window.
    """
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("refused", request=request)

    client, slept = client_for(handler, max_attempts=3, backoff=0.01)

    with pytest.raises(ExecutionError, match="attempt 3/3"):
        asyncio.run(client.submit(a_signal()))

    assert len(attempts) == 3
    assert len(slept) == 2, "no sleep after the final attempt"


def test_a_5xx_is_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "INTERNAL", "message": "busy"})
        return accepted()

    client, _ = client_for(handler, backoff=0.0)
    assert asyncio.run(client.submit(a_signal())).order_id == "4412"
    assert len(attempts) == 2


def test_a_429_is_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": "RATE_LIMITED", "message": "slow down"})
        return accepted()

    client, _ = client_for(handler, backoff=0.0)
    assert asyncio.run(client.submit(a_signal())).order_id == "4412"
    assert len(attempts) == 2


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_a_definitive_refusal_is_not_retried(status: int) -> None:
    """A 4xx is a decision. Sending it again cannot change the answer.

    This is the case that matters most for a strategy: a risk refusal that
    gets retried into a timeout is a refusal nobody can see.
    """
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(
            status,
            json={"error": "RISK_MAX_POSITION", "message": "order value too large"},
        )

    client, slept = client_for(handler, backoff=0.0)

    with pytest.raises(ExecutionError) as caught:
        asyncio.run(client.submit(a_signal()))

    assert caught.value.status == status
    assert "RISK_MAX_POSITION" in str(caught.value)
    assert len(attempts) == 1
    assert slept == []


def test_every_attempt_carries_the_same_key() -> None:
    """The retries re-send one decision, which is what makes them safe."""
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        raise httpx.ConnectError("refused", request=request)

    client, _ = client_for(handler, max_attempts=3, backoff=0.0)
    with pytest.raises(ExecutionError):
        asyncio.run(client.submit(a_signal()))

    assert len(bodies) == 3
    assert len(set(bodies)) == 1, "a key that varied per attempt would defeat the retry"


def test_an_accepted_body_without_an_order_id_is_an_error() -> None:
    """A 202 with no order id means the answer is unusable, not that it worked."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"status": "FILLED"})

    client, _ = client_for(handler)
    with pytest.raises(ExecutionError, match="without an order_id"):
        asyncio.run(client.submit(a_signal()))


def test_a_non_json_answer_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    client, _ = client_for(handler)
    with pytest.raises(ExecutionError, match="not JSON"):
        asyncio.run(client.submit(a_signal()))


def test_a_base_url_with_a_trailing_slash_does_not_double_it() -> None:
    """A config value copied with its slash must not become `//v1/signals`."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return accepted()

    client, _ = client_for(handler, base_url="http://execution.invalid:8080/")
    asyncio.run(client.submit(a_signal()))

    assert seen == [f"http://execution.invalid:8080{SIGNALS_PATH}"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"backoff_seconds": -1.0}, "backoff_seconds"),
    ],
)
def test_a_nonsensical_retry_budget_is_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExecutionClient("http://execution.invalid:8080", **kwargs)


def test_the_own_client_is_closed_and_an_injected_one_is_not() -> None:
    """A client this object created is this object's to close, and vice versa."""
    owned = ExecutionClient("http://execution.invalid:8080")
    asyncio.run(owned.aclose())
    asyncio.run(owned.aclose())  # idempotent

    transport = httpx.MockTransport(lambda request: accepted())
    injected = ExecutionClient(
        "http://execution.invalid:8080", client=httpx.AsyncClient(transport=transport)
    )
    asyncio.run(injected.aclose())
    # The injected client is the caller's; this one did not close it.
    assert injected._client.is_closed is False  # noqa: SLF001 - pinned ownership
