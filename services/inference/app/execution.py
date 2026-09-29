"""Submitting signals to the execution service (C7).

The contract is `docs/contracts.md` §7: `POST /v1/signals` with a
`contracts.Signal` body, `202` filled, `200` working, `400` rejected by risk,
`409` a replayed idempotency key carrying the original order id.

# What is retried, and what is not

Retrying is only correct for failures that leave the outcome unknown. A `400`
is a decision — the risk gate said no, and saying no again changes nothing. A
`409` is a success: the order was placed by an earlier attempt and C7 has
returned its id, so the retry has done its job. Retrying either would burn
attempts and, worse, obscure a refusal behind an eventual timeout.

So the retry set is exactly: a transport error (connect, read, write, pool), a
`429`, and a `5xx`. Everything else is returned or raised immediately.

# Why a 409 is not an error

Inference cannot tell a lost response from a lost request. If the POST lands
and the connection dies before the body arrives, a naive client retries and the
retry is answered with 409 and the original `order_id`. Treating that as a
failure would log a phantom problem on every slow minute; treating it as a
first-time fill would report an order id that was placed minutes ago. It is
reported as a distinct `duplicate` acknowledgement so the caller can log it
accurately.

# Bounded, and always with the same key

Every attempt carries the signal this client was given, byte for byte. The
retries therefore re-send one key, which is what makes them safe. Backoff is
exponential with full jitter, capped, and the attempt count is fixed. There is
no unbounded path out of this module.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

import httpx

from .signals import Signal

logger = logging.getLogger("mft.inference.execution")

#: Path of the signal endpoint. `docs/contracts.md` §7.
SIGNALS_PATH = "/v1/signals"

#: Total attempts including the first. Three is enough to ride out one
#: connection reset without holding the minute open for a long outage; the
#: next minute close is a fresh decision anyway.
DEFAULT_MAX_ATTEMPTS = 3

#: First backoff, doubled per attempt and capped at `_MAX_BACKOFF`.
DEFAULT_BACKOFF_SECONDS = 0.25

#: Backoff ceiling. A minute cadence does not reward a patient retry.
_MAX_BACKOFF_SECONDS = 4.0

#: Per-request timeout. The execution service answers in milliseconds; a
#: request that has not answered in this long is not going to.
DEFAULT_TIMEOUT_SECONDS = 5.0


class ExecutionError(RuntimeError):
    """Raised when a signal could not be delivered or was rejected.

    Carries the HTTP status when there was one, so the caller can log
    something more useful than "failed".
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class SignalAck:
    """The execution service's answer to a signal."""

    order_id: str
    status: str
    score: float
    duplicate: bool = False
    http_status: int = 0

    def __str__(self) -> str:
        state = "duplicate" if self.duplicate else "accepted"
        return f"{state} order_id={self.order_id} status={self.status}"


class ExecutionClient:
    """Posts signals to `base_url` with a timeout, bounded retries and backoff.

    Args:
        base_url: The execution service root, without a trailing slash.
        timeout: Per-request timeout in seconds.
        max_attempts: Total attempts per submit, including the first.
        backoff_seconds: First backoff; doubled per attempt, jittered.
        client: An existing `httpx.AsyncClient`. One is created and owned here
            when omitted.
        sleep: Injected for tests. Defaults to `asyncio.sleep`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        client: httpx.AsyncClient | None = None,
        sleep: Any = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if backoff_seconds < 0.0:
            raise ValueError(f"backoff_seconds must be >= 0, got {backoff_seconds}")
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds
        self._client = client if client is not None else httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), headers={"content-type": "application/json"}
        )
        self._owns_client = client is None
        self._sleep = sleep if sleep is not None else asyncio.sleep

    @property
    def base_url(self) -> str:
        """The execution service root this client posts to."""
        return self._base_url

    async def aclose(self) -> None:
        """Close the underlying client when this object created it."""
        if self._owns_client:
            await self._client.aclose()

    async def submit(self, signal: Signal) -> SignalAck:
        """Deliver `signal`, retrying only failures with an unknown outcome.

        Raises:
            ExecutionError: If every attempt failed, or the service gave a
                definitive answer that was not an acceptance.
        """
        body = signal.to_wire()
        url = f"{self._base_url}{SIGNALS_PATH}"
        last: ExecutionError | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(url, json=body)
            except httpx.HTTPError as err:
                last = ExecutionError(
                    f"{signal.symbol} {signal.idempotency_key}: attempt "
                    f"{attempt}/{self._max_attempts} to {url} failed: {err}"
                )
                logger.warning("%s", last)
                if attempt < self._max_attempts:
                    await self._backoff_for(attempt)
                continue

            if response.status_code in (200, 202):
                return _ack_from(response, duplicate=False)
            if response.status_code == 409:
                # A replayed key. The order exists; the retry has succeeded.
                ack = _ack_from(response, duplicate=True)
                logger.info(
                    "%s %s was already processed as order %s",
                    signal.symbol,
                    signal.idempotency_key,
                    ack.order_id,
                )
                return ack

            detail = _error_detail(response)
            if response.status_code < 500 and response.status_code != 429:
                # A decision, not a fault. Retrying would only repeat it.
                raise ExecutionError(
                    f"{signal.symbol} {signal.idempotency_key}: execution "
                    f"refused the signal with {response.status_code}: {detail}",
                    status=response.status_code,
                )

            last = ExecutionError(
                f"{signal.symbol} {signal.idempotency_key}: attempt "
                f"{attempt}/{self._max_attempts} to {url} returned "
                f"{response.status_code}: {detail}",
                status=response.status_code,
            )
            logger.warning("%s", last)
            if attempt < self._max_attempts:
                await self._backoff_for(attempt)

        raise last or ExecutionError("signal was never attempted")

    async def _backoff_for(self, attempt: int) -> None:
        """Exponential backoff with full jitter, capped."""
        ceiling = min(self._backoff * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
        if ceiling <= 0.0:
            return
        await self._sleep(random.uniform(0.0, ceiling))


def _ack_from(response: httpx.Response, *, duplicate: bool) -> SignalAck:
    """Parse an accepted or replayed answer. Raises on an unusable body."""
    try:
        document = response.json()
    except ValueError as err:
        raise ExecutionError(
            f"execution returned {response.status_code} with a body that is not JSON",
            status=response.status_code,
        ) from err
    if not isinstance(document, dict):
        raise ExecutionError(
            f"execution returned {response.status_code} with a non-object body",
            status=response.status_code,
        )

    order_id = str(document.get("order_id") or "")
    if not order_id:
        raise ExecutionError(
            f"execution returned {response.status_code} without an order_id",
            status=response.status_code,
        )

    score = document.get("score", 0.0)
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    return SignalAck(
        order_id=order_id,
        status=str(document.get("status") or ""),
        score=score_value,
        duplicate=duplicate,
        http_status=response.status_code,
    )


def _error_detail(response: httpx.Response) -> str:
    """The `message` from a contract error body, or the raw text."""
    try:
        document = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(document, dict):
        code = str(document.get("error") or "")
        message = str(document.get("message") or "")
        if code and message:
            return f"{code}: {message}"
        if message:
            return message
    return response.text[:200]


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "SIGNALS_PATH",
    "ExecutionClient",
    "ExecutionError",
    "SignalAck",
]
