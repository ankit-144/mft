"""Submitting signals to the execution service (C7)."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

import httpx

from .signals import Signal

logger = logging.getLogger("mft.inference.execution")


SIGNALS_PATH = "/v1/signals"


DEFAULT_MAX_ATTEMPTS = 3


DEFAULT_BACKOFF_SECONDS = 0.25


_MAX_BACKOFF_SECONDS = 4.0


DEFAULT_TIMEOUT_SECONDS = 5.0


class ExecutionError(RuntimeError):
    """Raised when a signal could not be delivered or was rejected."""

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
    """Posts signals to `base_url` with a timeout, bounded retries and backoff."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        client: httpx.AsyncClient | None = None,
        sleep: Any = None,
        api_token: str = "",
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
        self._headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

    @property
    def base_url(self) -> str:
        """The execution service root this client posts to."""
        return self._base_url

    async def aclose(self) -> None:
        """Close the underlying client when this object created it."""
        if self._owns_client:
            await self._client.aclose()

    async def submit(self, signal: Signal) -> SignalAck:
        """Deliver `signal`, retrying only failures with an unknown outcome."""
        body = signal.to_wire()
        url = f"{self._base_url}{SIGNALS_PATH}"
        last: ExecutionError | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(url, json=body, headers=self._headers)
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
    """Parse an accepted or replayed answer."""
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
