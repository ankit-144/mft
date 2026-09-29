"""`contracts.Signal` construction and the idempotency key.

The key is the load-bearing part of this component. Inference retries — a
timeout, a 502, a process restart mid-minute — and C7 answers a replayed key
with `409` and the original order id, placing nothing. So a key that varies
per attempt turns a retry into a second order, and a key that does not vary
per minute turns the strategy into exactly one order per symbol per day.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.signals import (
    KEY_MINUTE_FORMAT,
    Signal,
    idempotency_key,
    rfc3339,
    side_for_score,
)

AS_OF = datetime(2026, 9, 29, 10, 31, tzinfo=timezone.utc)


def test_the_key_is_the_format_the_contract_documents() -> None:
    """`RELIANCE:BUY:20260929T1031`, the literal example in contracts §7."""
    assert idempotency_key("RELIANCE", "BUY", AS_OF) == "RELIANCE:BUY:20260929T1031"
    assert KEY_MINUTE_FORMAT == "%Y%m%dT%H%M"


def test_a_retry_of_the_same_minute_reuses_the_key() -> None:
    """Same symbol, same side, same minute close, any number of attempts.

    This is the property that makes the retry safe. It is checked through the
    full `from_score` path rather than on the key function alone, because a
    caller that builds the key differently is the realistic failure.
    """
    keys = {
        Signal.from_score(
            symbol="RELIANCE",
            score=0.72,
            price=2934.5,
            quantity=10,
            model="heuristic-v1",
            as_of=AS_OF,
        ).idempotency_key
        for _ in range(25)
    }
    assert keys == {"RELIANCE:BUY:20260929T1031"}


def test_the_next_minute_is_a_different_key() -> None:
    """One minute on, the key differs. Otherwise C7's 409 eats the strategy."""
    first = idempotency_key("RELIANCE", "BUY", AS_OF)
    second = idempotency_key("RELIANCE", "BUY", AS_OF + timedelta(minutes=1))
    assert first != second
    assert second == "RELIANCE:BUY:20260929T1032"


def test_a_flip_at_the_same_minute_is_a_different_key() -> None:
    """BUY then SELL at one minute close is two decisions, not one replay.

    Folding the side out of the key would let a genuine reversal be answered
    with a 409 and silently dropped.
    """
    buy = Signal.from_score(
        symbol="RELIANCE", score=0.8, price=100.0, quantity=1, model="m", as_of=AS_OF
    )
    sell = Signal.from_score(
        symbol="RELIANCE", score=-0.8, price=100.0, quantity=1, model="m", as_of=AS_OF
    )
    assert buy.idempotency_key == "RELIANCE:BUY:20260929T1031"
    assert sell.idempotency_key == "RELIANCE:SELL:20260929T1031"
    assert buy.idempotency_key != sell.idempotency_key


def test_two_symbols_at_one_minute_do_not_collide() -> None:
    assert idempotency_key("RELIANCE", "BUY", AS_OF) != idempotency_key(
        "TCS", "BUY", AS_OF
    )


def test_the_key_does_not_move_with_the_host_timezone(monkeypatch) -> None:
    """The minute is rendered in UTC, so `TZ` cannot shift a key.

    Inference and execution are separate processes and may well have different
    `TZ`. A key derived from a local-time rendering would differ between them,
    and a 409 would never fire.
    """
    import time

    monkeypatch.setenv("TZ", "Asia/Kolkata")
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        # The same instant, expressed in two zones, must key identically.
        kolkata = AS_OF.astimezone(timezone(timedelta(hours=5, minutes=30)))
        assert idempotency_key("RELIANCE", "BUY", kolkata) == "RELIANCE:BUY:20260929T1031"
    finally:
        monkeypatch.undo()
        if hasattr(time, "tzset"):
            time.tzset()


def test_case_and_whitespace_do_not_change_the_key() -> None:
    """C7 upper-cases what it receives, so two spellings are one decision."""
    assert idempotency_key(" reliance ", " buy ", AS_OF) == "RELIANCE:BUY:20260929T1031"


def test_the_side_comes_from_the_sign_of_the_score() -> None:
    assert side_for_score(0.01) == "BUY"
    assert side_for_score(-0.01) == "SELL"
    assert side_for_score(1.0) == "BUY"
    assert side_for_score(-1.0) == "SELL"


@pytest.mark.parametrize("score", [0.0, -0.0])
def test_a_zero_score_has_no_side(score: float) -> None:
    """Zero is a real model output, and it is not a buy.

    C5's `scale_to_unit` maps a diverged model to 0.0 on purpose. Coercing
    that into a side would trade the neutral answer, which is the one answer
    that means "no opinion".
    """
    with pytest.raises(ValueError, match="no direction"):
        side_for_score(score)


@pytest.mark.parametrize("score", [1.0001, -1.0001, float("nan"), float("inf")])
def test_a_score_outside_the_contract_is_refused(score: float) -> None:
    with pytest.raises(ValueError):
        side_for_score(score)


def test_the_wire_body_is_the_contract_struct() -> None:
    """Exactly the nine fields of `contracts.Signal`, with the JSON names §1 uses.

    C7 decodes this body with `encoding/json` into the Go struct and rejects an
    empty `idempotency_key` with a 400, so an extra or renamed field is a
    400 at the far end of a trading loop.
    """
    signal = Signal.from_score(
        symbol="RELIANCE",
        score=0.72,
        price=2934.5,
        quantity=10,
        model="heuristic-v1",
        as_of=AS_OF,
    )
    body = signal.to_wire()

    assert set(body) == {
        "symbol",
        "side",
        "quantity",
        "price",
        "score",
        "model",
        "as_of",
        "idempotency_key",
    }
    assert body["side"] == "BUY"
    assert body["quantity"] == 10
    assert body["price"] == 2934.5
    assert body["score"] == 0.72
    assert body["model"] == "heuristic-v1"
    # RFC3339 with a Z, which is how Go's time.Time marshals: the contract's
    # own example is "2026-09-29T10:31:00Z".
    assert body["as_of"] == "2026-09-29T10:31:00Z"
    assert body["idempotency_key"] == "RELIANCE:BUY:20260929T1031"


def test_as_of_is_normalised_to_utc_on_the_wire() -> None:
    """An offset-aware instant is rendered in UTC, whatever zone it arrived in."""
    kolkata = AS_OF.astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert Signal.from_score(
        symbol="RELIANCE",
        score=0.5,
        price=1.0,
        quantity=1,
        model="m",
        as_of=kolkata,
    ).to_wire()["as_of"] == "2026-09-29T10:31:00Z"


def test_a_naive_as_of_is_refused() -> None:
    """A timestamp with no zone has no single meaning on the wire."""
    with pytest.raises(ValueError, match="timezone"):
        Signal(
            symbol="RELIANCE",
            side="BUY",
            quantity=1,
            price=1.0,
            score=0.5,
            model="m",
            as_of=datetime(2026, 9, 29, 10, 31),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("symbol", "  ", "symbol is required"),
        ("price", 0.0, "must be positive"),
        ("price", -1.0, "must be positive"),
        ("quantity", 0, "must be positive"),
        ("score", 1.5, r"outside \[-1, 1\]"),
    ],
)
def test_a_signal_that_execution_would_refuse_is_never_built(field, value, match) -> None:
    """C7 rejects these with a 400; the loop must not build one at all.

    The risk gate sizes a position as quantity*price, so a zero price is not a
    small order, it is an unmeasurable one.
    """
    kwargs = {
        "symbol": "RELIANCE",
        "side": "BUY",
        "quantity": 10,
        "price": 100.0,
        "score": 0.5,
        "model": "m",
        "as_of": AS_OF,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        Signal(**kwargs)


def test_an_unknown_field_is_refused() -> None:
    """A typo in a signal field is a bug, not something to send and let fail."""
    with pytest.raises(ValueError):
        Signal(
            symbol="RELIANCE",
            side="BUY",
            quantity=1,
            price=1.0,
            score=0.5,
            model="m",
            as_of=AS_OF,
            side_typo="BUY",
        )


def test_rfc3339_renders_utc_with_a_z() -> None:
    assert rfc3339(AS_OF) == "2026-09-29T10:31:00Z"
    assert (
        rfc3339(datetime(2026, 9, 29, 16, 1, tzinfo=timezone(timedelta(hours=5, minutes=30))))
        == "2026-09-29T10:31:00Z"
    )
