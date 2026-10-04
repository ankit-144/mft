"""The backtest harness: lookahead, the risk gate, and whether the arithmetic holds.

# The test that matters is `test_no_lookahead_*`

A backtest that leaks the future is worse than no backtest, because it
manufactures a result that looks like an edge and is not. The property is
structural rather than statistical, so it is tested structurally: replace the
final `horizon` bars with a violent move and assert that nothing upstream of
the label changes — not one score, not one candidate key, not one price.

Those final `horizon` bars are exactly the bars that are labels and never
contexts, which is what makes the mutation a clean experiment. The loop runs `i`
over `[first_decision, len - horizon)`, so the newest bar any prompt can contain
is `candles[len - horizon - 1]`. Bars from `len - horizon` onward are read only
by `candles[i + horizon]`, i.e. only to compute realised P&L for a decision that
has already been made. A 6x jump with a million-share volume on those bars
would be unmissable in a score if it leaked.

On the wording "identical to the same series with those bars removed": *removed*
is a different experiment from *replaced*, and both are asserted. Truncating
`horizon` bars shortens the series, which removes `horizon` decision points
from the end of the run, so a truncated run is legitimately a different run
with fewer trades and the two cannot be equal. Replacing them holds the decision
set fixed and varies only the labels, which is what isolates the leak.

# What is *not* tested here, and why

No test loads the TabFM checkpoint, opens a socket, or writes outside
`tmp_path`. The model is a constant-score or scripted stub: this suite is about
the harness's slicing and its accounting, and a 6.6 GB checkpoint would test
neither. A stub also takes the model out of the equation, so a metrics
assertion is asserting the harness and not C5's blend.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd
import pytest

from app.backtest import (
    BARS_PER_YEAR,
    Candidate,
    MIN_USABLE_BARS,
    BarOutcome,
    BacktestError,
    BacktestResult,
    ChargeModel,
    EquityPoint,
    ExecutionLimits,
    Portfolio,
    Reason,
    WalkForward,
    context_from_frame,
    projected_rebuild_minutes,
    run_backtest,
    sharpe_of,
)
from app.candles import Candle
from app.config import Config, InferenceConfig
from app.features import WARMUP_ROWS, FeatureBuilder, trading_zone
from app.predictor import Predictor
from app.synth import (
    DEFAULT_ANNUAL_VOLATILITY,
    SESSION_BARS_PER_DAY,
    SynthSpec,
    generate,
    session_grid,
)
from support import ConstantModel

#: A Monday. `SynthSpec`'s default start, named here so the in-session
#: assertions below do not have to discover it.
START_DAY = date(2026, 1, 5)

#: The first bar the walk-forward can decide on, for `horizon_bars: 1`: the
#: warm-up plus the model's own row floor plus the label bar.
FIRST_DECISION = WARMUP_ROWS + 60 + 1 - 1

#: Permissive risk limits: let everything through, so a test that is about the
#: model or the threshold is not also a test of `max_position_pct`.
PERMISSIVE: dict[str, Any] = dict(
    capital=1_000_000.0,
    debounce_ttl_seconds=0,
    idempotency_ttl_seconds=0,
    max_position_pct=100.0,
    max_open_positions=10,
    max_drawdown_pct=100.0,
    daily_loss_limit=1e12,
    max_order_quantity=10_000,
)


def permissive(**overrides: Any) -> ExecutionLimits:
    """`ExecutionLimits` that refuse nothing, with the named fields replaced."""
    return ExecutionLimits(**{**PERMISSIVE, **overrides})


def inference(**overrides: Any) -> InferenceConfig:
    """An `inference` section that trades every bar the model likes."""
    fields: dict[str, Any] = dict(
        model="heuristic",
        execution_url="http://execution.invalid:8080",
        context_rows=100,
        horizon_bars=1,
        score_threshold=0.5,
        order_quantity=10,
        instruments=["TEST"],
        dry_run=True,
    )
    fields.update(overrides)
    return InferenceConfig(**fields)


class ScoreScript:
    """An `InferenceModel` that returns a scripted score per call.

    `Predictor` calls `predict` once per decision bar, in order, on one thread,
    so a counter is enough to script a per-bar signal. It exists because a
    constant score cannot express "BUY, then SELL" and the debounce test needs
    that to tell a per-side key apart from a per-symbol one.
    """

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = list(scores)
        self._index = 0

    @property
    def name(self) -> str:
        return "script-v1"

    def is_loaded(self) -> bool:
        return True

    async def load(self) -> None:
        return None

    def predict(self, context: Any, horizon: int) -> float:
        value = self._scores[self._index % len(self._scores)]
        self._index += 1
        return value


def predictor_for(model: Any) -> Predictor:
    """Wrap any stub in a loaded `Predictor`, which is what the walk-forward wants.

    `Predictor.is_loaded()` is its own flag *and* the model's, so a stub has to
    go through `load()` even though it has nothing to load. The thread pool it
    owns is not bound to the event loop that created it, so a predictor loaded
    under one `asyncio.run` is usable under the next.
    """
    predictor = Predictor(model, model_name=getattr(model, "name", "stub"))
    asyncio.run(predictor.load())
    return predictor


def constant_predictor(score: float) -> Predictor:
    """A `Predictor` that always returns `score`."""
    return predictor_for(ConstantModel(score))


async def _run_once(
    candles: dict[str, list[Candle]],
    predictor: Predictor,
    limits: ExecutionLimits,
    section: InferenceConfig,
    mode: str,
    charges: ChargeModel | None,
    max_bars: int,
    synthetic: bool,
) -> BacktestResult:
    try:
        return await run_backtest(
            candles,
            config=Config(inference=section),
            limits=limits,
            predictor=predictor,
            mode=mode,  # type: ignore[arg-type]
            charges=charges,
            synthetic=synthetic,
            seed=7 if synthetic else None,
            data_source="synthetic" if synthetic else "parquet",
            max_bars=max_bars,
        )
    finally:
        await predictor.aclose()


def backtest(
    candles: dict[str, list[Candle]],
    *,
    score: float = 0.9,
    scores: Sequence[float] | None = None,
    limits: ExecutionLimits | None = None,
    section: InferenceConfig | None = None,
    mode: str = "rebuild",
    charges: ChargeModel | None = ChargeModel.zero(),
    max_bars: int = 5000,
    synthetic: bool = True,
) -> BacktestResult:
    """Run the harness over a small in-memory series."""
    model = ScoreScript(scores) if scores is not None else ConstantModel(score)
    return asyncio.run(
        _run_once(
            candles,
            predictor_for(model),
            limits or permissive(),
            section or inference(),
            mode,
            charges,
            max_bars,
            synthetic,
        )
    )


def walk_forward(section: InferenceConfig | None = None, mode: str = "rebuild") -> WalkForward:
    """The phase-1 walk-forward, wired to a constant-score predictor."""
    return WalkForward(
        builder=FeatureBuilder("Asia/Kolkata"),
        predictor=constant_predictor(0.9),
        inference=section or inference(),
        mode=mode,  # type: ignore[arg-type]
    )


def linear_candles(
    closes: Sequence[float], *, symbol: str = "TEST", offset: timedelta = timedelta(0)
) -> list[Candle]:
    """Well-formed candles from a close series, on the NSE session grid.

    `open[i]` is the previous close, so a bar's body is exactly the price move
    its label measures, and `high`/`low` are padded by 50 paise so every bar
    passes `FeatureBuilder._validate` without a rounding surprise. `offset`
    shifts the whole grid, which is how two instruments are made to stop
    colliding on the same minute.
    """
    stamps = [
        stamp + offset
        for stamp in session_grid(len(closes), start=START_DAY, zone=trading_zone("Asia/Kolkata"))
    ]
    out: list[Candle] = []
    for index, (stamp, close) in enumerate(zip(stamps, closes)):
        open_ = closes[index - 1] if index else close
        out.append(
            Candle(
                symbol=symbol,
                timestamp=stamp,
                open=round(open_, 2),
                high=round(max(open_, close) + 0.5, 2),
                low=round(min(open_, close) - 0.5, 2),
                close=round(close, 2),
                volume=10_000 + index,
            )
        )
    return out


def sawtooth_candles(n: int, *, period: int = 5, symbol: str = "TEST") -> list[Candle]:
    """A price series whose bar-to-bar change cycles through five values.

    The close moves `+1, -2, +3, -4, +5` and repeats, so the change at index `i`
    is exactly `(1,-2,3,-4,5)[i % period]` and a 10-share round trip earns
    exactly `10x` that. Every P&L assertion below is arithmetic a reader can do
    on paper.
    """
    closes: list[float] = []
    price = 1000.0
    for index in range(n):
        closes.append(round(price, 2))
        price += (1, -2, 3, -4, 5)[index % period]
    return linear_candles(closes, symbol=symbol)


# --------------------------------------------------------------------------
# The lookahead property
# --------------------------------------------------------------------------


def corrupt_tail(candles: Sequence[Candle], bars: int, *, factor: float = 6.0) -> list[Candle]:
    """Replace the final `bars` candles with a violent, well-formed move.

    Well-formed on purpose. A malformed bar would make `FeatureBuilder` reject
    the whole series, which in `incremental` mode would turn a leak test into a
    build-failure test and prove nothing. The replacement is a 6x jump with a
    wide range and a million-share volume, so anything derived from it would be
    unmissable in a score.
    """
    out = list(candles)
    price = out[-bars - 1].close
    for index in range(len(out) - bars, len(out)):
        price *= factor
        out[index] = Candle(
            symbol=out[index].symbol,
            timestamp=out[index].timestamp,
            open=round(price / factor, 2),
            high=round(price * 1.5, 2),
            low=round(price / factor * 0.5, 2),
            close=round(price, 2),
            volume=10_000_000,
        )
    return out


@pytest.mark.parametrize("mode", ["rebuild", "incremental"])
def test_no_lookahead_scores_and_candidates_are_identical(mode: str) -> None:
    """A violent move in the label bar must not change one score.

    This is the assertion the module exists to support. The mutated series and
    the clean one must produce the same score for every decision bar, and
    therefore the same candidate set, the same idempotency keys and the same
    entry prices. If any of those differ, a label bar reached the model.
    """
    clean = generate("TEST", SynthSpec(bars=200, seed=11, start_price=1000.0))
    dirty = corrupt_tail(clean, 1)

    a = backtest({"TEST": clean}, mode=mode)
    b = backtest({"TEST": dirty}, mode=mode)

    # Same decision points, so the trade lists line up one for one.
    assert len(a.trades) == len(b.trades)
    assert [t.entry_as_of for t in a.trades] == [t.entry_as_of for t in b.trades]
    assert [t.entry_price for t in a.trades] == [t.entry_price for t in b.trades]
    assert [t.key for t in a.trades] == [t.key for t in b.trades]
    assert [t.score for t in a.trades] == [t.score for t in b.trades]

    # The single label bar is the *only* thing allowed to differ, and only for
    # the one trade that reads it.
    differing = [
        i for i, (x, y) in enumerate(zip(a.trades, b.trades)) if x.exit_price != y.exit_price
    ]
    assert differing == [len(a.trades) - 1], f"only the last trade may differ, got {differing}"


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_no_lookahead_is_exact_for_every_horizon(horizon: int) -> None:
    """The property holds for any horizon, not only the configured 1.

    With `horizon` bars there are `horizon` bars that are labels and never
    contexts, so the mutated run must differ in exactly those `horizon` exits —
    no more and no less.
    """
    clean = generate("TEST", SynthSpec(bars=220, seed=13, start_price=1000.0))
    dirty = corrupt_tail(clean, horizon)
    section = inference(horizon_bars=horizon)

    a = backtest({"TEST": clean}, section=section)
    b = backtest({"TEST": dirty}, section=section)

    assert len(a.trades) == len(b.trades)
    assert [t.entry_as_of for t in a.trades] == [t.entry_as_of for t in b.trades]
    assert [t.score for t in a.trades] == [t.score for t in b.trades]
    assert [t.key for t in a.trades] == [t.key for t in b.trades]
    assert [t.entry_price for t in a.trades] == [t.entry_price for t in b.trades]

    differing = [
        i for i, (x, y) in enumerate(zip(a.trades, b.trades)) if x.exit_price != y.exit_price
    ]
    assert differing == list(range(len(a.trades) - horizon, len(a.trades))), (
        f"only the last {horizon} labels may differ, but {differing} did"
    )


def test_truncating_the_label_bars_shortens_the_run_and_nothing_else() -> None:
    """The "same series with those bars removed" half of the property.

    Truncation is a different experiment from mutation and it is worth being
    explicit about why: removing the final `horizon` bars removes `horizon`
    decision points, because the loop stops at `len - horizon`. So the two runs
    are not equal, and a test asserting they were would be wrong. What must hold
    is that every decision they share is the same decision, i.e. the truncated
    run's trades are a strict prefix of the full run's, P&L included.
    """
    horizon = 3
    clean = generate("TEST", SynthSpec(bars=220, seed=17, start_price=1000.0))
    section = inference(horizon_bars=horizon)

    full = backtest({"TEST": clean}, section=section)
    short = backtest({"TEST": list(clean[: len(clean) - horizon])}, section=section)

    assert len(short.trades) == len(full.trades) - horizon
    for a, b in zip(short.trades, full.trades):
        assert (a.key, a.score, a.entry_price, a.exit_price, a.net_pnl) == (
            b.key,
            b.score,
            b.entry_price,
            b.exit_price,
            b.net_pnl,
        )


def test_the_prompt_never_receives_a_row_from_after_the_decision_bar() -> None:
    """The mechanism, asserted directly rather than through the outcome.

    The score-equality tests prove the leak is absent end to end. This one
    proves why: at decision bar `i` the prompt is exactly the feature rows for
    candles `0..i`, truncated to `context_rows` with the warm-up dropped, and its
    newest row is candle `i` and not a later one.
    """
    walk = walk_forward(mode="incremental")
    candles = generate("TEST", SynthSpec(bars=200, seed=19, start_price=1000.0))
    frame = walk.builder.build(candles).frame

    for index in (walk.first_decision, walk.first_decision + 1, walk.first_decision + 40,
                  len(candles) - 2):
        context = context_from_frame(frame, index, walk.context_rows)
        available = index + 1 - WARMUP_ROWS
        assert len(context) == min(walk.context_rows, available)
        # Nothing past the decision bar, and the newest row is its own row and
        # not a near neighbour of it.
        assert len(context) <= available
        assert context.iloc[-1].equals(frame.iloc[index])
        assert context.iloc[0].equals(frame.iloc[index - len(context) + 1])


def test_the_walk_starts_where_the_model_can_actually_use_a_prompt() -> None:
    """`WARMUP_ROWS` alone is not enough, and starting there would be a bug.

    `FeatureTable.context` drops the warm-up and `model.validate_context` then
    wants `MIN_CONTEXT_ROWS + horizon` survivors. Starting at bar 60 hands the
    model one row, which answers with a confident number computed from almost
    nothing.
    """
    walk = walk_forward()
    assert walk.first_decision == FIRST_DECISION
    assert walk.first_decision == MIN_USABLE_BARS - 1
    # At the first decision bar there are exactly 61 survivors: the model's 60
    # rows plus the label bar it must not be shown.
    assert walk.first_decision + 1 - WARMUP_ROWS == 61

    # A series that ends at the first decision bar decides on nothing, and says
    # so rather than crashing.
    short = asyncio.run(walk.run("TEST", generate("TEST", SynthSpec(bars=FIRST_DECISION, seed=3))))
    assert short.scored == 0
    assert short.candidates == []


# --------------------------------------------------------------------------
# The two walk-forward modes must be the same computation
# --------------------------------------------------------------------------


def test_incremental_and_rebuild_produce_identical_prompts() -> None:
    """The O(n) shortcut is the same arithmetic, value for value.

    `incremental` slices rows out of one table rather than rebuilding a table
    per bar. That is only legitimate because every feature at row `i` is a pure
    function of candles `0..i` with the same summation order. This asserts it
    exactly (`check_exact=True`), not approximately, because a feature whose
    value drifts with the length of its history is exactly the drift this
    architecture cannot tolerate — and C3 pins the newest row of a 100-bar build
    to the newest row of a 100000-bar build for the same reason.
    """
    walk = walk_forward()
    candles = generate("TEST", SynthSpec(bars=200, seed=23, start_price=1000.0))
    full = walk.builder.build(candles).frame
    checked = 0
    for index in range(walk.first_decision, len(candles) - walk.horizon, 7):
        reference = walk.builder.build(candles[: index + 1]).context(walk.context_rows)
        pd.testing.assert_frame_equal(
            context_from_frame(full, index, walk.context_rows), reference, check_exact=True
        )
        checked += 1
    assert checked > 5


def test_both_modes_agree_on_the_whole_run() -> None:
    """And therefore on the trades, the P&L, the metrics and the rejections."""
    section = inference(horizon_bars=2)
    candles = generate("TEST", SynthSpec(bars=200, seed=29, start_price=1000.0))
    a = backtest({"TEST": candles}, mode="rebuild", section=section)
    b = backtest({"TEST": candles}, mode="incremental", section=section)

    assert [t.to_json() for t in a.trades] == [t.to_json() for t in b.trades]
    assert a.metrics.to_json() == b.metrics.to_json()
    assert a.rejections == b.rejections
    assert a.outcomes == b.outcomes
    # And the shortcut earns its name.
    assert b.timing.feature_seconds < a.timing.feature_seconds / 4.0


# --------------------------------------------------------------------------
# Risk gate
# --------------------------------------------------------------------------


def test_debounce_ttl_suppresses_a_second_trade_in_the_same_window() -> None:
    """A constant BUY every minute, with a 10-minute debounce.

    The debounce is per `SYMBOL:SIDE` with `execution.debounce_ttl_seconds`, as
    `docs/contracts.md` §6 check 5 words it: a BUY at 13:00 suppresses BUYs for
    ten minutes and nothing else.
    """
    candles = sawtooth_candles(200)
    result = backtest({"TEST": candles}, limits=permissive(debounce_ttl_seconds=600))

    first = candles[FIRST_DECISION].timestamp
    entries = [trade.entry_as_of for trade in result.trades]
    # A fill every 600 s and not a minute sooner: the window reopens exactly at
    # the TTL, not a bar later and not a bar earlier.
    assert entries == [first + timedelta(seconds=600 * k) for k in range(8)]
    # Every other candidate in the run was a candidate the gate threw away.
    assert result.metrics.trades + sum(result.rejections.values()) == 79
    assert result.rejections == {Reason.DEBOUNCED.value: 71}


def test_debounce_does_not_suppress_the_opposite_side() -> None:
    """The key is `SYMBOL:SIDE`, so a reversal is a different decision.

    `docs/contracts.md` §1 puts the side in the idempotency key for the same
    reason: a model that flips from BUY to SELL at the same minute close has
    made a different decision, and suppressing it would suppress a genuine
    reversal. An alternating score is the only way to tell a per-side key apart
    from a per-symbol one, which is why this test scripts the model.
    """
    result = backtest(
        {"TEST": sawtooth_candles(200)},
        scores=[0.9, -0.9],
        limits=permissive(debounce_ttl_seconds=600),
    )
    # The first BUY fills and the very next bar's SELL fills too: a BUY's
    # debounce window does not reach across the side. After that each side
    # debounces itself on its own 10-minute cadence, offset by one bar — which
    # is the point. A per-symbol key would have suppressed the SELL as well and
    # the run would have been half the size.
    first = sawtooth_candles(200)[FIRST_DECISION].timestamp
    assert [(t.side, t.entry_as_of) for t in result.trades[:4]] == [
        ("BUY", first),
        ("SELL", first + timedelta(minutes=1)),
        ("BUY", first + timedelta(minutes=10)),
        ("SELL", first + timedelta(minutes=11)),
    ]
    assert {t.side for t in result.trades} == {"BUY", "SELL"}
    assert result.metrics.trades == 16  # 8 per side
    assert result.rejections[Reason.DEBOUNCED.value] == 79 - 16


def test_max_drawdown_halts_trading() -> None:
    """A drawdown past `execution.max_drawdown_pct` refuses every later signal.

    The series falls Rs 5 a bar and the order is 10 shares, so every fill loses
    Rs 50 and the peak equity never rises. At 0.2% of Rs 1,000,000 the gate is
    breached on the 42nd fill — Rs 2050, or 0.205% — and because equity never
    recovers, every candidate after it is refused with the contract's own code.
    """
    n = 180
    limits = permissive(max_drawdown_pct=0.2)
    candles = linear_candles([round(2000.0 - 5.0 * i, 2) for i in range(n)])
    result = backtest({"TEST": candles}, limits=limits)

    decision_bars = n - 1 - FIRST_DECISION
    assert result.metrics.trades + sum(result.rejections.values()) == decision_bars
    assert result.metrics.trades == 41
    assert result.rejections == {Reason.MAX_DRAWDOWN.value: 18}
    assert result.drawdown_halted is True
    assert result.metrics.max_drawdown_pct == pytest.approx(41 * 50.0 / 1_000_000.0 * 100.0)

    # Nothing is filled once the gate is breached.
    first_refusal = candles[FIRST_DECISION + result.metrics.trades].timestamp
    assert result.trades[-1].entry_as_of < first_refusal
    assert all(trade.entry_as_of < first_refusal for trade in result.trades)


def test_max_position_pct_refuses_an_oversized_order() -> None:
    """Check 1: `|qty * price| / equity <= max_position_pct`.

    0.1% of Rs 1,000,000 is a Rs 1,000 cap and a 10-share order on a ~Rs 1,000
    stock is Rs 10,000, so check 1 refuses every candidate and nothing later in
    the ordered list is ever consulted.
    """
    result = backtest({"TEST": sawtooth_candles(200)}, limits=permissive(max_position_pct=0.1))
    assert result.metrics.trades == 0
    assert result.rejections == {Reason.MAX_POSITION.value: 79}


def test_max_open_positions_counts_distinct_symbols() -> None:
    """`len(OpenPositions)` is a count of symbols, because the contract's is a map.

    With a one-symbol limit and two symbols on the same grid, the alphabetically
    first candidate at each timestamp wins the slot and the other is refused;
    a second leg on the already-open symbol would not consume a slot at all, so
    the same symbol keeps trading every bar. Raising the limit to two lets both
    through, which is what shows the refusal was the limit and nothing about the
    second symbol's data.
    """
    candles = {
        "AAA": sawtooth_candles(200, symbol="AAA"),
        "BBB": sawtooth_candles(200, symbol="BBB"),
    }
    one = backtest(candles, limits=permissive(max_open_positions=1, debounce_ttl_seconds=0))
    assert one.metrics.trades == 79
    assert {t.symbol for t in one.trades} == {"AAA"}
    assert one.rejections == {Reason.MAX_POSITIONS.value: 79}

    two = backtest(candles, limits=permissive(max_open_positions=2, debounce_ttl_seconds=0))
    assert two.metrics.trades == 158
    assert {t.symbol for t in two.trades} == {"AAA", "BBB"}
    assert two.rejections == {}

    # And the second symbol on its own trades fine, so nothing about BBB's
    # candles was the problem.
    alone = backtest({"BBB": candles["BBB"]}, limits=permissive(max_open_positions=1))
    assert alone.metrics.trades == 79


def test_bad_quantity_is_refused() -> None:
    """Check 6: `1 <= qty <= max_order_quantity`."""
    result = backtest(
        {"TEST": sawtooth_candles(200)},
        limits=permissive(max_order_quantity=5),
        section=inference(order_quantity=10),
    )
    assert result.metrics.trades == 0
    assert result.rejections == {Reason.BAD_QUANTITY.value: 79}


def test_daily_loss_limit_refuses_the_rest_of_the_day_then_resets() -> None:
    """Check 4: realised P&L is per IST day, and the day roll clears it.

    The series falls Rs 1 a bar and the order is 10 shares, so every fill loses
    Rs 10 and the limit of Rs 100 is breached by the 12th fill of a session.
    1600 bars is four and a bit NSE sessions, so the run trades eleven times per
    day and is refused for the rest of each — which is only true if the counter
    resets at the day boundary.
    """
    n = 1600
    limits = permissive(daily_loss_limit=100.0)
    candles = linear_candles([round(2000.0 - 1.0 * i, 2) for i in range(n)])
    result = backtest({"TEST": candles}, limits=limits, mode="incremental")

    days = {t.entry_as_of.astimezone(trading_zone("Asia/Kolkata")).date() for t in result.trades}
    assert len(days) == 5, "1600 bars is five partial NSE sessions"
    assert result.metrics.trades == 55, "11 fills per session, 5 sessions"
    assert result.rejections == {Reason.DAILY_LOSS.value: (n - 1 - FIRST_DECISION) - 55}
    for day in days:
        assert sum(1 for t in result.trades
                   if t.entry_as_of.astimezone(trading_zone("Asia/Kolkata")).date() == day) == 11


def test_market_hours_accepts_the_session_and_refuses_everything_else() -> None:
    """Check 7: 09:15-15:30 IST, weekdays, minus `execution.market_holidays`."""
    limits = permissive(market_holidays=["2026-01-06"])
    gate = Portfolio(limits, ChargeModel.zero(), zone=trading_zone("Asia/Kolkata"))
    ist = trading_zone("Asia/Kolkata")

    def at(day: int, hour: int, minute: int) -> bool:
        return gate._is_market_open(  # noqa: SLF001 - a test of the gate itself
            datetime(2026, 1, day, hour, minute, tzinfo=ist)
        )

    assert at(5, 9, 15) is True  # Monday, the open
    assert at(5, 15, 29) is True  # the last bar
    assert at(5, 15, 30) is False  # the close
    assert at(5, 9, 14) is False  # before the open
    assert at(5, 12, 0) is True  # lunch is inside the session
    assert at(10, 12, 0) is False  # a Saturday
    assert at(6, 12, 0) is False  # a configured holiday
    # The predicate works in the trading zone, not in whatever zone the stamp
    # happens to carry: 04:00 UTC is 09:30 IST and is inside the session.
    assert gate._is_market_open(  # noqa: SLF001 - a test of the gate itself
        datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
    ) is True
    assert gate._is_market_open(  # noqa: SLF001
        datetime(2026, 1, 10, 4, 0, tzinfo=timezone.utc)  # a Saturday in UTC too
    ) is False


def test_intraday_trade_that_exits_next_session_is_refused() -> None:
    """MIS positions must not remain open overnight to reach their label bar."""
    candles = session_grid(376, start=START_DAY, zone=trading_zone("Asia/Kolkata"))
    candidate = Candidate(
        symbol="TEST", index=374, as_of=candles[374], exit_as_of=candles[375],
        price=100.0, exit_price=101.0, score=0.9, side="BUY", quantity=10, key="once",
    )
    gate = Portfolio(permissive(), ChargeModel.zero(), zone=trading_zone("Asia/Kolkata"))
    assert gate._check(candidate, 1_000_000.0) is Reason.SESSION_CROSSING  # noqa: SLF001


def test_the_risk_gate_order_puts_the_first_failure_first() -> None:
    """`docs/contracts.md` §6: "first failure rejects the signal".

    With every gate tripped at once, the reported code is check 1's, not the
    most severe one and not an arbitrary one.
    """
    limits = permissive(
        max_position_pct=0.001,  # check 1
        max_open_positions=0,  # check 2
        max_drawdown_pct=0.0,  # check 3
        max_order_quantity=1,  # check 6
    )
    result = backtest({"TEST": sawtooth_candles(200)}, limits=limits)
    assert set(result.rejections) == {Reason.MAX_POSITION.value}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_are_correct_on_a_hand_computed_series() -> None:
    """Every headline number, computed on paper from a known price series.

    The close moves `+1, -2, +3, -4, +5` and repeats from Rs 1000, so the
    bar-to-bar change at index `i` is exactly `(1,-2,3,-4,5)[i % 5]` and a
    10-share round trip earns exactly `10x` that. Charges are zeroed so the
    arithmetic is the gross arithmetic and nothing else.

    With `horizon_bars: 1` the loop runs `i` over `[120, 128]`, nine trades whose
    P&Ls are +10, -20, +30, -40, +50, +10, -20, +30, -40: Rs 10 net, five
    winners out of nine, and a peak equity of Rs 1,000,050 from which the run
    gives back Rs 40.
    """
    n = 130
    candles = sawtooth_candles(n)
    closes = [candle.close for candle in candles]
    result = backtest({"TEST": candles}, charges=ChargeModel.zero())

    indices = list(range(FIRST_DECISION, n - 1))
    assert indices == [120, 121, 122, 123, 124, 125, 126, 127, 128]
    pnl = [10.0, -20.0, 30.0, -40.0, 50.0, 10.0, -20.0, 30.0, -40.0]

    assert result.metrics.trades == 9
    assert [t.net_pnl for t in result.trades] == pnl
    assert [t.entry_price for t in result.trades] == [closes[i] for i in indices]
    assert [t.exit_price for t in result.trades] == [closes[i + 1] for i in indices]
    assert [t.quantity for t in result.trades] == [10] * 9
    assert [t.side for t in result.trades] == ["BUY"] * 9
    assert [t.bars_held for t in result.trades] == [1] * 9

    metrics = result.metrics
    assert metrics.wins == 5 and metrics.losses == 4
    assert metrics.hit_rate == pytest.approx(5 / 9)
    assert metrics.net_pnl == pytest.approx(10.0)
    assert metrics.gross_pnl == pytest.approx(10.0)
    assert metrics.net_return == pytest.approx(10.0 / 1_000_000.0)
    assert metrics.gross_return == pytest.approx(10.0 / 1_000_000.0)
    assert metrics.charges == 0.0
    assert metrics.charges_per_trade == 0.0
    # A trade's return is on its own notional, so the mean is not the mean P&L
    # over a common price — which is exactly why it is computed this way.
    per_trade = [p / (10.0 * closes[i]) for p, i in zip(pnl, indices)]
    assert metrics.mean_trade_return == pytest.approx(sum(per_trade) / len(per_trade))
    assert metrics.worst_trade_return == pytest.approx(-40.0 / (10.0 * closes[123]))
    assert metrics.best_trade_return == pytest.approx(50.0 / (10.0 * closes[124]))
    assert metrics.median_trade_return == pytest.approx(sorted(per_trade)[len(per_trade) // 2])
    # The peak when the worst point is measured is Rs 1,000,020, not the
    # Rs 1,000,050 the run ends on: the trough at decision 124 is Rs 999,980 and
    # the run's own high comes three bars later, so a peak measured from the
    # end would understate the drawdown that actually happened.
    assert metrics.max_drawdown_pct == pytest.approx(40.0 / 1_000_020.0 * 100.0)
    # Both legs of all nine trades: 10 x the 18 entry and exit prices,
    # 1072..1073, 1073..1071, ... which sum to Rs 19,325 of price.
    assert metrics.notional == pytest.approx(193_250.0)
    assert metrics.turnover == pytest.approx(0.19325)
    # The curve includes every candle timestamp, not only model candidates.
    # Nine of the 131 opening-plus-market-bar points had a position open.
    assert metrics.steps == 131
    assert metrics.exposure_pct == pytest.approx(9 / 131)
    assert metrics.bars_per_year == BARS_PER_YEAR
    assert metrics.position_steps == 9  # nine fills, each open for exactly one step
    assert result.curve[-1].equity == pytest.approx(1_000_010.0)
    assert result.curve[0].equity == pytest.approx(1_000_000.0)
    assert result.curve[-1].as_of == candles[-1].timestamp


def test_threshold_and_no_direction_are_counted_separately() -> None:
    """A score under the threshold is not a refusal by the risk gate.

    The two are different events and belong in different lines of the report:
    `below_threshold` is the model declining to have an opinion, `RISK_*` is the
    risk engine declining to let it act. Conflating them would make a
    conservative threshold look like a broken risk gate.
    """
    quiet = backtest({"TEST": sawtooth_candles(200)}, score=0.2)
    assert quiet.metrics.trades == 0
    assert quiet.outcomes[BarOutcome.BELOW_THRESHOLD.value] == 79
    assert quiet.rejections == {}
    assert quiet.outcomes.get(BarOutcome.NO_DIRECTION.value, 0) == 0


def test_a_zero_threshold_with_a_zero_score_trades_nothing() -> None:
    """A score of exactly 0.0 has no direction and is refused, not coerced.

    `side_for_score` raises rather than pick a side, and the loop's own comment
    notes that a zero score only reaches that path with a threshold of zero.
    This is that case, and the refusal has to be counted as `no_direction` rather
    than silently becoming a buy.
    """
    result = backtest(
        {"TEST": sawtooth_candles(200)},
        score=0.0,
        section=inference(score_threshold=0.0),
    )
    assert result.outcomes[BarOutcome.NO_DIRECTION.value] == 79
    assert result.outcomes.get(BarOutcome.BELOW_THRESHOLD.value, 0) == 0
    assert result.metrics.trades == 0


def test_sharpe_annualises_by_the_stated_bar_count() -> None:
    """The annualisation is `sqrt(BARS_PER_YEAR)` and nothing else.

    Stating the bar count is the point: a Sharpe quoted without it cannot be
    checked, and a 1-minute strategy's annualised figure is enormous by
    construction.
    """
    metrics = backtest({"TEST": sawtooth_candles(200)}).metrics
    assert metrics.bars_per_year == SESSION_BARS_PER_DAY * 252 == 94_500
    assert metrics.sharpe == pytest.approx(metrics.sharpe_per_bar * math.sqrt(BARS_PER_YEAR))


def test_sharpe_of_a_flat_curve_is_zero_not_infinity() -> None:
    """No dispersion means no Sharpe, and 0.0 is the honest answer."""
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    flat = [
        EquityPoint(as_of=start + timedelta(minutes=i), equity=1_000_000.0, open_positions=0,
                    cash=1_000_000.0)
        for i in range(10)
    ]
    assert sharpe_of(flat) == 0.0


# --------------------------------------------------------------------------
# Synthetic input
# --------------------------------------------------------------------------


def test_synthetic_data_is_deterministic_under_a_fixed_seed() -> None:
    """Same seed, same candles, on any machine.

    `random.Random(str)` seeds from a SHA-512 of the string, so this does not
    depend on `PYTHONHASHSEED` and a failing assertion is reproducible without
    recording a fixture.
    """
    spec = SynthSpec(bars=300, seed=4242, start_price=1000.0)
    assert generate("TEST", spec) == generate("TEST", spec)
    assert generate("TEST", spec) != generate("TEST", SynthSpec(bars=300, seed=4243,
                                                                start_price=1000.0))
    assert generate("TEST", spec) != generate("TEST", SynthSpec(bars=300, seed=4242,
                                                                start_price=1500.0))
    # A different symbol is a different series, or a three-instrument portfolio
    # would be three copies of one bet.
    assert generate("AAA", spec) != generate("BBB", spec)


def test_the_whole_run_is_reproducible_from_a_seed() -> None:
    """Two runs of the same synthetic series produce the same trades.

    Not just the candles: the scores, the candidates, the fills and the P&L, so
    a reported number can be regenerated rather than trusted.
    """
    series = {"TEST": generate("TEST", SynthSpec(bars=200, seed=99, start_price=1000.0))}
    a = backtest(series, charges=ChargeModel())
    b = backtest(series, charges=ChargeModel())
    assert a.metrics.to_json() == b.metrics.to_json()
    assert [t.to_json() for t in a.trades] == [t.to_json() for t in b.trades]


def test_synthetic_candles_are_well_formed_and_on_the_session_grid() -> None:
    """The generator must produce candles the real path accepts, unmodified."""
    candles = generate("TEST", SynthSpec(bars=800, seed=5, start_price=1000.0))
    assert len(candles) == 800
    table = FeatureBuilder("Asia/Kolkata").build(candles)
    assert len(table.frame) == 800
    assert not table.frame.isna().to_numpy().any()

    zone = trading_zone("Asia/Kolkata")
    for index, candle in enumerate(candles):
        assert candle.low <= candle.open <= candle.high
        assert candle.low <= candle.close <= candle.high
        assert candle.close > 0.0 and candle.volume > 0
        if index:
            assert candle.timestamp > candles[index - 1].timestamp
        local = candle.timestamp.astimezone(zone)
        assert local.weekday() < 5, "no weekend bars"
        assert 555 <= local.hour * 60 + local.minute < 555 + SESSION_BARS_PER_DAY
    # A session rolls rather than running through: bar 374 is 15:29 IST and
    # bar 375 opens the next weekday at 09:15 IST.
    assert candles[374].timestamp.astimezone(zone).strftime("%H:%M") == "15:29"
    assert candles[375].timestamp.astimezone(zone).strftime("%H:%M") == "09:15"
    assert (
        candles[375].timestamp.astimezone(zone).date()
        > candles[374].timestamp.astimezone(zone).date()
    )


def test_synthetic_volume_is_heavier_at_the_session_edges_than_at_midday() -> None:
    """A flat volume profile would make `volume_ratio` unlike a live session."""
    candles = generate("TEST", SynthSpec(bars=375, seed=8, start_price=1000.0))
    per_bar_open = sum(c.volume for c in candles[:20]) / 20
    per_bar_midday = sum(c.volume for c in candles[160:215]) / 55
    assert per_bar_open > per_bar_midday * 1.8


# --------------------------------------------------------------------------
# Accounting invariants
# --------------------------------------------------------------------------


def test_charges_are_charged_on_both_legs_and_only_where_due() -> None:
    """The itemised arithmetic, checked by hand for one buy round trip."""
    charges = ChargeModel()
    buy = charges.leg(is_buy=True, quantity=10, price=1388.48)
    sell = charges.leg(is_buy=False, quantity=10, price=1388.85)

    assert buy.notional == pytest.approx(13_884.80)
    assert buy.brokerage == pytest.approx(4.16544)  # 0.03% of 13884.80, under the Rs 20 cap
    assert buy.stt == pytest.approx(13.88480)  # 0.1% of the buy leg
    assert buy.stamp_duty == pytest.approx(2.08272)  # 0.015%, buy side only
    assert buy.dp_charge == 0.0  # DP is levied on the sell transaction
    assert buy.exchange == pytest.approx(0.41237856)  # 0.00297%, both sides
    assert buy.sebi == pytest.approx(0.0138848)  # 0.0001%, both sides
    assert buy.gst == pytest.approx((4.16544 + 0.41237856 + 0.0138848) * 0.18)  # not on STT
    assert buy.total == pytest.approx(21.3857, abs=1e-3)

    assert sell.stamp_duty == 0.0
    assert sell.dp_charge == pytest.approx(15.93)
    assert sell.gst == pytest.approx((4.16655 + 0.41249145 + 0.0138885 + 15.93) * 0.18)
    assert sell.total == pytest.approx(38.1056, abs=1e-3)

    # `min(percentage, floor)`, so which term binds depends on the order size,
    # and both branches are worth pinning: at this platform's default 10-share
    # order the percentage applies, and only a notional above Rs 66,667 reaches
    # the Rs 20 floor.
    assert charges.leg(is_buy=True, quantity=10, price=1400.0).brokerage == pytest.approx(4.2)
    assert charges.leg(is_buy=True, quantity=10, price=7000.0).brokerage == pytest.approx(20.0)
    assert charges.leg(is_buy=True, quantity=10, price=1_000_000.0).brokerage == pytest.approx(20.0)
    # A zero charge model really is zero, so gross and net can be compared.
    assert ChargeModel.zero().round_trip("BUY", 10, 1400.0, 1401.0).total == 0.0


def test_a_round_trip_costs_about_half_a_percent_of_the_entry_notional() -> None:
    """The number the whole charge model exists to produce, stated once.

    At `inference.order_quantity: 10` on a ~Rs 1,400 stock a round trip is about
    Rs 60 on a Rs 14,000 entry notional, or 0.43%. A 1-bar standard deviation on
    the default synthetic series is 0.30 / sqrt(94500) = 0.098%, so a 1-minute
    decision has to beat about four and a half of its own standard deviations on
    every trade merely to break even. That is a property of the cost schedule and
    the order size, not of any strategy, and it is the number a costless
    backtest would be hiding.
    """
    trip = ChargeModel().round_trip("BUY", 10, 1400.0, 1400.0)
    cost_fraction = trip.total / (10 * 1400.0)
    assert cost_fraction == pytest.approx(0.0043, abs=2e-4)

    one_bar_sigma = DEFAULT_ANNUAL_VOLATILITY / math.sqrt(BARS_PER_YEAR)
    assert cost_fraction / one_bar_sigma == pytest.approx(4.4, abs=0.3)


def test_charges_turn_a_marginal_trade_into_a_loss() -> None:
    """The finding the cost model exists to produce.

    At a 1-minute cadence with `order_quantity: 10`, a round trip pays about
    0.43% of notional, so the strategy has to be right by more than that before
    it has made anything. A backtest with no costs would report that difference
    as an edge.
    """
    candles = {"TEST": sawtooth_candles(200)}
    costed = backtest(candles, charges=ChargeModel())
    free = backtest(candles, charges=ChargeModel.zero())

    assert costed.metrics.trades == free.metrics.trades
    assert costed.metrics.charges > 0.0
    assert costed.metrics.net_pnl < free.metrics.net_pnl
    cost_fraction = costed.metrics.charges_per_trade / (costed.trades[0].entry_price * 10)
    assert cost_fraction == pytest.approx(0.0043, abs=5e-4)


def test_a_trade_reports_its_own_charges_and_they_add_up() -> None:
    """`charges` is the two legs, and the two legs are in the JSON too."""
    result = backtest({"TEST": sawtooth_candles(200)}, charges=ChargeModel())
    trade = result.trades[0]
    assert trade.charges == pytest.approx(trade.entry_leg.total + trade.exit_leg.total)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.charges)
    assert trade.net_return == pytest.approx(trade.net_pnl / (trade.quantity * trade.entry_price))

    row = trade.to_json()
    assert row["entry_leg"] == pytest.approx(trade.entry_leg.total, abs=1e-4)
    assert row["exit_leg"] == pytest.approx(trade.exit_leg.total, abs=1e-4)
    assert sum(t.charges for t in result.trades) == pytest.approx(result.metrics.charges)
    assert sum(t.net_pnl for t in result.trades) == pytest.approx(result.metrics.net_pnl)


def test_every_fill_becomes_a_trade() -> None:
    """A position opened on the last bar is closed at its label, not left open.

    A run where the trade list and the equity curve disagree by one leg cannot be
    checked by a reader, so the tail position is realised at the label bar the
    data already holds.
    """
    result = backtest({"TEST": sawtooth_candles(200)})
    assert result.outcomes[BarOutcome.TRADED.value] == result.metrics.trades
    assert all(trade.exit_as_of > trade.entry_as_of for trade in result.trades)
    assert result.limits.capital + sum(t.net_pnl for t in result.trades) == pytest.approx(
        result.curve[-1].equity
    )


def test_a_short_is_a_short_and_a_long_is_a_long() -> None:
    """SELL is a short round trip: it profits when the price falls."""
    rising = linear_candles([round(1000.0 + i, 2) for i in range(200)])
    falling = linear_candles([round(1000.0 - 0.5 * i, 2) for i in range(200)])
    short_up = backtest({"TEST": rising}, score=-0.9, charges=ChargeModel.zero())
    short_down = backtest({"TEST": falling}, score=-0.9, charges=ChargeModel.zero())
    long_up = backtest({"TEST": rising}, score=0.9, charges=ChargeModel.zero())
    long_down = backtest({"TEST": falling}, score=0.9, charges=ChargeModel.zero())

    assert all(t.side == "SELL" for t in short_up.trades)
    assert short_up.metrics.net_pnl < 0.0 < short_down.metrics.net_pnl
    assert long_up.metrics.net_pnl > 0.0 > long_down.metrics.net_pnl
    # Monotonic series, so every fill is the same size: a Rs 1 move on 10 shares
    # is Rs 10 and a Rs 0.50 move is Rs 5. A short and a long are the same
    # magnitude with opposite signs.
    assert {t.gross_pnl for t in long_up.trades} == {10.0}
    assert {t.gross_pnl for t in short_down.trades} == {5.0}
    assert {t.gross_pnl for t in long_down.trades} == {-5.0}
    assert {t.gross_pnl for t in short_up.trades} == {-10.0}


def test_a_series_shorter_than_the_first_decision_decides_nothing() -> None:
    """It must not crash, and it must not pretend to have run."""
    result = backtest({"TEST": generate("TEST", SynthSpec(bars=100, seed=31, start_price=1000.0))})
    assert result.metrics.trades == 0
    assert result.timing.bars_scored == 0
    assert result.timing.model_ms_per_bar == 0.0
    assert result.first_decision is None
    assert result.metrics.sharpe == 0.0


def test_an_empty_symbol_set_is_refused() -> None:
    """An empty run is a refusal, not a report of zeroes."""
    with pytest.raises(BacktestError, match="no symbols"):
        backtest({})


def test_a_series_over_the_bar_limit_is_refused_not_truncated() -> None:
    """`--max-bars` is a wall. A silently shortened run is a different run."""
    candles = generate("TEST", SynthSpec(bars=200, seed=37, start_price=1000.0))
    with pytest.raises(BacktestError, match="over the --max-bars limit"):
        backtest({"TEST": candles}, max_bars=199)
    # And the message says what to do about it, because a refusal the user
    # cannot act on is a bug report waiting to happen.
    with pytest.raises(BacktestError, match="incremental"):
        backtest({"TEST": candles}, max_bars=199)


def test_the_rebuild_cost_projection_matches_what_the_run_actually_spends() -> None:
    """The estimate in the refusal message is an estimate of the real thing.

    A refusal that projects a wrong number is worse than one that projects
    none, because the user sizes `--max-bars` from it. This pins the projection
    against a measured run of the same size.
    """
    candles = generate("TEST", SynthSpec(bars=800, seed=41, start_price=1000.0))
    result = backtest({"TEST": candles}, mode="rebuild")
    measured = result.timing.feature_seconds / 60.0
    projected = projected_rebuild_minutes(800, 1)
    assert projected == pytest.approx(measured, rel=0.5)
    # And it grows quadratically, which is the whole reason for the limit.
    assert projected_rebuild_minutes(1600, 1) == pytest.approx(4 * projected, rel=0.05)
