"""Cross-language feature parity: this port against C3's Go builder.

`docs/contracts.md` §4 freezes 18 columns, and `core/features` (C3) implements
them in Go. This service reimplements them in Python, which is the one thing
the architecture cannot have twice: a model served on a subtly different
distribution than it was trained on does not fail, it fails quietly, and the
drift is invisible until the strategy stops making money.

So the two implementations are pinned to the same numbers. The fixture is
`fixtures/feature_golden.json`, transcribed from `core/features/golden_test.go`:
the same 64 bars and the same seven expected feature rows. Go asserts them at
1e-9 relative (`assertClose` in `core/features/features_test.go`), and so does
this file. C3's own comment records that its values came from a third
implementation written from the contract, so the vectors are a property of the
contract rather than of either codebase.

What this test cannot catch is covered at the end: if the Go builder changes
and this file is not updated, both suites still pass and the two have drifted.
The fix is to move the vectors into one shared file both sides read. That
change is outside this component's ownership and is called out in the report.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.features import (
    SESSION_OPEN_MINUTE,
    WARMUP_ROWS,
    FeatureBuilder,
    FeatureError,
    trading_zone,
)
from support import golden_candles, load_golden

#: The same tolerance `core/features` uses in `assertClose`.
TOLERANCE = 1e-9


def assert_close(got: float, want: float, name: str) -> None:
    """Assert `got` is within the Go suite's tolerance of `want`."""
    tol = TOLERANCE * max(1.0, abs(want))
    assert abs(got - want) <= tol, f"{name} = {got!r}, want {want!r} (tolerance {tol:.3g})"


def test_golden_rows_match_the_go_fixture() -> None:
    """Every golden row, value by value, in schema order."""
    document = load_golden()
    columns = document["columns"]
    builder = FeatureBuilder(document["zone"])
    table = builder.build(golden_candles(document["symbol"]))

    assert list(table.columns) == columns
    assert len(table.frame) == len(document["bars"])
    assert not table.frame.isna().to_numpy().any(), "the table must be dense"

    for index, expected in sorted(document["rows"].items(), key=lambda kv: int(kv[0])):
        row = table.frame.iloc[int(index)].tolist()
        assert len(row) == 18
        for name, want in zip(columns, expected):
            assert_close(row[columns.index(name)], want, f"row {index} {name}")


def test_golden_fixture_is_anchored_where_the_go_fixture_is() -> None:
    """The fixture starts at 07:50 UTC, which is 13:20 IST.

    Worth pinning separately: the two calendar features are the ones a
    timezone mix-up breaks, and it breaks them silently — the numbers stay
    finite, just wrong by an offset.
    """
    document = load_golden()
    candles = golden_candles(document["symbol"])
    start = datetime.fromisoformat(document["start_utc"].replace("Z", "+00:00"))

    assert candles[0].timestamp == start == datetime(2026, 8, 3, 7, 50, tzinfo=timezone.utc)

    ist = trading_zone("Asia/Kolkata")
    local = candles[0].timestamp.astimezone(ist)
    assert (local.hour, local.minute) == (13, 20)
    # 13:20 IST is 800 minutes past midnight; the session opens at 555.
    assert local.hour * 60 + local.minute - SESSION_OPEN_MINUTE == 245


def test_builder_is_row_aligned_with_its_input() -> None:
    """One row per candle, in order, with AsOf from the last candle.

    The alignment is the contract, not an implementation detail: the model
    zips a feature row with the candle it came from, so a table shifted by the
    warm-up length would be perfectly well-formed and perfectly wrong.
    """
    candles = golden_candles()
    table = FeatureBuilder("Asia/Kolkata").build(candles)

    assert len(table.frame) == len(candles)
    assert table.as_of == candles[-1].timestamp
    # ret_1 is log(close[i] / close[i-1]): the cheapest proof that row i
    # belongs to candle i.
    for index in (1, 2, 17, 63):
        want = math.log(candles[index].close / candles[index - 1].close)
        got = table.frame["ret_1"].iloc[index]
        assert_close(got, want, f"ret_1 at row {index}")


def test_newest_row_does_not_depend_on_how_much_history_was_pulled() -> None:
    """A 200-candle build and a 64-candle build agree on every informed row.

    This is the property that makes pull viable. If a feature's value drifted
    with the length of the window, inference pulling 100 bars a minute and a
    backfill replaying 100000 would hand the model two different numbers for
    the same market state, and neither would be wrong enough to notice.

    The comparison starts at row WARMUP_ROWS, and that is not a convenience.
    Before it, a feature is a function of a window the window did not contain:
    row 0 has no previous bar, so its ret_1 is zero in the short build and a
    real return in the long one. Those rows are declared context-only by
    contract, and the loop drops them before scoring. From row WARMUP_ROWS on,
    every feature has its full history and the two must agree exactly.
    """
    from support import make_candles

    long_window = make_candles(rows=200)
    short_window = long_window[-64:]

    short = FeatureBuilder("Asia/Kolkata").build(short_window)
    long = FeatureBuilder("Asia/Kolkata").build(long_window)

    assert len(short.frame) == 64
    assert len(long.frame) == 200
    for index in range(WARMUP_ROWS, len(short.frame)):
        a = short.frame.iloc[index]
        b = long.frame.iloc[len(long.frame) - 64 + index]
        for name in short.columns:
            assert_close(
                float(b[name]),
                float(a[name]),
                f"{name} at overlapping row {index}",
            )


def test_warmup_rows_are_dropped_not_zero_padded() -> None:
    """The context handed to a model contains no warm-up rows."""
    builder = FeatureBuilder("Asia/Kolkata")
    table = builder.build(golden_candles())

    assert WARMUP_ROWS == 60
    assert table.warmup_rows == WARMUP_ROWS
    context = table.context()
    assert len(context) == len(table.frame) - WARMUP_ROWS
    # The newest row is still the newest row after trimming.
    assert table.frame["minute_of_session"].iloc[-1] == context["minute_of_session"].iloc[-1]


def test_context_caps_the_prompt() -> None:
    """`context(rows)` keeps the most recent rows, not the oldest."""
    from support import make_candles

    table = FeatureBuilder("Asia/Kolkata").build(make_candles(rows=200))
    context = table.context(50)

    assert len(context) == 50
    assert table.frame["minute_of_session"].iloc[-1] == context["minute_of_session"].iloc[-1]
    assert table.frame["minute_of_session"].iloc[-60] != context["minute_of_session"].iloc[0]


def test_degenerate_bars_take_the_documented_zero() -> None:
    """A doji, a bar that never traded and a flat window are zeros, not NaNs.

    Every one of these is an ordinary market state, and every one of them
    divides by zero somewhere in the schema. The policy is uniform — a zero
    column means "no information" — and a NaN would poison a whole context
    table rather than degrade one row.
    """
    document = load_golden()
    table = FeatureBuilder(document["zone"]).build(golden_candles())
    frame = table.frame

    doji = frame.loc[25]
    for name in ("range_1", "body_1", "upper_wick_1", "lower_wick_1", "spread_proxy"):
        assert doji[name] == 0.0, f"{name} on a doji"

    no_volume = frame.loc[33]
    assert no_volume["spread_proxy"] == 0.0
    assert no_volume["volume_ratio"] == 0.0
    assert no_volume["volume_z_20"] < 0.0, "a zero volume is far below its window"

    flat = frame.loc[39]
    for name in ("vol_5", "vol_20", "vol_ratio", "momentum_rsi_14"):
        assert flat[name] == 0.0, f"{name} on a 20-bar flat window"
    assert_close(float(flat["sma_gap_10"]), 0.0, "sma_gap_10 on a flat window")


def test_the_warmup_boundary_is_where_the_contract_says_it_is() -> None:
    """Row WARMUP_ROWS-1 is context-only; row WARMUP_ROWS is a real target."""
    table = FeatureBuilder("Asia/Kolkata").build(golden_candles())

    assert table.frame["ret_60"].iloc[WARMUP_ROWS - 1] == 0.0
    assert table.frame["ret_60"].iloc[WARMUP_ROWS] != 0.0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda rows: rows[:59], "under the warm-up"),
        (lambda rows: rows + [rows[-1]], "not strictly ascending"),
        (lambda rows: rows[:30] + [rows[0]] + rows[31:], "a rewound timestamp"),
        (lambda rows: rows[:10] + [_replace(rows[10], symbol="TCS")] + rows[11:], "two symbols"),
        (lambda rows: [_replace(rows[20], close=-1.0)] + rows[21:], "a non-positive price"),
        (lambda rows: [_replace(rows[20], close=0.0)] + rows[21:], "a zero price"),
        (lambda rows: [_replace(rows[20], high=rows[20].open * 0.5)] + rows[21:], "high below open"),
        (lambda rows: [_replace(rows[20], low=rows[20].open * 2.0)] + rows[21:], "low above open"),
        (lambda rows: [_replace(rows[20], volume=-1)] + rows[21:], "negative volume"),
        (lambda rows: [_replace(rows[20], close=float("nan"))] + rows[21:], "a NaN price"),
    ],
)
def test_bad_input_is_an_error_not_a_table_of_zeros(mutate, reason: str) -> None:
    """Every rejecting check in the Go builder is rejecting here too.

    All of them produce finite numbers if they are skipped, which is exactly
    why they are explicit: fabricating features from bad data is how a broken
    feed turns into a confidently wrong signal an hour later.
    """
    builder = FeatureBuilder("Asia/Kolkata")
    with pytest.raises(FeatureError):
        builder.build(mutate(golden_candles()))


def test_a_naive_timestamp_is_refused_by_the_calendar_features() -> None:
    """A candle with no zone is ambiguous, and the calendar features need one.

    `as_of` on the wire is always UTC. A naive timestamp here would make
    `minute_of_session` depend on the machine's local zone, which is the drift
    the whole port exists to prevent.
    """
    rows = golden_candles()
    naive = [_replace(rows[0], timestamp=rows[0].timestamp.replace(tzinfo=None))] + rows[1:]
    with pytest.raises(FeatureError, match="no timezone"):
        FeatureBuilder("Asia/Kolkata").build(naive)


def test_ist_falls_back_to_a_fixed_offset_without_a_tz_database(monkeypatch) -> None:
    """A slim container with no tzdata still labels bars in IST.

    `Asia/Kolkata` has observed +05:30 with no daylight saving since 1945, so
    the fallback is exact for every timestamp the platform can hold. C3's Go
    tests fall back the same way.
    """
    import app.features as features

    def no_zone(name: str) -> None:
        raise features.ZoneInfoNotFoundError(name)

    monkeypatch.setattr(features, "ZoneInfo", no_zone)
    assert features.trading_zone("Asia/Kolkata") == features.IST_FIXED_OFFSET

    with pytest.raises(features.FeatureError):
        features.trading_zone("Mars/Olympus_Mons")


def test_warmup_length_agrees_with_the_models_own_floor() -> None:
    """This port's warm-up and C5's model floor are the same 60 rows.

    Two components, two constants, one number. If either moved, a window would
    be either full of warm-up zeros or shorter than the model accepts, and
    neither failure is loud.
    """
    from model import MIN_CONTEXT_ROWS

    assert WARMUP_ROWS == MIN_CONTEXT_ROWS == 60


def _replace(candle, **changes):
    """A copy of a candle with fields changed."""
    from dataclasses import replace

    return replace(candle, **changes)


def test_fixture_agrees_with_its_own_recorded_source() -> None:
    """The fixture names the Go file it was transcribed from.

    Cheap, and it is the line a reader checks first when the two sides
    disagree.
    """
    document = load_golden()
    assert document["source"] == "core/features/golden_test.go"
    assert len(document["columns"]) == 18
    assert document["start_utc"].endswith("Z")
    assert timedelta(hours=0) == datetime.fromisoformat(
        document["start_utc"].replace("Z", "+00:00")
    ).utcoffset()
