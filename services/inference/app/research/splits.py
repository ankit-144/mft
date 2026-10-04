"""Chronological split construction with target-overlap purge and embargo."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Half-open candle index ranges for decision rows in one symbol."""

    first_decision: int
    stop: int
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]
    train_cut: int
    validation_cut: int
    purge_bars: int
    embargo_bars: int

    def indices(self, name: str) -> range:
        """Return the decision index range for one split."""
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {name!r}")
        lo, hi = getattr(self, name)
        return range(lo, hi)


@dataclass(frozen=True, slots=True)
class GlobalSplitPlan:
    """Per-symbol index ranges cut at shared chronological timestamps."""

    plans: Mapping[str, SplitPlan]
    train_cut: datetime
    validation_cut: datetime


def plan_global_splits(
    timestamps: Mapping[str, Sequence[datetime]],
    *,
    horizon: int,
    first_decision: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    purge_bars: int | None = None,
    embargo_bars: int | None = None,
) -> GlobalSplitPlan:
    """Use common timestamp boundaries even when symbols have unequal histories."""
    if horizon < 1 or first_decision < 0:
        raise ValueError("horizon must be positive and first_decision nonnegative")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("split fractions must sum to less than one")
    purge = horizon if purge_bars is None else purge_bars
    embargo = horizon if embargo_bars is None else embargo_bars
    if purge < horizon:
        raise ValueError("purge_bars must be at least horizon to prevent label overlap")
    if embargo < 0:
        raise ValueError("embargo_bars must be nonnegative")
    if not timestamps or any(list(values) != sorted(values) for values in timestamps.values()):
        raise ValueError("timestamp series must be nonempty and ascending")

    usable = sorted({stamp for rows in timestamps.values() for stamp in rows[first_decision:-horizon]})
    train_index = int(len(usable) * train_fraction)
    validation_index = int(len(usable) * (train_fraction + validation_fraction))
    if train_index <= 0 or validation_index <= train_index or validation_index >= len(usable):
        raise ValueError("global timestamp range is too short for nonempty train/validation/test splits")
    train_cut, validation_cut = usable[train_index], usable[validation_index]
    plans: dict[str, SplitPlan] = {}
    for symbol, rows in timestamps.items():
        train_boundary = bisect_left(rows, train_cut)
        validation_boundary = bisect_left(rows, validation_cut)
        stop = len(rows) - horizon
        plan = SplitPlan(
            first_decision=first_decision,
            stop=stop,
            train=(first_decision, min(stop, train_boundary - purge)),
            validation=(min(stop, train_boundary + embargo), min(stop, validation_boundary - purge)),
            test=(min(stop, validation_boundary + embargo), stop),
            train_cut=train_boundary,
            validation_cut=validation_boundary,
            purge_bars=purge,
            embargo_bars=embargo,
        )
        if any(lo >= hi for lo, hi in (plan.train, plan.validation, plan.test)):
            raise ValueError(f"{symbol} does not span all global train/validation/test windows")
        plans[symbol] = plan
    return GlobalSplitPlan(plans, train_cut, validation_cut)


def plan_splits(
    candle_count: int,
    *,
    horizon: int,
    first_decision: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    purge_bars: int | None = None,
    embargo_bars: int | None = None,
) -> SplitPlan:
    """Split usable decision rows chronologically and guard both boundaries."""
    if horizon < 1 or first_decision < 0 or candle_count < 0:
        raise ValueError("horizon and candle_count must be positive and first_decision nonnegative")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    purge = horizon if purge_bars is None else purge_bars
    embargo = horizon if embargo_bars is None else embargo_bars
    if purge < horizon:
        raise ValueError("purge_bars must be at least horizon to prevent label overlap")
    if embargo < 0:
        raise ValueError("embargo_bars must be nonnegative")

    stop = candle_count - horizon
    usable = max(0, stop - first_decision)
    train_n = int(usable * train_fraction)
    validation_n = int(usable * validation_fraction)
    train_cut = first_decision + train_n
    validation_cut = train_cut + validation_n
    plan = SplitPlan(
        first_decision=first_decision,
        stop=stop,
        train=(first_decision, train_cut - purge),
        validation=(train_cut + embargo, validation_cut - purge),
        test=(validation_cut + embargo, stop),
        train_cut=train_cut,
        validation_cut=validation_cut,
        purge_bars=purge,
        embargo_bars=embargo,
    )
    if any(lo >= hi for lo, hi in (plan.train, plan.validation, plan.test)):
        raise ValueError(
            f"{candle_count} candles do not leave nonempty train/validation/test splits "
            f"after {purge}-bar purge and {embargo}-bar embargo"
        )
    return plan


__all__ = ["GlobalSplitPlan", "SplitPlan", "plan_global_splits", "plan_splits"]
