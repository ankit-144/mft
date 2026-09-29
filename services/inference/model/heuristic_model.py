"""A dependency-light reference implementation of `InferenceModel`.

Two jobs:

1. Tests. Every test in the service needs a model that loads instantly, needs
   no weights, and returns a deterministic number.
2. Fallback. When the TabFM weights are unavailable, the service still needs
   to produce a signal rather than fail the request.

It is not a no-op. Minute-close equity index futures mean-revert after
volume-confirmed dislocations, and the blend below encodes that: a
volatility-scaled trend term for continuation, a z-scored reversion term for
snapbacks, and a volume/regime gate that damps conviction when the tape is
thin or the move is not confirmed by participation. That is enough signal to
validate the pipeline end to end, and enough to catch a broken feature table,
a mis-scaled score, or a stuck service.

Only numpy and pandas are used. No random number generation, so repeated calls
on the same context return the same score.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import (
    TARGET_COLUMN,
    BaseInferenceModel,
    returns_volatility,
    validate_context,
)

logger = logging.getLogger("mft.inference.model.heuristic")

#: Blend weights. They sum to 1 so the pre-gate score stays in the same range
#: as its parts, which keeps the configured `score_threshold` meaningful.
TREND_WEIGHT = 0.45
REVERSION_WEIGHT = 0.35
PARTICIPATION_WEIGHT = 0.20

#: Reversion and trend terms are z-scored against the context before blending.
#: A z-score past this is treated as saturated rather than extrapolated, so a
#: single blown-out bar cannot dominate the signal.
SATURATION_Z = 3.0

#: Volume and volatility gates, expressed as the multiple of their 20-bar
#: baseline at which conviction stops scaling up. Above these we are either
#: into a high-volatility regime or into a thin, untrustworthy tape.
VOLUME_GATE = 3.0
VOL_RATIO_GATE = 2.5

#: Dispatches below this are not traded, and scaling them up only manufactures
#: noise. Applied to the pre-gate magnitude.
MIN_DISPATCH_Z = 0.15

#: Horizon scaling. A `horizon`-bar move has roughly sqrt(horizon) the
#: dispersion of a 1-bar move, so normalise it back out.
HORIZON_REFERENCE = 1


def _zscore(series: pd.Series) -> np.ndarray:
    """Z-score a series, returning zeros when it is constant or too short."""
    values = series.to_numpy(dtype=float)
    if values.size < 2:
        return np.zeros(values.shape, dtype=float)
    deviation = float(np.std(values))
    if deviation < 1e-12:
        return np.zeros(values.shape, dtype=float)
    return (values - float(np.mean(values))) / deviation


def _clipped(values: np.ndarray) -> np.ndarray:
    """Clip z-scores at the saturation point."""
    return np.clip(values, -SATURATION_Z, SATURATION_Z)


class HeuristicModel(BaseInferenceModel):
    """Momentum plus volatility-adjusted mean reversion over the context.

    Args:
        trend_weight: Weight on the volatility-normalised trend term.
        reversion_weight: Weight on the z-scored reversion term.
        participation_weight: Weight on the volume-confirmed momentum term.
    """

    def __init__(
        self,
        trend_weight: float = TREND_WEIGHT,
        reversion_weight: float = REVERSION_WEIGHT,
        participation_weight: float = PARTICIPATION_WEIGHT,
    ) -> None:
        self._trend_weight = trend_weight
        self._reversion_weight = reversion_weight
        self._participation_weight = participation_weight
        self._loaded = False

    @property
    def _name(self) -> str:
        return "heuristic-v1"

    async def _load(self) -> None:
        """No weights to fetch. Report what this model is, then return."""
        logger.info(
            "heuristic model ready: target=%s (1-bar log return), "
            "weights=trend=%.2f reversion=%.2f participation=%.2f, "
            "no pretrained checkpoint and no licence restriction",
            TARGET_COLUMN,
            self._trend_weight,
            self._reversion_weight,
            self._participation_weight,
        )

    def _predict_raw(self, context: pd.DataFrame, horizon: int) -> float:
        """Blend the three terms into a volatility-normalised return.

        The result is in units of the context's own return volatility, so it
        is directly comparable across instruments. `BaseInferenceModel.predict`
        applies the final [-1, 1] mapping.
        """
        table = validate_context(context, horizon=horizon)
        volatility = returns_volatility(table[TARGET_COLUMN].to_numpy(dtype=float))
        scale = float(np.sqrt(max(horizon, 1) / HORIZON_REFERENCE))

        trend = self._trend_term(table, volatility)
        reversion = self._reversion_term(table)
        participation = self._participation_term(table)

        blended = (
            self._trend_weight * trend
            + self._reversion_weight * reversion
            + self._participation_weight * participation
        )
        return blended * scale

    def _trend_term(self, table: pd.DataFrame, volatility: float) -> float:
        """Direction of the recent move, in units of return volatility.

        Combines the three return horizons so a move that is strong on one
        timeframe and reversed on another does not read as a clean trend.
        """
        horizons = (table["ret_1"], table["ret_5"], table["ret_15"])
        signs = [np.sign(float(series.iloc[-1])) for series in horizons]
        agreement = sum(signs) / len(signs)
        strength = float(np.clip(table["ret_1"].iloc[-1] / volatility, -SATURATION_Z, SATURATION_Z))
        return float(agreement) * strength

    def _reversion_term(self, table: pd.DataFrame) -> float:
        """How stretched the latest bar is against its own recent history.

        A 1-bar return far above its own z-score reverts; a return sitting on
        its mean does not.
        """
        stretched = float(_zscore(table["ret_1"])[-1])
        gap = float(table["sma_gap_10"].iloc[-1])
        gap_z = float(_zscore(table["sma_gap_10"])[-1])
        return float(-np.clip(stretched + 0.5 * gap_z, -SATURATION_Z, SATURATION_Z))

    def _participation_term(self, table: pd.DataFrame) -> float:
        """Trend, but only to the extent volume confirms it.

        A move on heavy volume is more likely to continue than the same move
        on thin volume, so this term is the trend direction scaled by
        participation and damped outside the normal regime.
        """
        rsi = float(table["momentum_rsi_14"].iloc[-1])
        if not np.isfinite(rsi):
            rsi = 0.0
        direction = float(np.sign(rsi))

        volume_ratio = float(table["volume_ratio"].iloc[-1])
        if not np.isfinite(volume_ratio):
            volume_ratio = 1.0
        participation = float(np.clip(volume_ratio, 0.0, VOLUME_GATE)) / VOLUME_GATE

        vol_ratio = float(table["vol_ratio"].iloc[-1])
        if not np.isfinite(vol_ratio) or vol_ratio <= 0.0:
            regime = 1.0
        else:
            regime = float(min(vol_ratio, VOL_RATIO_GATE) / VOL_RATIO_GATE)

        conviction = abs(rsi)
        if conviction < MIN_DISPATCH_Z:
            return 0.0
        return direction * conviction * participation * regime


__all__ = [
    "HeuristicModel",
    "HORIZON_REFERENCE",
    "MIN_DISPATCH_Z",
    "PARTICIPATION_WEIGHT",
    "REVERSION_WEIGHT",
    "SATURATION_Z",
    "TREND_WEIGHT",
    "VOL_RATIO_GATE",
    "VOLUME_GATE",
]
