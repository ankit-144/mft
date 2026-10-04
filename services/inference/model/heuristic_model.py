"""A dependency-light reference implementation of `InferenceModel`."""

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


TREND_WEIGHT = 0.45
REVERSION_WEIGHT = 0.35
PARTICIPATION_WEIGHT = 0.20


SATURATION_Z = 3.0


VOLUME_GATE = 3.0
VOL_RATIO_GATE = 2.5


MIN_DISPATCH_Z = 0.15


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
    """Momentum plus volatility-adjusted mean reversion over the context."""

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
        """No weights to fetch."""
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
        """Blend the three terms into a volatility-normalised return."""
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
        """Direction of the recent move, in units of return volatility."""
        horizons = (table["ret_1"], table["ret_5"], table["ret_15"])
        signs = [np.sign(float(series.iloc[-1])) for series in horizons]
        agreement = sum(signs) / len(signs)
        strength = float(np.clip(table["ret_1"].iloc[-1] / volatility, -SATURATION_Z, SATURATION_Z))
        return float(agreement) * strength

    def _reversion_term(self, table: pd.DataFrame) -> float:
        """How stretched the latest bar is against its own recent history."""
        stretched = float(_zscore(table["ret_1"])[-1])
        gap = float(table["sma_gap_10"].iloc[-1])
        gap_z = float(_zscore(table["sma_gap_10"])[-1])
        return float(-np.clip(stretched + 0.5 * gap_z, -SATURATION_Z, SATURATION_Z))

    def _participation_term(self, table: pd.DataFrame) -> float:
        """Trend, but only to the extent volume confirms it."""
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
