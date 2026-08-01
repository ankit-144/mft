"""TabFM model wrapper. Instantiated once at startup and held in memory.

The actual TabFM import and tensor formatting will be added here once the
model weights and feature schema are available.
"""

import logging

logger = logging.getLogger("mft.inference.model")


class TabFMModel:
    def __init__(self) -> None:
        self._loaded = False

    async def load(self) -> None:
        # TODO: load TabFM weights into GPU VRAM (or system RAM) and pin memory.
        logger.info("TabFM model loaded")
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, payload: dict) -> float:
        # TODO: format candles into Pandas DataFrame, run TabFM, return score.
        return 0.0
