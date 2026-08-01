"""MFT Inference Engine — FastAPI entrypoint.

Loads the TabFM model once and exposes /v1/predict for minute-close
inference. Candles are pulled from the fluxKV API endpoint (Go).
"""

import logging

from fastapi import FastAPI

from .model import TabFMModel

logger = logging.getLogger("mft.inference")

app = FastAPI(title="MFT Inference Engine")
model = TabFMModel()


@app.on_event("startup")
async def load_model() -> None:
    await model.load()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "model_loaded": model.is_loaded()}


@app.post("/v1/predict")
async def predict(payload: dict) -> dict:
    """Accepts a minute-close payload and returns a conviction score."""
    score = model.predict(payload)
    return {"score": score}
