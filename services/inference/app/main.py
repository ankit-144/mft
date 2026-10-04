"""MFT Inference Engine — FastAPI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from model import MAX_CONTEXT_ROWS, InvalidContextError

from .candles import Candle
from .config import Config, ConfigError, load_config, configure_logging
from .features import WARMUP_ROWS, FeatureError
from .predictor import PredictorBusyError
from .runtime import Runtime, build_runtime

logger = logging.getLogger("mft.inference")


class PredictRequest(BaseModel):
    """Body of `POST /v1/predict`, per `docs/contracts.md` §7."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, description="instrument symbol, e.g. RELIANCE")
    context_rows: int | None = Field(
        default=None,
        ge=1,
        description="rows to score; defaults to inference.context_rows",
    )


class PredictResponse(BaseModel):
    """Answer of `POST /v1/predict`."""

    symbol: str
    score: float
    model: str
    as_of: str


class ContextRow(BaseModel):
    """One feature row, keyed by name so a reader can find a column."""

    as_of: str
    values: dict[str, float]


class ContextResponse(BaseModel):
    """Answer of `GET /v1/context`."""

    symbol: str
    columns: list[str]
    rows: list[ContextRow]
    as_of: str
    warmup_rows: int


class HealthResponse(BaseModel):
    """Answer of `GET /healthz`."""

    status: str
    model_loaded: bool
    model: str
    dry_run: bool
    instruments: list[str]


def _error(status: int, code: str, message: str) -> HTTPException:
    """The platform's error body, as an exception FastAPI will serialise."""
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def _rfc3339(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_app(runtime: Runtime, *, run_scheduler: bool = True) -> FastAPI:
    """Build the FastAPI application around an assembled runtime."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await runtime.startup(run_scheduler=run_scheduler)
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(
        title="MFT Inference Engine",
        version="1.0.0",
        description=__doc__,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(HTTPException)
    async def contract_error(_request: Any, exc: HTTPException) -> JSONResponse:
        """Render errors in the platform's shape, not FastAPI's."""
        if isinstance(exc.detail, dict):
            content: dict[str, Any] = exc.detail
        else:
            content = {"error": "ERROR", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Liveness plus whether the model can currently score."""
        return HealthResponse(
            status="ok",
            model_loaded=runtime.predictor.is_loaded(),
            model=runtime.predictor.name,
            dry_run=runtime.dry_run,
            instruments=list(runtime.inference.instruments),
        )

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = runtime.predictor.is_loaded() and (runtime._task is None or not runtime._task.done())
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        values = [
            "# HELP mft_inference_model_loaded Whether the model is ready.",
            "# TYPE mft_inference_model_loaded gauge",
            f"mft_inference_model_loaded {int(runtime.predictor.is_loaded())}",
            "# HELP mft_inference_scheduler_ticks_total Completed scheduler passes.",
            "# TYPE mft_inference_scheduler_ticks_total counter",
            f"mft_inference_scheduler_ticks_total {runtime.scheduler.ticks}",
            "# HELP mft_inference_decisions_total Decisions by outcome.",
            "# TYPE mft_inference_decisions_total counter",
        ]
        values.extend(f'mft_inference_decisions_total{{outcome="{key}"}} {value}' for key, value in sorted(runtime.scheduler.outcome_counts.items()))
        return PlainTextResponse("\n".join(values) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/v1/context", response_model=ContextResponse)
    async def context(
        symbol: Annotated[str, Query(min_length=1)],
        rows: Annotated[int, Query(ge=1, le=MAX_CONTEXT_ROWS)] = 20,
    ) -> ContextResponse:
        """Return the last N feature rows for a symbol."""
        wanted = min(rows, runtime.inference.context_rows)
        candles = await _read(runtime, symbol, wanted + WARMUP_ROWS)
        if not candles:
            raise _error(404, "NO_CONTEXT", f"no candles are stored for {symbol}")

        try:
            table = runtime.scheduler.builder.build(candles)
        except FeatureError as err:
            raise _error(422, "BAD_CONTEXT", str(err)) from err

        context_rows = table.context(wanted)
        return ContextResponse(
            symbol=symbol,
            columns=list(table.columns),
            rows=[
                ContextRow(
                    as_of=_rfc3339(_row_time(candles, index, WARMUP_ROWS)),
                    values={name: float(value) for name, value in zip(table.columns, row)},
                )
                for index, row in enumerate(context_rows.to_numpy().tolist())
            ],
            as_of=_rfc3339(table.as_of),
            warmup_rows=table.warmup_rows,
        )

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        """Score one symbol now and return the conviction."""
        wanted = request.context_rows or runtime.inference.context_rows
        wanted = min(wanted, MAX_CONTEXT_ROWS)
        horizon = runtime.inference.horizon_bars
        symbol = request.symbol.strip().upper()

        if not runtime.predictor.is_loaded():
            raise _error(503, "MODEL_NOT_LOADED", "the model is not loaded")

        candles = await _read(runtime, symbol, wanted + WARMUP_ROWS)
        if not candles:
            raise _error(404, "NO_CONTEXT", f"no candles are stored for {symbol}")
        if len(candles) < WARMUP_ROWS:
            raise _error(
                422,
                "BAD_CONTEXT",
                f"{len(candles)} candles for {symbol} is under the {WARMUP_ROWS}-bar warm-up",
            )

        try:
            table = runtime.scheduler.builder.build(candles)
            context_rows = table.context(wanted)
        except FeatureError as err:
            raise _error(422, "BAD_CONTEXT", str(err)) from err

        try:
            score = await runtime.predictor.predict(context_rows, horizon)
        except PredictorBusyError as err:
            raise _error(503, "MODEL_BUSY", str(err)) from err
        except (InvalidContextError, ValueError) as err:


            raise _error(422, "BAD_CONTEXT", str(err)) from err

        return PredictResponse(
            symbol=symbol,
            score=score,
            model=runtime.predictor.name,
            as_of=_rfc3339(table.as_of),
        )

    return app


async def _read(runtime: Runtime, symbol: str, limit: int) -> list[Candle]:
    """Fetch candles, turning a store failure into a 502."""
    try:
        return await runtime.store.tail(symbol, limit)
    except Exception as err:  # noqa: BLE001
        logger.warning("cannot read candles for %s: %s", symbol, err)
        raise _error(
            502,
            "STORE_UNAVAILABLE",
            f"could not read candles for {symbol}: {type(err).__name__}: {err}",
        ) from err


def _row_time(candles: list[Candle], index: int, warmup: int) -> datetime:
    """The candle timestamp behind context row `index`."""
    position = warmup + index
    if position < len(candles):
        return candles[position].timestamp
    return candles[-1].timestamp


def build_app() -> FastAPI:
    """Load the config from `MFT_CONFIG` and assemble the whole service."""
    config: Config = load_config()
    configure_logging(config.app.log_level)
    runtime = build_runtime(config)
    return create_app(runtime)


def _module_app() -> FastAPI:
    """The ASGI app `uvicorn app.main:app` is given."""
    try:
        return build_app()
    except (ConfigError, RuntimeError) as err:
        logger.error("cannot start the inference service: %s", err)
        raise


def __getattr__(name: str) -> Any:
    if name == "app":
        return _module_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContextResponse",
    "ContextRow",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "build_app",
    "create_app",
]
