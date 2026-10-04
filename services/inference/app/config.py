"""Configuration, read from the same YAML the Go services read."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("mft.inference.config")


CONFIG_ENV_VAR: Final[str] = "MFT_CONFIG"


DEFAULT_CONFIG_PATH: Final[str] = "configs/config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing, unparsable or invalid."""


class AppConfig(BaseModel):
    """`app` — general application settings."""

    model_config = ConfigDict(extra="ignore")

    name: str = "mft"
    env: str = "dev"
    log_level: str = "info"
    timezone: str = "Asia/Kolkata"

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        level = value.strip().lower()
        if level not in {"debug", "info", "warn", "error"}:
            raise ValueError(f"app.log_level {value!r} is not debug|info|warn|error")
        return level


class StorageConfig(BaseModel):
    """`storage` — the Parquet cold store this service reads from."""

    model_config = ConfigDict(extra="ignore")

    data_dir: str = "data"
    flush_interval_seconds: int = Field(default=60, gt=0)
    flush_max_rows: int = Field(default=10000, gt=0)
    partition_by: str = "date"

    @field_validator("partition_by")
    @classmethod
    def _date_only(cls, value: str) -> str:
        if value.strip() != "date":
            raise ValueError(f"storage.partition_by {value!r} is not supported, only 'date'")
        return "date"


class InferenceConfig(BaseModel):
    """`inference` — everything the loop is configured with."""

    model_config = ConfigDict(extra="ignore")

    addr: str = ":8000"
    model: str = "heuristic"
    execution_url: str = "http://localhost:8080"
    context_rows: int = 100
    horizon_bars: int = 1
    score_threshold: float = 0.0
    order_quantity: int = 10
    instruments: list[str] = Field(default_factory=list)
    dry_run: bool = True
    max_concurrency: int = Field(default=4, ge=1, le=64)
    max_context_age_seconds: int = Field(default=120, gt=0)
    cursor_path: str = ""

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        name = value.strip().lower()
        if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise ValueError(f"inference.model {value!r} must be a registry identifier")
        return name

    @field_validator("score_threshold")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"inference.score_threshold {value} is outside [0, 1]")
        return value

    @field_validator("context_rows")
    @classmethod
    def _positive_rows(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"inference.context_rows must be positive, got {value}")
        return value

    @field_validator("horizon_bars")
    @classmethod
    def _positive_horizon(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"inference.horizon_bars must be positive, got {value}")
        return value

    @field_validator("order_quantity")
    @classmethod
    def _positive_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"inference.order_quantity must be positive, got {value}")
        return value

    @field_validator("execution_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"inference.execution_url {value!r} is not an http(s) URL")
        return url

    @field_validator("instruments")
    @classmethod
    def _normalise(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for raw in value:
            symbol = raw.strip().upper()
            if not symbol:
                raise ValueError("inference.instruments contains an empty symbol")
            if symbol not in seen:
                seen.append(symbol)
        return seen


class ExecutionConfig(BaseModel):
    """Credentials and operating mode shared with execution."""

    model_config = ConfigDict(extra="ignore")
    paper_trading: bool = True
    api_token: str = Field(default="", exclude=True, repr=False)


class JobsConfig(BaseModel):
    """Published historical candles used by research and live warm-up."""

    model_config = ConfigDict(extra="ignore")
    historical_dir: str = ""


class Config(BaseModel):
    """The subset of the frozen schema this service reads."""

    model_config = ConfigDict(extra="ignore")

    app: AppConfig = Field(default_factory=AppConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)


def resolve_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Locate the config file: an explicit path, else `MFT_CONFIG`, else default."""
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    from_root = Path(__file__).resolve().parents[3] / candidate
    if from_root.exists():
        return from_root
    return candidate


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Read and validate the YAML config."""
    resolved = resolve_path(path)
    try:
        text = resolved.read_text()
    except OSError as err:
        raise ConfigError(f"cannot read config {resolved}: {err}") from err

    try:
        document = yaml.safe_load(text) or {}
    except yaml.YAMLError as err:
        raise ConfigError(f"cannot parse config {resolved}: {err}") from err

    if not isinstance(document, dict):
        raise ConfigError(f"config {resolved} is not a YAML mapping")

    overrides = {"model": "MFT_INFERENCE_MODEL", "execution_url": "MFT_EXECUTION_URL"}
    inference = document.setdefault("inference", {})
    if isinstance(inference, dict):
        for key, env in overrides.items():
            if value := os.environ.get(env):
                inference[key] = value

    try:
        config = Config.model_validate(document)
    except ValueError as err:
        raise ConfigError(f"invalid config {resolved}: {err}") from err

    base = resolved.resolve().parent
    if base.name == "configs":
        base = base.parent
    config.storage.data_dir = str(_absolute(base, config.storage.data_dir))
    cursor = config.inference.cursor_path or str(Path(config.storage.data_dir) / "inference" / "cursors.json")
    config.inference.cursor_path = str(_absolute(base, cursor))
    historical = config.jobs.historical_dir or str(Path(config.storage.data_dir) / "historical")
    config.jobs.historical_dir = str(_absolute(base, historical))
    if config.inference.horizon_bars >= config.inference.context_rows:
        raise ConfigError("inference.horizon_bars must be below context_rows")

    logger.debug(
        "loaded config %s: model=%s dry_run=%s instruments=%s context_rows=%d",
        resolved,
        config.inference.model,
        config.inference.dry_run,
        ",".join(config.inference.instruments) or "<none>",
        config.inference.context_rows,
    )
    return config


def _absolute(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def configure_logging(level: str) -> None:
    """Send the service's logs to stdout at `level`, in a parseable shape."""
    import logging as _logging

    handler = _logging.StreamHandler()
    handler.setFormatter(
        _logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = _logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(_logging, level.upper(), _logging.INFO))


def data_dir_of(config: Config) -> Path:
    """The configured `storage.data_dir`, as a path."""
    return Path(config.storage.data_dir)


def as_mapping(config: Config) -> dict[str, Any]:
    """The config as plain data, for logging and for the debug endpoints."""
    return config.model_dump()


__all__ = [
    "AppConfig",
    "CONFIG_ENV_VAR",
    "Config",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "InferenceConfig",
    "StorageConfig",
    "as_mapping",
    "configure_logging",
    "data_dir_of",
    "load_config",
    "resolve_path",
]
