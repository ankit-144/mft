"""Configuration, read from the same YAML the Go services read.

`docs/contracts.md` §8 freezes the schema and `core/config/config.go`
implements it. This module is the Python reading of the same document: it
declares no key that `config.go` does not, and it defaults every key exactly
where `config.Validate` does, so a service started with an empty config file
behaves the same as a Go one.

`MFT_CONFIG` names the file, defaulting to `configs/config.yaml`, which is what
`make dev` and `make run-inference` set.

# PyYAML is a transitive dependency

`requirements.txt` is frozen and does not name PyYAML; it arrives with
`uvicorn[standard]`, which every documented way of running this service uses.
This is a real coupling and is reported rather than worked around — the fix is
one line in a file this component does not own.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("mft.inference.config")

#: Environment variable naming the config file. Same name the Go services read.
CONFIG_ENV_VAR: Final[str] = "MFT_CONFIG"

#: Used when `MFT_CONFIG` is unset. Relative to the working directory, which is
#: the repository root for `make run-inference`... except that target `cd`s
#: into `services/inference`, so the path is also tried relative to the
#: directory two levels up. See `resolve_path`.
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
    flush_interval_seconds: int = 300
    flush_max_rows: int = 10000
    partition_by: str = "date"

    @field_validator("partition_by")
    @classmethod
    def _date_only(cls, value: str) -> str:
        if value.strip() != "date":
            raise ValueError(f"storage.partition_by {value!r} is not supported, only 'date'")
        return "date"


class InferenceConfig(BaseModel):
    """`inference` — everything the loop is configured with.

    `dry_run` defaults to `True` and `model` to `tabfm`, matching
    `config.Validate`. The default is the safe one and this module will not
    make it any other way: a signal that leaves this process becomes an order
    at the broker.
    """

    model_config = ConfigDict(extra="ignore")

    addr: str = ":8000"
    model: str = "tabfm"
    execution_url: str = "http://localhost:8080"
    context_rows: int = 100
    horizon_bars: int = 1
    score_threshold: float = 0.55
    order_quantity: int = 10
    instruments: list[str] = Field(default_factory=list)
    dry_run: bool = True

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        name = value.strip().lower()
        if name not in {"tabfm", "heuristic"}:
            raise ValueError(f"inference.model {value!r} must be one of tabfm, heuristic")
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


class Config(BaseModel):
    """The subset of the frozen schema this service reads.

    Sections it does not read — broker, analytics, execution, jobs — are
    `extra="ignore"` rather than absent from the file: the config is shared by
    every service and none of them owns it.
    """

    model_config = ConfigDict(extra="ignore")

    app: AppConfig = Field(default_factory=AppConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def resolve_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Locate the config file: an explicit path, else `MFT_CONFIG`, else default.

    The default is looked up relative to the working directory first and then
    relative to the repository root, because `make run-inference` runs
    `cd services/inference && uvicorn app.main:app` while `make dev` exports
    `MFT_CONFIG=configs/config.yaml` from the root. Without the second lookup
    the same command finds the file or does not depending on how it was
    invoked.
    """
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
    """Read and validate the YAML config.

    Raises:
        ConfigError: If the file is missing, is not a mapping, or fails
            validation. A misconfigured trading service must not start on
            defaults it was not given.
    """
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

    try:
        config = Config.model_validate(document)
    except ValueError as err:
        raise ConfigError(f"invalid config {resolved}: {err}") from err

    logger.debug(
        "loaded config %s: model=%s dry_run=%s instruments=%s context_rows=%d",
        resolved,
        config.inference.model,
        config.inference.dry_run,
        ",".join(config.inference.instruments) or "<none>",
        config.inference.context_rows,
    )
    return config


def configure_logging(level: str) -> None:
    """Send the service's logs to stdout at `level`, in a parseable shape.

    `logging`, never `print`: the Go services read this stream, and a bare
    print in a scheduler tick is indistinguishable from a successful signal.
    """
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
