"""Reading the frozen config.

`core/config/config.go` is the reference implementation and `docs/contracts.md`
§8 the schema. These tests exist to catch the Python reading of that schema
drifting away from the Go one, in either direction: a key the Go config
validates and this one does not, or a default that differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import (
    CONFIG_ENV_VAR,
    Config,
    ConfigError,
    InferenceConfig,
    load_config,
    resolve_path,
)

MINIMAL = """
inference:
  model: heuristic
  instruments: [RELIANCE]
"""


def write(tmp_path: Path, document: object) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    return str(path)


def test_defaults_match_the_go_validators() -> None:
    """Every default here is the one `config.Validate` fills in.

    Read side by side with `core/config/config.go`: an empty `inference`
    block must produce the same service in Python as in Go, or the same config
    file behaves differently depending on which process reads it.
    """
    inference = InferenceConfig()
    assert inference.addr == ":8000"
    assert inference.model == "tabfm"
    assert inference.execution_url == "http://localhost:8080"
    assert inference.context_rows == 100
    assert inference.horizon_bars == 1
    assert inference.order_quantity == 10
    assert inference.dry_run is True
    assert inference.instruments == []


def test_dry_run_defaults_to_true_and_is_not_derived_from_anything(tmp_path: Path) -> None:
    """The one default that must never be argued with.

    A service that starts on an absent, empty or partial config must not be
    the one case that places a real order.
    """
    assert Config().inference.dry_run is True
    assert Config(inference=InferenceConfig(dry_run=False)).inference.dry_run is False
    # An explicit `dry_run: null` in YAML is still a false value, so it is
    # refused rather than silently defaulted to trading.
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, {"inference": {"dry_run": None}}))


def test_the_example_config_parses() -> None:
    """The file `make setup` copies is the file this service must read.

    It is the only config in the repository that is not gitignored, so it is
    the one that gets read by hand most often.
    """
    example = Path(__file__).resolve().parents[4] / "configs" / "config.example.yaml"
    config = load_config(example)

    assert config.app.timezone == "Asia/Kolkata"
    assert config.storage.data_dir == "data"
    assert config.inference.instruments == ["RELIANCE", "TCS", "INFY"]
    assert config.inference.dry_run is True
    assert config.inference.execution_url == "http://localhost:8080"


def test_unrelated_sections_are_ignored_not_refused(tmp_path: Path) -> None:
    """One file is shared by five services and none of them owns it.

    The broker credentials and the jobs schedule are C4's and C8's business;
    this service must read past them without complaint.
    """
    config = load_config(
        write(
            tmp_path,
            {
                "app": {"name": "mft", "env": "dev", "log_level": "info"},
                "broker": {"api_key": "x", "instruments": ["TCS"]},
                "analytics": {"duckdb_path": "data/mft.duckdb", "read_only": True},
                "execution": {"capital": 1000, "max_position_pct": 10.0},
                "jobs": {"schedule": "0 2 * * 6"},
                "inference": {"model": "heuristic", "dry_run": True},
            },
        )
    )
    assert config.inference.model == "heuristic"
    assert not hasattr(config, "broker")


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ({"inference": {"model": "xgboost"}}, "must be one of tabfm, heuristic"),
        ({"inference": {"score_threshold": 1.5}}, r"outside \[0, 1\]"),
        ({"inference": {"score_threshold": -0.1}}, r"outside \[0, 1\]"),
        ({"inference": {"context_rows": 0}}, "context_rows must be positive"),
        ({"inference": {"horizon_bars": -1}}, "horizon_bars must be positive"),
        ({"inference": {"order_quantity": 0}}, "order_quantity must be positive"),
        ({"inference": {"execution_url": "localhost:8080"}}, "not an http"),
        ({"inference": {"instruments": ["  "]}}, "empty symbol"),
        ({"app": {"log_level": "chatty"}}, "log_level"),
        ({"storage": {"partition_by": "symbol"}}, "not supported"),
    ],
)
def test_an_invalid_config_is_refused_rather_than_defaulted(
    tmp_path: Path, document: dict, match: str
) -> None:
    """A misconfigured trading service must not start on defaults.

    The alternative is worse than a crash: a service that silently defaults
    `execution_url` to localhost and `order_quantity` to 10 looks configured
    when it is not.
    """
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, document))


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "nope.yaml")


def test_a_non_mapping_document_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="not a YAML mapping"):
        load_config(path)


def test_broken_yaml_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("inference:\n  model: 'unterminated\n")
    with pytest.raises(ConfigError, match="cannot parse config"):
        load_config(path)


def test_an_empty_file_is_a_valid_default_config(tmp_path: Path) -> None:
    """An empty file is a config with everything defaulted, not a broken one."""
    path = tmp_path / "config.yaml"
    path.write_text("")
    config = load_config(path)
    assert config.inference.dry_run is True
    assert config.inference.model == "tabfm"


def test_instruments_are_normalised_and_deduplicated() -> None:
    """C7 upper-cases what it receives, so the config's spelling is cosmetic."""
    inference = InferenceConfig(instruments=[" reliance ", "TCS", "RELIANCE", "tcs"])
    assert inference.instruments == ["RELIANCE", "TCS"]


def test_the_config_path_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    """`MFT_CONFIG` is what `make dev` exports for every service."""
    path = write(tmp_path, {"inference": {"model": "heuristic", "instruments": ["TCS"]}})
    monkeypatch.setenv(CONFIG_ENV_VAR, path)

    assert resolve_path() == Path(path)
    assert load_config().inference.instruments == ["TCS"]


def test_the_default_path_is_found_from_the_service_directory(monkeypatch) -> None:
    """`make run-inference` cds into `services/inference` before starting.

    A default that only resolved from the repository root would make the same
    command work or not depending on how it was invoked.
    """
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    resolved = resolve_path()
    assert resolved.name == "config.yaml"
    assert resolved.parent.name == "configs"


def test_a_partial_block_keeps_the_other_defaults() -> None:
    """Setting one key must not reset the ones beside it."""
    inference = InferenceConfig(model="heuristic")
    assert inference.context_rows == 100
    assert inference.order_quantity == 10
    assert inference.dry_run is True
