from __future__ import annotations

import json

from app.research.__main__ import main, resolve_data_dir
from app.research.adapters import register_strategy


def test_synthetic_cli_writes_candidate_grid_as_reproducible_json(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    code = main(
        [
            "--source", "synthetic",
            "--symbols", "TEST",
            "--bars", "500",
            "--seed", "42",
            "--algorithm", "baseline_momentum",
            "--threshold-grid", "0,1",
            "--cost-preset", "zerodha-mis",
            "--spread-bps", "1.25",
            "--slippage-bps", "3.5",
            "--strategy-options", '{"threshold":0.1,"long_only":true}',
            "--output", str(artifact),
        ]
    )
    document = json.loads(artifact.read_text())
    assert code == 0
    assert document["source"] == "synthetic"
    assert len(document["candidates"]) == 2
    assert document["selection"]["parameters"]["threshold"] in {0.0, 1.0}
    assert document["configuration_fingerprint"]
    config = document["config"]
    assert config["costs"]["schedule"] == "zerodha_equity_mis_nse"
    assert config["costs"]["effective_date"] == "2026-10-04"
    assert config["costs"]["source_url"] == "https://zerodha.com/charges"
    assert config["costs"]["spread_bps"] == 1.25
    assert config["costs"]["slippage_bps"] == 3.5
    assert config["holding_policy"] == "intraday"
    assert config["strategy"] == "threshold"
    assert config["strategy_options"] == [["long_only", True], ["threshold", 0.1]]


def test_cnc_cli_selects_delivery_policy_and_allows_explicit_override(tmp_path) -> None:
    artifact = tmp_path / "cnc.json"
    code = main(
        [
            "--symbols", "TEST", "--bars", "500", "--algorithm", "baseline_momentum",
            "--cost-preset", "zerodha-cnc", "--spread-bps", "0", "--slippage-bps", "0",
            "--output", str(artifact),
        ]
    )
    assert code == 0
    document = json.loads(artifact.read_text())
    assert document["config"]["costs"]["schedule"] == "zerodha_equity_cnc_nse"
    assert document["config"]["costs"]["effective_date"] == "2026-10-04"
    assert document["config"]["costs"]["brokerage_bps"] == 0.0
    assert document["config"]["holding_policy"] == "delivery"

    override = tmp_path / "cnc-intraday.json"
    code = main(
        [
            "--symbols", "TEST", "--bars", "500", "--algorithm", "baseline_momentum",
            "--cost-preset", "zerodha-cnc", "--holding-policy", "intraday",
            "--output", str(override),
        ]
    )
    assert code == 0
    assert json.loads(override.read_text())["config"]["holding_policy"] == "intraday"


def test_cli_uses_registered_strategy_and_json_keyword_options(tmp_path) -> None:
    class FlatStrategy:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def side(self, score: float) -> int:
            return 0

    register_strategy("cli_flat_test", FlatStrategy)
    artifact = tmp_path / "strategy.json"
    code = main(
        [
            "--symbols", "TEST", "--bars", "500", "--algorithm", "baseline_momentum",
            "--strategy", "cli_flat_test", "--strategy-options", '{"marker":7}',
            "--output", str(artifact),
        ]
    )
    assert code == 0
    config = json.loads(artifact.read_text())["config"]
    assert config["strategy"] == "cli_flat_test"
    assert config["strategy_options"] == [["marker", 7]]


def test_synthetic_cli_refuses_unbounded_bar_count(capsys) -> None:
    code = main(["--symbols", "TEST", "--bars", "100001"])
    assert code == 2
    assert "between 1 and" in capsys.readouterr().err


def test_cli_rejects_invalid_or_non_object_strategy_options(capsys) -> None:
    for raw in ("{", "[]"):
        code = main(["--symbols", "TEST", "--strategy-options", raw])
        assert code == 2
        assert "--strategy-options" in capsys.readouterr().err


def test_cli_rejects_nonfinite_strategy_option_from_experiment_config(capsys) -> None:
    code = main(["--symbols", "TEST", "--strategy-options", '{"threshold":NaN}'])
    assert code == 2
    assert "finite" in capsys.readouterr().err


def test_relative_store_directory_uses_repository_root() -> None:
    from pathlib import Path

    expected = Path(__file__).resolve().parents[4] / "data"
    assert resolve_data_dir("data") == expected


def test_parquet_cli_sets_reader_budget_from_explicit_experiment_limit(monkeypatch, capsys) -> None:
    import app.research.__main__ as cli

    budgets = []

    def refuse_store(root, *, max_rows):
        budgets.append(max_rows)
        raise RuntimeError("fixture stopped before reading assets")

    monkeypatch.setattr(cli, "DuckDBCandleStore", refuse_store)
    assert main(["--source", "parquet", "--symbols", "TEST", "--max-bars-per-symbol", "100123"]) == 2
    assert budgets == [100124]
    assert "fixture stopped" in capsys.readouterr().err
