"""The `python -m app.backtest` surface: refusals, artifacts, and what it must not do.

# The three properties this file is about

**It refuses a weighted model by default.** `inference.model` is `tabfm` in the
frozen config and the TabFM checkpoint is ~6.6 GB of non-commercial weights on
a machine whose OOM killer is active. A backtest that followed that key because
nobody passed a flag would be a foot-gun aimed at the person who wanted a number
quickly, so the default is `heuristic` and `--model tabfm` needs
`--allow-weights`. The refusal is asserted here *and* the accept path is
asserted to be reachable, because a gate that cannot be opened is not a gate.

**Every artifact says the input was synthetic.** The summary block, the JSON's
top level and the JSON's `data` section all carry it. The failure mode this
prevents is a reader skimming a pretty equity curve and mistaking it for a
strategy result, so the flag is repeated where a split-up artifact will still
keep it.

**It contacts nothing.** A backtest posts nowhere. `test_a_run_opens_no_socket`
breaks `socket.socket` for the duration of a real CLI run, so a stray HTTP
client — the exact mistake of reaching for `ExecutionClient` because a run
"felt like" it needed an order acknowledged — fails the test rather than
silently depending on a local service being up.
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any

import pytest

from app import backtest as bt
from app.backtest import (
    DEFAULT_MODEL,
    EXIT_OK,
    EXIT_REFUSED,
    WEIGHTED_MODELS,
    build_parser,
    load_execution_limits,
    main,
    parse_day,
    parse_day_end,
    plan_model,
)

#: A config with no credentials and a real capital figure, which is the only
#: section of the frozen schema the harness reads besides `inference`.
CONFIG_YAML = """
app:
  timezone: Asia/Kolkata
inference:
  model: tabfm
  context_rows: 100
  horizon_bars: 1
  score_threshold: 0.55
  order_quantity: 10
  instruments: [RELIANCE, TCS]
  dry_run: true
execution:
  capital: 1000000
  debounce_ttl_seconds: 300
  max_position_pct: 10.0
  max_open_positions: 10
  max_drawdown_pct: 5.0
  daily_loss_limit: 25000
  max_order_quantity: 500
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A config file with `inference.model: tabfm`, as the frozen one has."""
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML)
    return path


@pytest.fixture(autouse=True)
def captured_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Leave the CLI's own logging setup alone so pytest can capture it.

    `configure_logging` removes every handler on the root logger, which is right
    for a service and wrong for a test: it throws away the handler `caplog`
    installed, and the CLI calls it on every run. Stubbing it out means these
    tests assert on *what the module logs* rather than on *how it configures
    logging*, which is the more durable of the two things to test.
    """
    monkeypatch.setattr(bt, "configure_logging", lambda level: None)
    caplog.set_level(logging.DEBUG)


def run_cli(*args: str) -> int:
    """Invoke the CLI the way `make backtest` does."""
    return main(list(args))


def test_parquet_cli_sets_reader_budget_from_explicit_replay_limit(config_file, monkeypatch) -> None:
    import app.backtest_cli as cli

    budgets = []

    def refuse_store(root, *, max_rows):
        budgets.append(max_rows)
        raise cli.BacktestError("fixture stopped before reading assets")

    monkeypatch.setattr(cli, "DuckDBCandleStore", refuse_store)
    assert run_cli("--config", str(config_file), "--max-bars", "100123") == 2
    assert budgets == [100124]


def flat(messages: Any) -> str:
    """Collapse a report to one whitespace-normalised line.

    The summary is a reflowed block, so a phrase can straddle a line break. A
    test that searched for it verbatim would be testing the line width rather
    than the wording.
    """
    return " ".join("\n".join(messages).split())


# --------------------------------------------------------------------------
# The weights guard
# --------------------------------------------------------------------------


def test_the_default_model_is_heuristic_not_the_configured_one(config_file: Path) -> None:
    """`inference.model: tabfm` must not pull 6.6 GB into a research run."""
    config = bt.load_config(config_file)
    assert config.inference.model == "tabfm"
    args = build_parser().parse_args([])
    assert args.model == DEFAULT_MODEL == "heuristic"
    assert plan_model(args, config) == "heuristic"


@pytest.mark.parametrize("model", sorted(WEIGHTED_MODELS))
def test_a_weighted_model_is_refused_without_the_flag(model: str, config_file: Path) -> None:
    """`--model tabfm` alone is a refusal, and the message says what to do."""
    config = bt.load_config(config_file)
    args = build_parser().parse_args(["--model", model])
    with pytest.raises(bt.BacktestError) as caught:
        plan_model(args, config)
    message = " ".join(str(caught.value).split())
    assert "refusing" in message
    assert "6.6 GB" in message
    assert "non-commercial" in message
    assert "--allow-weights" in message and "--model heuristic" in message


def test_the_refusal_happens_before_any_model_is_built(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is a plan, not a post-hoc check.

    `conftest`'s autouse `no_weights` fixture refuses every model but the
    heuristic by patching `model.base.create_model`, so if this run reached the
    factory at all it would fail loudly rather than quietly loading 6.6 GB. The
    exit code is the assertion that the guard fired first.
    """
    assert run_cli(
        "--config", str(config_file), "--model", "tabfm", "--synthetic", "--bars", "150"
    ) == EXIT_REFUSED


def test_the_cli_will_open_the_gate_when_the_flag_is_given(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`--allow-weights` gets past the guard, and the warning is loud.

    The accept path is never exercised to completion here — that would load the
    checkpoint — so this asserts the plan and the banner, and leaves the loading
    to a human on a machine with free memory.
    """
    config = bt.load_config(config_file)
    args = build_parser().parse_args(["--model", "tabfm", "--allow-weights"])
    with caplog.at_level(logging.WARNING):
        assert plan_model(args, config) == "tabfm"
    assert any("non-commercial" in record.getMessage() for record in caplog.records)
    assert any("6.6 GB" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------
# The execution section
# --------------------------------------------------------------------------


def test_execution_limits_are_read_from_the_frozen_schema(config_file: Path) -> None:
    limits = load_execution_limits(config_file)
    assert limits.capital == 1_000_000.0
    assert limits.debounce_ttl_seconds == 300
    assert limits.max_position_pct == 10.0
    assert limits.max_open_positions == 10
    assert limits.max_drawdown_pct == 5.0
    assert limits.daily_loss_limit == 25_000.0
    assert limits.max_order_quantity == 500


def test_the_repository_config_is_usable_as_shipped() -> None:
    """`make backtest` has to work against the config in the repository."""
    resolved = bt.resolve_path()
    limits = load_execution_limits()
    assert limits.capital > 0.0, f"{resolved} must set execution.capital"
    assert load_config_instruments()


def load_config_instruments() -> list[str]:
    from app.config import load_config

    return load_config().inference.instruments


def test_a_missing_capital_is_refused_with_the_key_named(tmp_path: Path) -> None:
    """`core/config/config.go` does not default `capital`, so neither do we.

    With capital 0 every order fails `RISK_MAX_POSITION` against zero equity,
    and a run that quietly reported a hundred percent loss would look like a
    strategy result.
    """
    path = tmp_path / "no-capital.yaml"
    path.write_text("app:\n  timezone: Asia/Kolkata\nexecution: {}\n")
    with pytest.raises(bt.BacktestError, match="execution.capital"):
        load_execution_limits(path)


@pytest.mark.parametrize(
    "body, message",
    [
        ("execution:\n  capital: -1\n", "execution.capital"),
        ("execution:\n  capital: 100\n  max_position_pct: 0\n", "max_position_pct"),
        ("execution:\n  capital: 100\n  market_holidays: [not-a-date]\n", "market_holidays"),
        ("execution:\n  capital: 100\n  debounce_ttl_seconds: -5\n", "debounce_ttl_seconds"),
        ("execution:\n  capital: 100\n  max_open_positions: -1\n", "max_open_positions"),
        ("- a list\n", "not a YAML mapping"),
        ("execution: 7\n", "not a YAML mapping"),
    ],
)
def test_a_bad_execution_section_is_refused(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(body)
    with pytest.raises(bt.BacktestError, match=message):
        load_execution_limits(path)


def test_a_missing_config_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(bt.BacktestError, match="cannot read config"):
        load_execution_limits(tmp_path / "not-there.yaml")


# --------------------------------------------------------------------------
# Date window
# --------------------------------------------------------------------------


def test_dates_are_parsed_in_ist_not_utc() -> None:
    """`--from 2026-01-01` means 09:15 IST on the 1st, and the zone shows it.

    A UTC reading of the same string would start the run six hours and thirty
    minutes early, which on a 09:15-15:30 session is most of a session.
    """
    start = parse_day("2026-01-01", "from")
    assert start is not None
    assert start.utcoffset().total_seconds() == 5.5 * 3600
    assert start.astimezone(bt.timezone.utc).isoformat() == "2025-12-31T18:30:00+00:00"
    assert parse_day(None, "from") is None


def test_the_end_date_covers_the_whole_day() -> None:
    """`--to 2026-06-30` includes 15:30 IST on the 30th, not 00:00 on it."""
    end = parse_day_end("2026-06-30")
    assert end is not None
    assert end.hour == 23 and end.minute == 59 and end.microsecond == 999_999
    assert parse_day_end(None) is None


def test_a_malformed_date_is_refused() -> None:
    with pytest.raises(bt.BacktestError, match="YYYY-MM-DD"):
        parse_day("30-06-2026", "to")


def test_window_candles_is_inclusive_at_both_ends() -> None:
    from app.candles import Candle

    candles = [
        Candle("T", bt.datetime(2026, 1, 5, 4, minute, tzinfo=bt.timezone.utc), 1, 1, 1, 1, 1)
        for minute in range(0, 5)
    ]
    kept = bt.window_candles(candles, parse_day("2026-01-05", "from"), parse_day_end("2026-01-05"))
    assert len(kept) == 5  # the whole session is inside the day
    empty = bt.window_candles(candles, parse_day("2026-02-01", "from"), None)
    assert empty == []


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def test_a_synthetic_run_marks_itself_in_the_summary_and_the_json(
    config_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The honesty requirement, end to end through the real CLI."""
    out = tmp_path / "report.json"
    with caplog.at_level(logging.INFO):
        code = run_cli(
            "--config", str(config_file),
            "--synthetic", "--bars", "400", "--symbol", "RELIANCE",
            "--feature-mode", "incremental", "--json", str(out),
        )
    assert code == EXIT_OK

    summary = flat(record.getMessage() for record in caplog.records)
    assert "MFT BACKTEST — SYNTHETIC INPUT. THIS IS NOT A STRATEGY RESULT." in summary
    assert "SYNTHETIC INPUT." in summary
    assert "NOT evidence of an edge" in summary
    assert "app/synth.py" in summary
    assert "geometric-Brownian-motion series" in summary
    assert "Every charge rate in this run is an ASSUMPTION" in summary
    assert "Re-verify every rate against your broker's current contract" in summary
    assert "WHAT THIS RUN DOES NOT TELL YOU" in summary
    assert "Out-of-sample performance" in summary
    assert "forecasting a random number generator" in summary

    document = json.loads(out.read_text())
    assert document["schema"] == "mft.backtest.v1"
    assert document["synthetic"] is True
    assert document["data"]["synthetic"] is True
    assert document["data"]["source"] == "synthetic"
    assert document["data"]["seed"] == 20260929
    assert "SYNTHETIC INPUT." in document["disclaimer"]
    assert "ASSUMPTION" in document["charge_disclaimer"]
    assert document["charge_model"]["assumptions"]["stt_buy_pct"] == 0.1
    assert "Position model: one order is one round trip" in document["position_model"]
    assert document["risk_drift_notes"], "the drift notes travel with the artifact"


def test_the_json_carries_everything_the_report_promises(config_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    assert run_cli(
        "--config", str(config_file), "--synthetic", "--bars", "400", "--symbol", "RELIANCE",
        "--feature-mode", "incremental", "--json", str(out),
    ) == EXIT_OK
    document = json.loads(out.read_text())

    for key in (
        "metrics", "per_symbol", "trades", "rejections", "outcomes", "timing",
        "equity_curve", "walk_forward", "config", "execution_limits", "data",
    ):
        assert key in document, f"{key} is promised by the report and missing from the artifact"

    metrics = document["metrics"]
    for key in (
        "trades", "hit_rate", "net_return_pct", "gross_return_pct", "mean_trade_return_pct",
        "sharpe_annualised", "sharpe_per_bar", "bars_per_year", "max_drawdown_pct",
        "turnover_x_capital", "traded_notional", "exposure_pct", "charges_total",
    ):
        assert key in metrics, key
    assert metrics["trades"] == len(document["trades"])
    assert metrics["trades"] > 0, "400 bars of a random walk should produce some trades"
    assert metrics["charges_total"] > 0.0, "a run with no charges is not a backtest"
    assert metrics["bars_per_year"] == 94_500
    assert document["walk_forward"]["complexity"] == "O(n) per symbol"
    assert document["walk_forward"]["first_decision_index"] == 120
    # Every fill became a trade, and the artifact says so in both numbers so a
    # reader can check the invariant instead of trusting it.
    assert document["walk_forward"]["fills"] == document["walk_forward"]["trades"]
    assert document["walk_forward"]["fills"] == len(document["trades"])
    assert document["config"]["model_selector"] == "heuristic"
    assert document["config"]["model"] == "heuristic-v1"
    assert document["timing"]["model_ms_per_bar"] > 0.0

    row = document["per_symbol"][0]
    assert row["symbol"] == "RELIANCE"
    assert row["trades"] == metrics["trades"]

    trade = document["trades"][0]
    assert trade["idempotency_key"].count(":") == 2
    assert trade["side"] in ("BUY", "SELL")
    assert trade["net_pnl"] == pytest.approx(
        trade["gross_pnl"] - trade["charges"], abs=1e-3
    )
    assert trade["entry_as_of"].endswith("Z")


def test_a_json_run_is_byte_identical_across_two_runs(
    config_file: Path, tmp_path: Path
) -> None:
    """Same seed, same artifact, apart from the timestamp and the timings.

    A reported number has to be regenerable, and the only fields allowed to
    differ are the ones that describe the run rather than the result.
    """
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    for out in (first, second):
        assert run_cli(
            "--config", str(config_file), "--synthetic", "--bars", "400", "--symbol", "RELIANCE",
            "--feature-mode", "incremental", "--json", str(out),
        ) == EXIT_OK

    a = json.loads(first.read_text())
    b = json.loads(second.read_text())
    for volatile in ("generated_at", "timing"):
        a.pop(volatile)
        b.pop(volatile)
    assert a == b


def test_the_default_run_refuses_an_empty_store_rather_than_reporting_zeroes(
    config_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`data/candles/` is empty, so the default path has to say so and point at
    `--synthetic` rather than print a table of zeroes."""
    with caplog.at_level(logging.ERROR):
        code = run_cli("--config", str(config_file))
    assert code == EXIT_REFUSED
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "no candles" in message
    assert "--synthetic" in message


def test_an_unwritable_json_path_is_refused(
    config_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The artifact is part of the deliverable, so failing to write it is a refusal.

    A path whose parent is a regular file is the portable way to make the write
    fail: `mkdir` raises `NotADirectoryError`, which is an `OSError`, and the
    message names the path.
    """
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    with caplog.at_level(logging.ERROR):
        code = run_cli(
            "--config", str(config_file), "--synthetic", "--bars", "150", "--symbol", "RELIANCE",
            "--feature-mode", "incremental", "--json", str(blocker / "report.json"),
        )
    assert code == EXIT_REFUSED
    assert any("cannot write" in r.getMessage() for r in caplog.records)


def test_a_date_window_narrowed_to_nothing_is_refused(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        code = run_cli(
            "--config", str(config_file), "--synthetic", "--bars", "200", "--symbol", "RELIANCE",
            "--from", "2030-01-01", "--to", "2030-01-02",
        )
    assert code == EXIT_REFUSED
    assert any("no bars between" in r.getMessage() for r in caplog.records)


def test_a_reversed_window_is_refused(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        code = run_cli(
            "--config", str(config_file), "--from", "2026-06-30", "--to", "2026-01-01",
        )
    assert code == EXIT_REFUSED
    assert any("is after" in r.getMessage() for r in caplog.records)


def test_an_over_long_series_is_refused_and_the_message_says_what_to_do(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`--max-bars` is a wall, and a refusal the user cannot act on is a bug."""
    with caplog.at_level(logging.ERROR):
        code = run_cli(
            "--config", str(config_file), "--synthetic", "--bars", "200", "--symbol", "RELIANCE",
            "--max-bars", "199",
        )
    assert code == EXIT_REFUSED
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "over the --max-bars limit" in message
    assert "O(n^2)" in message
    assert "--feature-mode incremental" in message


def test_every_configured_instrument_runs_by_default(config_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "all.json"
    assert run_cli(
        "--config", str(config_file), "--synthetic", "--bars", "300",
        "--feature-mode", "incremental", "--json", str(out),
    ) == EXIT_OK
    document = json.loads(out.read_text())
    assert document["data"]["symbols"] == ["RELIANCE", "TCS"]
    assert [row["symbol"] for row in document["per_symbol"]] == ["RELIANCE", "TCS"]


# --------------------------------------------------------------------------
# What it must not do
# --------------------------------------------------------------------------


def test_a_run_opens_no_network_socket(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backtest posts nowhere. If this needs patching to pass, it has regressed.

    `AF_INET` and `AF_INET6` are refused, and so is `create_connection`.
    `AF_UNIX` is deliberately allowed: `asyncio` opens a self-pipe in that family
    to wake its own selector, which is the event loop's business and not a
    network call. What is being tested is that nothing reaches for the execution
    service, or for anything else over a network, which is the exact mistake of
    submitting a signal from a run that has no order to submit.
    """
    network = {socket.AF_INET, socket.AF_INET6}

    def refuse(family: Any = None, *args: Any, **kwargs: Any) -> Any:
        if family in network:
            raise AssertionError(
                "the backtest opened a network socket. It is a backtest: there is "
                "nothing to post to."
            )
        return original(family, *args, **kwargs)

    original = socket.socket
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: pytest.fail("the backtest opened a connection"),
    )
    assert run_cli(
        "--config", str(config_file), "--synthetic", "--bars", "300", "--symbol", "RELIANCE",
        "--feature-mode", "incremental",
    ) == EXIT_OK


def test_the_module_holds_no_http_client() -> None:
    """No `httpx`, no `ExecutionClient`, no sender of any kind.

    A static check rather than a behavioural one, so the reason survives: the
    temptation to post a signal from a backtest is exactly the mistake of
    forgetting that a backtest places no order.
    """
    import app.backtest as module

    source = Path(module.__file__).read_text()
    for forbidden in ("httpx", "ExecutionClient", "urllib.request", "requests", "POST /v1/signals"):
        assert forbidden not in source, f"{forbidden} must not appear in the backtest"
    assert not hasattr(module, "sender")


def test_the_report_states_the_cost_model_and_the_position_model(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A reader must not have to guess what a "trade" is or what it cost.

    Both caveats are asserted on the terminal output, because that is what a
    person actually reads. The JSON carries them too; this is the check that the
    human path is covered.
    """
    with caplog.at_level(logging.INFO):
        assert run_cli(
            "--config", str(config_file), "--synthetic", "--bars", "400", "--symbol", "RELIANCE",
            "--feature-mode", "incremental",
        ) == EXIT_OK
    summary = flat(record.getMessage() for record in caplog.records)
    assert "Position model: one order is one round trip over horizon bars" in summary
    assert "NOT delivery (CNC) buy-and-hold" in summary
    assert "no margin haircut and no borrow cost" in summary
    assert "drift note" in summary
    assert "Lot size is not in the frozen config" in summary
    assert "COST OF THE MODEL ITSELF" in summary
    assert "ms/bar" in summary
    assert "a 1-minute cadence allows 60000 ms/bar" in summary
    assert "PER-TRADE TABLE" in summary
    assert "RISK GATE" in summary
    # The only reason code this run produced is the contract's, spelled the
    # contract's way. A code the execution service would not emit would be worse
    # than no code at all.
    assert "RISK_DEBOUNCED" in summary
    assert "check 5" in summary
