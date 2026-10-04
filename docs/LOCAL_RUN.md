# Run MFT locally and test it before Kite

Applies to master `6e41b7d` (2026-10-04). Run commands from `/home/ankit144/dev/mft` unless stated otherwise. This guide and its helper are tracked alongside the architecture map. Local configs, generated reports and other documentation remain ignored.

## 1. Choose what you want to run

| Goal | Start here | Needs Kite? |
| --- | --- | --- |
| Read the architecture map | Section 2 | No |
| Replay dummy candles through features, model and simulated risk | Section 5 | No |
| Compare algorithms and evaluate a held-out period | Section 6 | No |
| Read Parquet and call the actual inference HTTP API | Sections 4 and 7 | No |
| Check Python → Go execution, risk, duplicate requests and restart | Section 8 | No |
| Import public market candles or your own CSV | Section 9 | No |
| Receive actual NSE prices while keeping execution in paper mode | Section 10 | Yes, market data session |

Start with **synthetic replay → algorithm comparison → component integration → historical NSE data → paper operation**. Dummy data verifies behavior and accounting. Assess a trading edge on representative market data with costs and periods that were excluded from selection.

## 2. Open the architecture map

Open [implementation/guide.html](implementation/guide.html) in a browser; no server or install is needed. It works offline. Source and concept links need internet access.

Optional Linux command:

```sh
xdg-open /home/ankit144/dev/mft/docs/implementation/guide.html
```

If you edit its descriptions, rebuild it with:

```sh
python3 docs/implementation/build_map.py
```

## 3. Install the lightweight runtime

Requirements: **Go 1.25+, Python 3.12+, Git, Make**. On Ubuntu/Debian, `python3-venv` is needed if environment creation fails. No GPU is needed for this guide's commands.

```sh
cd /home/ankit144/dev/mft
go version
python3 --version
make setup
services/inference/venv.sh --dev
```

`make setup` builds the three Go binaries, installs the Python runtime, and creates `configs/config.yaml` only if it is absent. `--dev` adds pytest and Ruff. It does not install the optional checkpoint backend.

If your existing `.venv` points into a component worktree, keep that worktree while using the environment. To make a separate lightweight environment without replacing it:

```sh
VENV=/home/ankit144/dev/mft/.venv/local-runtime services/inference/venv.sh --dev
```

The commands below use `services/inference/.venv/bin/python`. If you choose the separate environment, substitute `.venv/local-runtime/bin/python`; pass `VENV=.venv/local-runtime` to Make commands that use Python. The installer needs an absolute environment path, while those Make targets need a repository-relative path.

Use `heuristic` or the baseline algorithms. Do not run `make tabfm-weights`, install `--model`, or use `--allow-weights` for these exercises. They do not need the large assets or Hugging Face cache.

## 4. Create a separate local demo configuration

This leaves your existing `configs/config.yaml` alone. It isolates candles, cursors and the paper journal under `data/local-demo`.

```sh
services/inference/.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
target = root / "docs/local/config.yaml"
if target.exists():
    raise SystemExit(f"Already exists; edit it if needed: {target}")
config = yaml.safe_load((root / "configs/config.example.yaml").read_text())
data = root / "data/local-demo"
config["storage"]["data_dir"] = str(data)
config["analytics"]["duckdb_path"] = str(data / "mft.duckdb")
config["jobs"]["historical_dir"] = str(data / "historical")
config["broker"]["instruments"] = ["DEMO"]
config["execution"].update(
    paper_trading=True,
    journal_path=str(data / "execution/state.json"),
    api_token="local-demo-only",
)
config["inference"].update(
    model="heuristic", dry_run=True, instruments=["DEMO"],
    cursor_path=str(data / "inference/cursors.json"),
    execution_url="http://127.0.0.1:8080",
)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(config, sort_keys=False))
print(target)
PY

export MFT_CONFIG="/home/ankit144/dev/mft/docs/local/config.yaml"
export MFT_INFERENCE_MODEL=heuristic
export MFT_EXECUTION_URL=http://127.0.0.1:8080
export MFT_EXECUTION_API_TOKEN=local-demo-only
```

Repeat the exports in each new terminal. The token above is just a loopback demo value. `execution.paper_trading: true` uses simulated fills; `inference.dry_run: true` prevents the scheduler from sending signals to execution.

## 5. First check: dummy values, no network

```sh
make backtest ARGS='--synthetic --symbol DEMO --model heuristic --bars 1000 --seed 20261004 --feature-mode incremental --json /home/ankit144/dev/mft/docs/local/backtest.json'
```

Expected: a summary and `docs/local/backtest.json`. It generates deterministic one-minute candles on the NSE weekday session grid and runs the shared feature/model/decision path with offline risk and round-trip accounting. It contacts no broker and starts no HTTP service.

Read the report's assumptions and rejection counts. Zero trades can be a valid result when scores fail the threshold or risk checks reject entries. This replay's risk/accounting implementation is separate from the production Go engine; Section 8 exercises that engine.

To check that incremental feature calculation agrees with rebuilding, the existing tests are listed in Section 8.

## 6. Compare algorithms with costs

```sh
make research ARGS='--source synthetic --symbols DEMO --bars 1000 --seed 20261004 --algorithm baseline_momentum --algorithm baseline_rsi --algorithm baseline_donchian --algorithm baseline_ridge --threshold-grid 0.25,0.40,0.55 --cost-preset zerodha-mis --spread-bps 2 --slippage-bps 2 --output /home/ankit144/dev/mft/docs/local/research.json'
```

This compares momentum, RSI mean reversion, channel breakout and ridge regression. Defaults split time in order: **50% training, 35% validation, 15% held-out test**. Purge/embargo exclude overlapping target horizons at boundaries. Validation selects a candidate; the held-out test evaluates that selection afterward.

The JSON contains configuration, data fingerprints, split ranges, selection and test results. Inspect net returns after costs, drawdown, turnover, trade count and rejected trades; metric fields depend on the evaluator. A short synthetic run is a mechanics check. It cannot establish a profitable strategy.

For useful evidence, repeat on multiple NSE symbols and market periods. Freeze the candidate and assumptions before examining the final test period. Compare with simple baselines and test higher spread/slippage. The current simulator uses close-price fills, so partial fills, queue position, exchange rejects and network failures need separate testing. The Zerodha cost presets are dated estimates; verify current charges before using them in a financial decision.

Available options:

```sh
make research ARGS='--help'
make backtest ARGS='--help'
```

## 7. Run local HTTP services without Kite

First publish the synthetic candles into the same Parquet format the services read:

```sh
services/inference/.venv/bin/python docs/local/prepare_candles.py --source synthetic --symbol DEMO --bars 1000
```

Expected: JSON reporting 1000 rows, partition count and the isolated data directory. The helper reuses the existing generator. It writes closed bars with UTC millisecond timestamps and does not read any large assets.

### Terminal A: paper execution

Apply the exports from Section 4, then:

```sh
MFT_SERVICE_NAME=execution MFT_METRICS_ADDR=127.0.0.1:9091 make run-execution
```

### Terminal B: inference

Apply the exports from Section 4, then:

```sh
make run-inference
```

### Terminal C: inspect and predict

```sh
curl --fail-with-body -H 'Authorization: Bearer local-demo-only' http://127.0.0.1:8080/v1/health
curl --fail-with-body -H 'Authorization: Bearer local-demo-only' http://127.0.0.1:8080/v1/portfolio
curl --fail-with-body http://127.0.0.1:8000/healthz
curl --fail-with-body http://127.0.0.1:8000/readyz
curl --fail-with-body 'http://127.0.0.1:8000/v1/context?symbol=DEMO&rows=20'
curl --fail-with-body -H 'Content-Type: application/json' -d '{"symbol":"DEMO","context_rows":100}' http://127.0.0.1:8000/v1/predict
curl --fail-with-body http://127.0.0.1:8000/metrics
```

Expected: execution health is ready, the portfolio starts empty, inference reports `model_loaded: true` and `dry_run: true`, context contains the ordered 18 features, and prediction returns a finite score plus `as_of`.

**A successful `/readyz` is a process/model check; it does not certify fresh market data or a valid broker session.** The generated candles use historical dates. The scheduler should reject stale inputs; `/v1/predict` still lets you inspect a stored sample. An empty paper portfolio is expected because dry run does not send orders. Use Section 8 to verify actual signal delivery and simulated fills with an injected clock.

Stop the manually started services with Ctrl-C in each service terminal. Run one inference process per cursor path; ownership locks prevent competing schedulers.

Replay these same stored candles without services:

```sh
make backtest ARGS='--symbol DEMO --model heuristic --feature-mode incremental --json /home/ankit144/dev/mft/docs/local/stored-backtest.json'
make research ARGS='--source parquet --data-dir /home/ankit144/dev/mft/data/local-demo --symbols DEMO --output /home/ankit144/dev/mft/docs/local/stored-research.json'
```

## 8. Verify component behavior using existing tests

Run the cross-language integration first:

```sh
make integration
```

It runs three tests using a socket-free Go fixture and an injected clock. It checks:

- Go Parquet output → Python DuckDB reader → identical feature values → a constant fake model → HTTP request handling → Go paper execution/risk/accounting.
- Duplicate submission keeps the same order identity; scheduler cursors suppress repeats.
- Holdings and order identity survive a restart.
- Invalid requests and missing/wrong authentication cannot change the book.

It uses the real HTTP handlers through an in-process transport, **not a real TCP connection, Kite or an exchange**. It is a deterministic component integration check, not a load test.

More checks, from the repository root:

```sh
make test-python
make test
make lint
```

`make test-python` excludes checkpoint tests. `make test` includes the Go race detector; its fake HTTP/WebSocket tests need local sockets. In a restricted sandbox, report socket-related skips separately from passes.

For a smaller Python selection, run from the inference directory:

```sh
cd /home/ankit144/dev/mft/services/inference
PYTHONPATH=. .venv/bin/python -m pytest app/tests/test_backtest.py app/tests/test_backtest_cli.py app/tests/test_research.py app/tests/test_research_cli.py app/tests/test_candles.py -q
cd /home/ankit144/dev/mft
```

Those tests include no-lookahead, incremental/rebuild parity, chronological selection, held-out isolation, fee accounting and bounded data reads. Commands are provided here for you to run; they were not executed while writing this guide.

## 9. Wire a public API or your own historical data

### A. Public API: Binance market data, no API key

Binance documents a public, unauthenticated market-data-only endpoint. Its `GET /api/v3/klines` supports one-minute candles, UTC millisecond timestamps and up to 1000 candles per request. See [market-data access](https://developers.binance.com/en/docs/products/spot/faqs/market_data_only) and [candlestick schema](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market#klinecandlestick-data).

The wiring is:

```text
Public candle API → local helper → canonical Parquet
                                      ↓
                         existing DuckDB reader
                                      ↓
                      features → prediction / research
```

Download a small historical sample during an NSE session's clock hours:

```sh
services/inference/.venv/bin/python docs/local/prepare_candles.py --source binance --symbol ETHUSDT --bars 375 --start 2026-01-05T03:45:00Z --data-dir data/public-demo
```

The helper makes one GET, caps the response at 1 MB, uses a 20-second timeout, drops open bars, validates OHLC, and atomically publishes each partition. If HTTP 403/451 blocks access in your region, use synthetic data or the CSV route. On 429, wait according to the provider's response; the helper does not retry in a loop.

To inspect it using the inference API, stop Terminal B, edit `docs/local/config.yaml`:

```yaml
storage:
  data_dir: /home/ankit144/dev/mft/data/public-demo
inference:
  instruments: [ETHUSDT]
  model: heuristic
  dry_run: true
```

Change those fields **inside the existing config**, keeping its other settings. Restart Terminal B, then:

```sh
curl --fail-with-body -H 'Content-Type: application/json' -d '{"symbol":"ETHUSDT","context_rows":100}' http://127.0.0.1:8000/v1/predict
make research ARGS='--source parquet --data-dir /home/ankit144/dev/mft/data/public-demo --symbols ETHUSDT --cost-preset generic --output /home/ankit144/dev/mft/docs/local/public-research.json'
```

This is a **data integration sample**. ETHUSDT prices are in USDT; the repo assumes NSE session hours and its canonical volume is an integer. The helper stores Binance base-asset volume multiplied by 1,000,000 and rounded to fit that contract. Crypto price, volume, trading hours and cost assumptions differ from NSE shares. Do not relabel this as RELIANCE or interpret its returns/annualized metrics as evidence about your NSE strategy. One session may produce few or no held-out trades.

This importer is a one-time download, not a live stream or replacement for the Go Kite adapter. A generic live source would need implementations of `broker.Streamer`/`OrderClient`/`QuoteClient`, correct symbol/volume mapping and composition changes in `core/fx.go`. The existing scheduler and execution protections should remain in place.

### B. NSE or other vendor CSV: better for evaluating your intended market

Use authorized one-minute OHLCV data for the actual instruments and periods you intend to trade. Convert the vendor's columns to:

```csv
timestamp,open,high,low,close,volume
2026-01-05T03:45:00Z,1400,1402,1399,1401,12000
2026-01-05T03:46:00Z,1401,1403,1400,1402,12500
```

`timestamp` must be minute open time, with an explicit timezone or Unix milliseconds. `09:15:00+05:30` is the same instant as `03:45:00Z`. Use chronological, unique, closed one-minute bars and integer share volume. The optional `symbol` column must match `--symbol`.

The two rows above illustrate the format; actual prediction needs warm-up/context and research needs enough rows for all three splits. Import a bounded sample:

```sh
services/inference/.venv/bin/python docs/local/prepare_candles.py --source csv --csv /absolute/path/reliance-minute.csv --symbol RELIANCE --bars 10000 --data-dir data/nse-research
make research ARGS='--source parquet --data-dir /home/ankit144/dev/mft/data/nse-research --symbols RELIANCE --threshold-grid 0.25,0.40,0.55 --cost-preset zerodha-mis --output /home/ankit144/dev/mft/docs/local/nse-research.json'
```

The helper imports **at most the first `--bars` rows**; it does not load the whole CSV. Reruns append files; the reader deduplicates timestamps. Use a new data directory when changing source or assumptions to avoid mixing datasets. Check missing sessions, split adjustments, timezone and vendor coverage before comparing results.

Canonical layout:

```text
<data_dir>/candles/symbol=RELIANCE/date=YYYY-MM-DD/part-*.parquet
columns: symbol, timestamp(int64 milliseconds), open/high/low/close(float64), volume(int64)
```

The reader also supports sibling `historical/symbol=.../from=.../to=.../candles.parquet` files written by the Kite jobs service; live candles win on duplicate timestamps. Inference honors `jobs.historical_dir`. The offline CLIs default to the historical directory beside their candle dataset, so keep that sibling layout or convert an alternate location to the canonical CSV/Parquet route above.

## 10. Connect Kite market data after the offline checks

For this stage, keep **paper execution** enabled. A market-data connection does not require enabling real orders in this app.

1. Make a separate copy of your demo config at `docs/local/kite-paper.yaml`.
2. Set both `broker.instruments` and `inference.instruments` to actual symbols, e.g. `[RELIANCE, TCS, INFY]`.
3. Use absolute paths under a new `data/kite-paper` directory for storage, historical files, the journal and cursors.
4. Keep `execution.paper_trading: true` and initially `inference.dry_run: true`.
5. Put your own Kite credentials in the ignored `.env` file, using `.env.example` as the field list. The services read environment variables; `.env` is **not automatically loaded**. Load only a file you control:

```sh
set -a
. ./.env
set +a
export MFT_CONFIG="/home/ankit144/dev/mft/docs/local/kite-paper.yaml"
export MFT_INFERENCE_MODEL=heuristic
export MFT_EXECUTION_URL=http://127.0.0.1:8080
export MFT_EXECUTION_API_TOKEN=local-demo-only
```

Start these in separate terminals with those exports, using separate Go metrics ports:

```sh
MFT_SERVICE_NAME=execution MFT_METRICS_ADDR=127.0.0.1:9091 make run-execution
MFT_SERVICE_NAME=ingestion MFT_METRICS_ADDR=127.0.0.1:9090 make run-ingestion
make run-inference
```

To backfill immediately, run a bounded lookback you configured in `jobs.backfill_lookback_days`:

```sh
go run ./services/jobs/cmd/server --run-now
```

`make run-jobs` instead stays running and waits for the configured cron schedule. Received ticks must form **closed** minute candles before inference consumes them. Allow warm-up rows or backfill first; look at `as_of`, scheduler decisions and candle timestamps to verify freshness.

After confirming the feed and integration, setting **only `inference.dry_run: false` while keeping `execution.paper_trading: true`** lets the scheduler send signals to simulated execution. Restart inference after that config change. Market-hour, freshness, exposure and debounce checks still apply. Paper fills do not demonstrate exchange execution quality; the default runtime also has no automatic horizon exit policy, so it is not the same round-trip lifecycle as research.

Optional supervisor: `make dev`, `make status`, `make stop`. It launches all four processes and logs to `.dev/*.log`. **It always uses `configs/config.yaml`**, regardless of your exported `MFT_CONFIG`; use it only after deliberately preparing that config. It does not provide a dummy ingestion mode. Without usable Kite credentials, ingestion fails and the supervisor stops the other processes.

## 11. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `NO_CONTEXT` / no candles | Correct symbol and absolute `storage.data_dir`; run the helper or receive/backfill candles first |
| `BAD_CONTEXT` | At least 60 warm-up bars and enough context; valid OHLC, UTC minute timestamps, correct interval |
| Scheduler reports stale/unchanged | Historical data is intentionally stale; use offline replay or fresh closed bars; repeated input should not submit again |
| Execution rejects on weekend/after hours | Paper mode retains NSE market checks; the integration fixture supplies a trading-time clock |
| 401 on paper health/portfolio/orders | Use the configured bearer token; check `MFT_EXECUTION_API_TOKEN` overrides in both terminals |
| Port already in use | Stop the owning service; Go processes need separate `MFT_METRICS_ADDR` values |
| Cursor already owned | Stop the previous inference process; do not run two processes against one cursor path |
| `make dev` immediately stops | Read `.dev/ingestion.log`; start only execution/inference for the no-Kite demo |
| Venv Python missing | Preserve the linked worktree or create the separate lightweight environment in Section 3 |
| Test skips due to socket permission | Run locally in your normal terminal; skipped socket paths were not verified |
| Positive synthetic P&L | Check mechanics only; repeat with representative NSE history, held-out periods and realistic costs |

For this machine, start with one symbol, 1000–10000 bars and the weight-free models. Leave the bounded worker defaults in place. No 2.4 GB asset load or GPU setup is needed for these checks.
