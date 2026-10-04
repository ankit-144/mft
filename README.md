# MFT platform

Go ingestion, execution and historical jobs; Python inference and strategy
research; shared Parquet data. Defaults are **paper execution**, **inference dry
run** and a lightweight heuristic model.

## Components

| Path | Responsibility |
| --- | --- |
| `core/` | Config/contracts, Kite adapter, typed storage/features, cache, logging and metrics |
| `services/ingestion/` | Bounded tick processing, volume deltas and closed-candle publication |
| `services/execution/` | Durable order claims, risk reservations, reconciliation, fill accounting and HTTP |
| `services/jobs/` | Paced/resumable historical backfill |
| `services/inference/app/` | Canonical DuckDB reader, features, serialized model worker and minute scheduler |
| `services/inference/model/` | Predictor contract, heuristic and mathematical baseline registry |
| `services/inference/app/research/` | Independent strategies/evaluators, chronological selection and cost-aware simulation |

The Go modules are joined by `go.work`. Go reads Parquet through typed readers;
Python uses DuckDB. Both runtimes use the same ordered 18-feature schema.

## Architecture and local run guides

Open [the interactive architecture map](docs/implementation/guide.html) locally
in a browser after cloning. It explains components, responsibilities, contracts
and data flows through a searchable, clickable tree. The page works offline;
optional source links open the documented commit on GitHub.

See [the local run guide](docs/LOCAL_RUN.md) for setup, synthetic data, Parquet
replay, component integration tests, public candle imports and Kite paper mode.
[Map sources and rebuild instructions](docs/implementation/README.md) are included.

## Local setup

Requires Go 1.25 and Python 3.12 or later. From the repository root:

```sh
make setup
services/inference/venv.sh --dev
```

`make setup` builds the Go services, creates the Python environment and copies
the example config when no local config exists. Relative data and journal paths
resolve from the project root when the config is inside `configs/`. The runtime
install excludes torch and checkpoint packages; optional model dependencies are
in `services/inference/requirements-model.txt`.

Start individual services or the local process supervisor with:

```sh
make run-execution
make run-inference
make dev
make stop
```

`make dev` supervises its own subprocesses and `make stop` stops only that
recorded process group. Real ingestion and historical backfill require a usable
broker session. Services consume `MFT_CONFIG`; `MFT_INFERENCE_MODEL`,
`MFT_EXECUTION_URL`, `MFT_EXECUTION_ADDR` and `MFT_EXECUTION_API_TOKEN` override
relevant settings. Broker credential variables are listed in `.env.example`.

## Research

```sh
make research ARGS='--source synthetic --symbols RELIANCE --bars 1000'
make backtest ARGS='--synthetic --model heuristic --bars 200 --feature-mode incremental'
make benchmarks
```

Run `python -m app.research --help` from `services/inference` with the project
virtual environment for candidate, evaluator, cost and output options.
Algorithms, strategies and selection metrics are registered independently.
Validation chooses the candidate; held-out timestamps evaluate the selected
candidate afterward.

Synthetic results check accounting and integration; they do not establish
market profitability. Research assumes close-price fills with configurable
costs, while production execution uses broker fills. The default live graph has
no automatic horizon exit policy; callers can inject `ExitPolicy` with stable
keys and metadata.

## Services and containers

Execution defaults to `127.0.0.1:8080`; inference defaults to
`127.0.0.1:8000`. Live execution requires bearer authentication. Paper mode also
checks a configured token. `/v1/health` reports execution readiness; inference
exposes `/healthz`, `/readyz` and `/metrics`.

```sh
make docker-build
make docker-up
make docker-down
```

Compose uses the example config, internal service URLs and a shared data volume.
Host ports are loopback-bound. The Docker context excludes worktrees,
environments and data.
