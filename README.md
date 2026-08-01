# MFT Platform

Medium Frequency Trading (MFT) platform V3. Go services for low-latency
execution, Python (TabFM) inference, Parquet cold storage, and DuckDB
analytics. See [Plan.md](./Plan.md) for the full architecture spec.

## Structure

```
├── core/                    # Shared generic code (own Go module)
│   ├── broker/              # Broker connector interfaces (Zerodha Kite Connect)
│   ├── config/              # YAML config loading + validation
│   ├── fluxkv/              # In-memory TTL cache + 1-min candle aggregation
│   ├── log/                 # Shared zap logger
│   ├── metrics/             # Prometheus registry, handler, server
│   ├── storage/             # Parquet tick storage
│   └── fx.go                # Core FX module wiring
├── services/
│   ├── ingestion/           # Service 1: tick ingestion pipeline (own module)
│   ├── execution/           # Service 3: execution & risk engine (own module)
│   ├── jobs/                # Service 4: background jobs & backfill (own module)
│   └── inference/           # Service 2: Python/TabFM inference node
├── configs/                 # YAML configs
├── data/                    # Local data (parquet files, gitignored)
├── go.work                  # Go workspace tying modules together
└── Makefile                 # Build/run targets
```

Each Go microservice is its own Go module wired together by `go.work`. All
services use [Uber FX](https://github.com/uber-go/fx) for dependency
injection and [prometheus/client_golang](https://github.com/prometheus/client_golang)
for metrics. Shared code lives in `core/`.

## Getting Started

```sh
# 1. Resolve dependencies (from repo root)
make tidy

# 2. Build all modules
make build

# 3. Run a service
make run-ingestion
make run-execution
make run-jobs
```

Each service exposes a metrics endpoint on `:9090` by default
(override `metrics.addr` in config).

## Configuration

Copy `configs/config.example.yaml` to `configs/config.yaml` and fill in
broker credentials. Environment variables from `.env.example` are supported
by the broker connectors.
