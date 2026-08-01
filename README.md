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

## Testing

```sh
# Run unit tests across all Go modules (core + services)
make test

# Vet all modules
make vet
```

Test helpers live in `core/testutil` (mock broker streamer/client, isolated
Prometheus registry, metric value assertions). Coverage spans:

- `core/fluxkv` — TTL get/set/delete, minute-candle aggregation & rollover
- `core/config` — YAML load, defaults, error paths
- `core/storage` — Parquet flush + read-back
- `services/ingestion` — tick processing updates candles + queues storage
- `services/execution` — order placement, debounce rejection/expiry, broker errors
- `services/jobs` — backfill worker runs and bumps its metric

## Docker

The repo ships Dockerfiles per service plus a `docker-compose.yml` for the
full stack. Build context is the repo root, so the Go workspace and shared
`core` module are available to each image.

```sh
# Build all images
make docker-build

# Start the stack in the background
make docker-up

# Tail logs
make docker-logs

# Stop and remove containers (data volume persists)
make docker-down
```

Endpoints after `docker-up`:

| Service    | HTTP            | Metrics         |
| :--------- | :-------------- | :-------------- |
| ingestion  | —               | `:9090`         |
| execution  | `:8080` (POST /v1/orders) | `:9091` |
| jobs       | —               | `:9092`         |
| inference  | `:8000` (healthz, /v1/predict) | —        |

Images use the example config (empty credentials) by default — mount a
real config or set `MFT_CONFIG` via env to connect the broker. Parquet data
persists in the `mft_data` volume.

## Configuration

Copy `configs/config.example.yaml` to `configs/config.yaml` and fill in
broker credentials. Environment variables from `.env.example` are supported
by the broker connectors.
