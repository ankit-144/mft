# MFT Platform — North Star & Architecture

> This is the document `README.md` has referenced since the initial commit.
> It is the contract between the plan and the parallel component work.

## 1. North Star

> Given a local machine and a Zerodha Kite Connect account, run a complete
> medium-frequency trading loop end-to-end: stream ticks, aggregate 1-minute
> candles, build a feature context, score every minute close with a locally
> hosted **TabFM** tabular foundation model, gate each signal through a risk
> engine, and place orders — with every artifact durable in Parquet and
> queryable in DuckDB for backtesting and research.

Two hard constraints shape everything below:

1. **Local-first.** Everything must run on one machine with `make dev`. No
   Kubernetes, no message broker, no managed database. Optional extras must be
   opt-in.
2. **Tabular-native inference.** The model is a *tabular* foundation model
   operating on rows x columns, not an LLM. The feature pipeline is therefore
   a first-class component, not glue code.

## 2. Non-goals (v1)

- No live capital at scale. Paper/sandbox trading first.
- No multi-tenant auth, no RBAC, no billing.
- No order book reconstruction, no HFT/microsecond work. This is *medium*
  frequency: decision cadence is 1 minute.
- No horizontal scaling. Single-node, multi-process.

## 3. Architecture

```
                    ┌──────────────────────────────────────────┐
                    │  Zerodha Kite Connect                    │
                    └───────┬──────────────────────┬───────────┘
                     WS ticks│                      │ REST orders
                            ▼                      │
  ┌──────────────────────────────────┐             │
  │  Service 1 · ingestion (Go)      │             │
  │  WS client · dedup · reconnect   │             │
  │  → fluxKV 1m aggregator (hot)    │             │
  └───────────────┬──────────────────┘             │
                  │ append                          │
                  ▼                                 │
  ┌──────────────────────────────────┐             │
  │  Parquet cold store              │             │
  │  data/ticks/  data/candles/      │             │
  │  data/historical/                │             │
  └───────────────┬──────────────────┘             │
                  │ query                           │
                  ▼                                 │
  ┌──────────────────────────────────┐             │
  │  Service 4 · jobs (Go)           │             │
  │  weekly backfill · rate limited  │             │
  └──────────────────────────────────┘             │
                                                    │
  ┌──────────────────────────────────┐             │
  │  DuckDB (embedded, read+query)   │◄────────────┘
  │  research · backtest · features  │
  └───────────────┬──────────────────┘
                  │ context rows (last N candles + features)
                  ▼
  ┌──────────────────────────────────┐
  │  Service 2 · inference (Python)  │   PULL at each minute close
  │  FastAPI · TabFM (local, CPU)    │
  │  → conviction score in [-1, 1]   │
  └───────────────┬──────────────────┘
                  │ POST /v1/signals
                  ▼
  ┌──────────────────────────────────┐
  │  Service 3 · execution (Go)      │
  │  risk gate · position limits     │
  │  fluxKV (per-process risk state) │
  └───────────────┬──────────────────┘
                  │ PlaceOrder
                  └──────────────────► Kite REST
```

## 4. The central design decision

The current code has an architectural hole: `fluxKV` is an in-memory `map`, but
`execution.Engine` (`services/execution/engine.go:60`) reads it for debouncing
while `ingestion` is a **separate process** that populates it. Cross-process it
is always empty. `services/inference/app/main.py:4` already assumes a "fluxKV
API endpoint (Go)" that was never built.

**Resolution — `fluxKV` is a per-process cache, never a message bus.**

| Concern | Substrate | Rationale |
| :--- | :--- | :--- |
| Durable history | Parquet | Source of truth. Survives restart, cheap to append, DuckDB reads it natively. |
| Cross-service data | **Parquet via DuckDB** | Inference *pulls* at minute close. No new infra, catch-up is free. |
| Intra-process hot state | `fluxKV` TTL map | Risk limits, debounce, positions. Sub-millisecond, per-process, correct. |
| Control plane | HTTP | Inference → execution signals. One way, idempotent, observable. |

Why **pull** and not push for inference: at a 60-second decision cadence,
push latency buys nothing. Pull survives missed minutes (inference restarts and
catches up from the last candle it saw), needs no broker, and keeps the Python
service stateless and horizontally restartable. Push is the correct choice at
tick or second cadence; we are not building that in v1.

This decision is what makes the 10 components independently mergeable.

## 5. TabFM specifics

TabFM is a **Google Research** project (`github.com/google-research/tabfm`).
It is a scikit-learn-compatible zero-shot tabular foundation model using
in-context learning — the historical rows *are* the prompt. No training loop.

| Fact | Value |
| :--- | :--- |
| Source license | Apache-2.0 |
| **Weights license** | **`tabfm-non-commercial-v1.0` — non-commercial, non-production only** |
| Backends | JAX (`jax==0.10.1`, `flax==0.12.7`) or PyTorch (`torch==2.12.1`) |
| Python | `>= 3.11` (we have 3.14; torch 2.12.1 has `cp314` wheels — verified) |
| Hardware | CPU is fine. No GPU required for our table sizes. |
| Practical limits | `max_num_rows=100` context, `max_num_features=500` |
| Task | `TabFMRegressor` → predict next-minute return; `TabFMClassifier` → direction |

### ⚠️ Licensing decision required

The pretrained weights are **non-commercial**. This is a trading platform, and
"non-production" excludes live trading use. Three options, in order of
preference:

1. **Local research only (v1 default).** Accept the restriction. Use TabFM to
   validate the feature pipeline and backtest methodology. Do not trade live
   with its output.
2. **Swap the model behind an interface.** Model lives behind
   `InferenceModel` protocol (see `docs/contracts.md`). A commercially-licensed
   tabular model or a self-trained model drops in without touching the service.
3. **Train your own.** Use the framework, not the weights, and fine-tune on
   your own historical candles.

**Recommendation:** build for (1) now, architect for (2) always. The
`InferenceModel` protocol is a hard requirement of the inference component, not
an optional nicety — it is what makes this reversible.

## 6. Components

Ten components, each its own worktree, branch, and PR. Full interface
definitions live in [`docs/contracts.md`](./docs/contracts.md).

| # | Component | Branch | Owns (exclusive) | Stacks on |
| :-- | :--- | :--- | :--- | :--- |
| C1 | Kite broker connector | `feat/kite-broker` | `core/broker/**` | — |
| C2 | Parquet + DuckDB storage | `feat/storage-analytics` | `core/storage/**`, `core/analytics/**` | — |
| C3 | Feature engineering | `feat/features` | `core/features/**` | C2 |
| C4 | Ingestion pipeline | `feat/ingestion-pipeline` | `services/ingestion/**` | C1, C2 |
| C5 | TabFM model runtime | `feat/tabfm-model` | `services/inference/model/**`, `services/inference/requirements*` | — |
| C6 | Risk engine | `feat/risk-engine` | `services/execution/risk/**`, `services/execution/engine.go`, `core/fluxkv/**` | — |
| C7 | Execution API | `feat/execution-api` | `services/execution/http.go` | C6 |
| C8 | Historical backfill | `feat/jobs-backfill` | `services/jobs/**` | C1 |
| C9 | Observability | `feat/observability` | `core/metrics/**`, `core/log/**`, `observability/**` | — |
| C10 | Inference API + loop | `feat/inference-api` | `services/inference/app/**` | C3, C5 |

Wave 2 (after wave 1 merges): **local-runner** (dev orchestration, runbook) and
**e2e tests** (fake broker, full-loop integration test).

### Conflict avoidance rules

These are the rules that make parallel work mergeable. They are enforced by
review, not by hope:

1. **No component touches a file it does not own.** If you need a change
   outside your ownership column, add it to `docs/contracts.md` and stop.
2. **The config schema is frozen.** `core/config/config.go` already declares
   every key any component needs. Do not add keys.
3. **The Makefile is frozen.** Every target is pre-declared.
4. **Shared types live in `core/contracts/`.** One owner, imported by all. If a
   signature genuinely must change, that is a foundation PR, not a component PR.
5. **`go.work`, `go.mod`/`go.sum` are frozen.** Dependencies are pre-resolved.

## 7. Local development

```sh
make setup      # deps, venv, tabfm weights
make dev        # all services + health checks
make test       # unit tests, all modules
make lint       # gofmt + go vet
make backtest   # replay historical candles through the model
```

Ports: ingestion `:9090`, execution `:8080` + `:9091`, jobs `:9092`,
inference `:8000`. Prometheus metrics on each.

## 8. Risks

| Risk | Mitigation |
| :--- | :--- |
| TabFM weights are non-commercial | `InferenceModel` protocol; default to research-only |
| No GPU | Not needed. CPU inference is sufficient at 1/min. |
| 10 parallel agents on 16 cores | Wave 1 in batches of 4; `make -j4` |
| Merging conflicts in `core/` | Exclusive ownership column + frozen shared files |
| Parquet/DuckDB concurrent read while writing | Partition by date; DuckDB reads immutable files only |
| Broker API rate limits | Token bucket in backfill; WS subscription limits |
