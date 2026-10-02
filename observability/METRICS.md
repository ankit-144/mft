# MFT metrics catalogue

Every metric this platform exposes carries the **`mft_`** prefix. This document
is the contract for that: if a metric is not listed here, it does not belong on
a dashboard, and if it is listed here, its name, type and labels are fixed.

The prefix is what lets one Prometheus, one Grafana and one alert set cover four
services without two of them shadowing each other. It is enforced in code by
`core/metrics`: every constructor in `core/metrics/naming.go` calls
`metrics.ValidateName`, and a violation **panics at registration**.

## The rules

1. **Prefix.** `mft_` and nothing else. `mft_execution_orders_placed_total`, not
   `execution_orders_placed_total`.
2. **Subsystem, then measure.** `mft_<subsystem>_<noun>_<verb>`, so a metric
   sorts next to its siblings and the subsystem is obvious in an alert.
3. **Units in the name.** Latency is always `_duration_seconds`; money is
   `_inr`; percentage of equity is `_pct`; a bare `_pct` is a value in 0–100,
   never a 0–1 ratio. `core/metrics.DurationHistogram` rejects a latency metric
   that does not end in `_seconds`.
4. **Counters end in `_total`.** They only ever increase and are only meaningful
   under `rate()` or `increase()`.
5. **A Help string is mandatory.** It is the only documentation the metric will
   ever have.
6. **Bounded labels only.** Symbol, side, reason, status, method, route — all
   drawn from a small fixed set. Never a correlation id, an order id, a
   timestamp, or a free-text error as a label. For one-off correlations use
   `metrics.ObserveDuration(hist, start, correlationID)`, which attaches the id
   as a Prometheus *exemplar* rather than a label.

## Registering metrics

The helpers are opt-in — a component that keeps using `promauto.With(reg)` is
fine, the name is still subject to review. When you do adopt them:

```go
import "github.com/mft/core/metrics"

placed := metrics.CounterVec(reg, "mft_execution_orders_placed_total",
    "Orders accepted by the risk gate and sent to the broker.", "side", "symbol")

latency := metrics.DurationHistogramVec(reg, "mft_execution_order_duration_seconds",
    "Time from the risk gate to a broker order id.", "side", "symbol")

defer func() {
    metrics.ObserveDuration(latency.WithLabelValues(side, symbol), time.Now(), correlationID)
}()
```

`metrics.RegisterBuildInfo(reg, metrics.BuildInfo{Service: "execution", Env: cfg.App.Env})`
adds the build panel for your service; `metrics.DefaultChecks().Add(...)` makes
`/readyz` aware of a dependency.

## Platform metrics — provided by `core/metrics`

| Metric | Type | Labels | Meaning |
| :--- | :--- | :--- | :--- |
| `mft_build_info` | gauge | `service`, `version`, `commit`, `branch`, `goversion`, `env` (const) | Build metadata. Always 1. |
| `mft_uptime_seconds` | gauge func | — | Seconds since the process started. |
| `go_*`, `process_*` | mixed | — | Standard Go and process collectors. |
| `mft_http_requests_total` | counter | `method`, `route`, `status` | Requests through `log.Middleware`. Optional, per service. |
| `mft_http_request_duration_seconds` | histogram | `method`, `route` | Request latency. Optional, per service. |

## v1 signals — the four panels that matter

Registered by the owning component. The status column is the current state of
this branch; ⧗ means the metric still has to be added by its owner.

| Metric | Type | Labels | Owner | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `mft_ingestion_ticks_processed_total` | counter | — | C4 | Ticks accepted from the broker WebSocket. |
| `mft_ingestion_candles_written_total` | counter | `symbol` | C4 ⧗ | One-minute candles appended to Parquet. |
| `mft_ingestion_broker_reconnects_total` | counter | `reason` | C4 ⧗ | WebSocket reconnects. |
| `mft_execution_orders_placed_total` | counter | `side`, `symbol` | C6 | Orders that passed the risk gate. |
| `mft_execution_orders_rejected_total` | counter | `reason`, `side` | C6 | Signals refused by the risk gate. `reason` is **required**: the dashboard splits rejections by reason code. |
| `mft_execution_order_duration_seconds` | histogram | `side`, `symbol` | C6 ⧗ | Risk gate to broker order id. |
| `mft_execution_signals_total` | counter | `side`, `outcome` | C7 ⧗ | Signals received, `outcome` = `placed` or `rejected`. |
| `mft_risk_drawdown_pct` | gauge | — | C6 ⧗ | `(PeakEquity - Equity) / PeakEquity * 100`. Limit: `execution.max_drawdown_pct` = 5. |
| `mft_risk_open_positions` | gauge | — | C6 ⧗ | `len(OpenPositions)`. Limit: `execution.max_open_positions` = 10. |
| `mft_risk_daily_realised_pnl_inr` | gauge | — | C6 ⧗ | Session realised PnL. Floor: `-execution.daily_loss_limit` = -25,000. |
| `mft_risk_position_value_pct` | gauge | `symbol` | C6 ⧗ | `\|qty * price\| / equity * 100`. Limit: `execution.max_position_pct` = 10. |
| `mft_jobs_backfills_run_total` | counter | — | C8 | Historical backfill runs. |
| `mft_jobs_backfill_failures_total` | counter | `reason` | C8 ⧗ | Backfill runs that failed. |
| `mft_inference_predictions_total` | counter | `symbol`, `outcome` | C10 ⧗ | Model predictions, `outcome` = `submitted` or `below_threshold`. |
| `mft_inference_score` | gauge | `symbol` | C10 ⧗ | Last conviction score in [-1, 1]. |

`reason` on the risk metrics is one of the codes in `docs/contracts.md` §6:
`RISK_MAX_POSITION`, `RISK_MAX_POSITIONS`, `RISK_MAX_DRAWDOWN`,
`RISK_DAILY_LOSS`, `RISK_DEBOUNCED`, `RISK_BAD_QUANTITY`,
`RISK_MARKET_CLOSED`, `RISK_DUPLICATE`.

## Metrics that violate the prefix

These exist in the tree today without the `mft_` prefix. They work, and they
can be queried by their current name, but they break the naming rule and the
dashboard queries below assume the corrected names:

| Current name | Should be | File |
| :--- | :--- | :--- |
| `ingestion_ticks_processed_total` | `mft_ingestion_ticks_processed_total` | `services/ingestion/pipeline.go:43` |
| `execution_orders_placed_total` | `mft_execution_orders_placed_total` | `services/execution/engine.go:46` |
| `execution_orders_rejected_total` | `mft_execution_orders_rejected_total` | `services/execution/engine.go:50` |
| `jobs_backfills_run_total` | `mft_jobs_backfills_run_total` | `services/jobs/worker.go:43` |

None of them carry a `reason` or `side` label either, which the dashboard needs.
The rename belongs to whoever owns each file; it is a central fix, not a
component one, because it touches four components at once.

## Health and readiness

`core/metrics` serves these on the **metrics** port, alongside `/metrics`. They
are separate from the product APIs in `docs/contracts.md` §7, which own their
own `/v1/health` and `/healthz`.

| Endpoint | Status | Body |
| :--- | :--- | :--- |
| `GET /metrics` | 200 | Prometheus exposition (OpenMetrics when the client asks for it, so exemplars survive). |
| `GET /healthz` | 200 | `{"status","service","uptime_seconds"}`. Liveness: the process is up. Never fails. |
| `GET /readyz` | 200 / 503 | Liveness plus every dependency probe. 503 when a probe fails. |

```json
{
  "status": "degraded",
  "service": "execution",
  "uptime_seconds": 3601.24,
  "checks": [
    {"name": "broker", "status": "ok"},
    {"name": "storage", "status": "degraded", "error": "parquet: disk full"}
  ]
}
```

A component joins by adding one probe, with no change to the shared fx wiring:

```go
metrics.DefaultChecks().Add(metrics.Checker{Name: "broker", Check: client.Ping})
```

## Logging

`core/log` is the only place log formatting is decided.

| Setting | Source | Effect |
| :--- | :--- | :--- |
| `app.env` | config | `dev` → colourised console; anything else → JSON. |
| `app.log_level` | config | `debug`, `info`, `warn`, `error`. An unknown value is a startup error. |
| `app.timezone` | config | Timestamps are rendered in it (`Asia/Kolkata`), so a log line and a candle share a clock. |

Every logger is wrapped in a redacting core, so `zap.Any("config", cfg)` logs
the config with `api_secret` and `access_token` replaced by `[REDACTED]`, and a
`?access_token=…` in a request URI is scrubbed before it is written. HTTP
handlers wrapped in `log.Middleware` get an `X-Request-Id` correlation id that
is propagated, echoed and, via `metrics.ObserveDuration`, attached to latency
observations as an exemplar.
