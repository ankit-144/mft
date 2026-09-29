# Cross-Component Contracts

Every signature in this file is **frozen**. Components import or implement these
definitions; they do not redefine them. Changing anything here is a foundation
PR against `master`, never a component PR.

Implemented in `core/contracts/` — one owner (foundation), imported by all.

---

## 1. Domain types

```go
package contracts

// Tick is a raw market data tick from the broker.
type Tick struct {
    Symbol    string    `json:"symbol"`
    Token     int64     `json:"token"`
    Price     float64   `json:"price"`
    Volume    int64     `json:"volume"`
    Timestamp time.Time `json:"timestamp"`
}

// Candle is an aggregated one-minute OHLCV bar.
type Candle struct {
    Symbol    string    `json:"symbol"`
    Timestamp time.Time `json:"timestamp"` // truncated to the minute, UTC
    Open      float64   `json:"open"`
    High      float64   `json:"high"`
    Low       float64   `json:"low"`
    Close     float64   `json:"close"`
    Volume    int64     `json:"volume"`
}

// Signal is a model conviction for one symbol at one minute close.
type Signal struct {
    Symbol     string    `json:"symbol"`
    Side       string    `json:"side"`       // "BUY" | "SELL"
    Quantity   int       `json:"quantity"`
    Price      float64   `json:"price"`      // reference price at signal time
    Score      float64   `json:"score"`      // conviction in [-1, 1]
    Model      string    `json:"model"`      // e.g. "tabfm-v1.0.0"
    AsOf       time.Time `json:"as_of"`      // minute close this predicts from
    Idempotency string   `json:"idempotency_key"`
}
```

**Conventions** — all timestamps are UTC, RFC3339 on the wire, Unix millis in
Parquet. `Side` is uppercase `BUY`/`SELL`. Prices are `float64` in rupees,
quantity in whole shares (no fractional — Indian equities deliver whole units).

---

## 2. Broker — `core/broker`

```go
// Streamer connects to the broker WebSocket and delivers ticks.
type Streamer interface {
    Stream(ctx context.Context, symbols []string, out chan<- Tick) error
}

// Client executes and queries orders.
type Client interface {
    PlaceOrder(ctx context.Context, req OrderRequest) (string, error)
    CancelOrder(ctx context.Context, orderID string) error
    GetPositions(ctx context.Context) ([]Position, error)
}

// Instrument describes a tradable symbol.
type Instrument struct {
    Token   int64
    Symbol  string
    Exchange string
    LotSize int
}

type OrderRequest struct {
    Symbol   string
    Side     string
    Quantity int
    Price    float64   // 0 => market order
    Type     string    // "MARKET" | "LIMIT"
}
```

**C1 owns this.** The existing `Kite` stub at `core/broker/broker.go` is
replaced with a real implementation. C1 must keep the `Streamer` and `Client`
method sets above satisfied so `core/fx.go:32-33` bindings keep working.

---

## 3. Storage — `core/storage`, `core/analytics`

```go
// Writer persists rows to partitioned Parquet.
type Writer[T any] interface {
    Append(row T) error
    Flush(ctx context.Context) error
    Close() error
}

// Reader queries the Parquet store.
type Reader interface {
    // Candles returns up to limit candles for symbol, ascending by time.
    Candles(ctx context.Context, symbol string, limit int) ([]Candle, error)
}
```

**Parquet layout** — hive-partitioned, immutable files, one per flush:

```
data/candles/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<timestamp>.parquet
data/ticks/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<timestamp>.parquet
data/historical/symbol=<SYMBOL>/from=<date>/to=<date>/candles.parquet
```

**DuckDB rules** — the query engine reads *only closed, immutable* files. The
writer must never mutate a file after it is closed. This is what allows
inference to read while ingestion writes.

```go
// Engine is an embedded DuckDB handle for research and backtesting.
type Engine interface {
    Query(ctx context.Context, sql string, args ...any) (*duckdb.Rows, error)
    Close() error
}
```

**C2 owns this.**

---

## 4. Features — `core/features`

```go
// Builder turns a candle window into a model-ready feature table.
type Builder interface {
    // Build returns feature rows aligned to the input candles, newest last.
    Build(candles []Candle) (FeatureTable, error)
}

type FeatureTable struct {
    Columns []string      // ordered feature names
    Rows    [][]float64   // rows[i] aligns with candles[i]
    AsOf    time.Time     // timestamp of the last row
}
```

**Feature schema is frozen at the following 18 columns.** TabFM treats the
column set as part of the table contract; changing it invalidates historical
context rows.

| # | Feature | Description |
| :-- | :--- | :--- |
| 1 | `ret_1` | 1-bar log return |
| 2 | `ret_5` | 5-bar log return |
| 3 | `ret_15` | 15-bar log return |
| 4 | `ret_60` | 60-bar log return |
| 5 | `vol_5` | stdev of 1-bar returns, 5-bar window |
| 6 | `vol_20` | stdev of 1-bar returns, 20-bar window |
| 7 | `vol_ratio` | `vol_5 / vol_20` |
| 8 | `range_1` | `(high - low) / close` |
| 9 | `body_1` | `(close - open) / open` |
| 10 | `upper_wick_1` | upper wick fraction |
| 11 | `lower_wick_1` | lower wick fraction |
| 12 | `volume_z_20` | volume z-score, 20-bar window |
| 13 | `volume_ratio` | `volume / sma(volume, 20)` |
| 14 | `momentum_rsi_14` | RSI(14), normalised to [-1, 1] |
| 15 | `sma_gap_10` | `(close - sma(close,10)) / sma(close,10)` |
| 16 | `minute_of_session` | bars elapsed since 09:15 IST |
| 17 | `hour_of_day` | 0–23, IST |
| 18 | `spread_proxy` | `abs(close - open) / volume` |

Warm-up: the first 60 rows are context-only and are never used as prediction
targets. `Builder` must return an error if fewer than 60 candles are supplied.

**C3 owns this.**

---

## 5. Inference model — `services/inference/model`

This protocol exists so TabFM can be replaced without touching the service.
See `Plan.md` §5 for the licensing reason.

```python
class InferenceModel(Protocol):
    @property
    def name(self) -> str: ...
    async def load(self) -> None: ...
    def predict(
        self,
        context: pd.DataFrame,   # rows x 18 features
        horizon: int,            # bars ahead, 1
    ) -> float:                 # expected return, in [-1, 1] after scaling
    def is_loaded(self) -> bool: ...
```

Implementations: `TabFMModel` (v1 default), plus a `HeuristicModel` used for
tests and for environments where TabFM weights are unavailable. The service
must work with either, selected by config.

**C5 owns this.**

---

## 6. Risk engine — `services/execution/risk`

```go
// Checker validates a signal against risk policy. All checks must pass.
type Checker interface {
    Check(ctx context.Context, sig contracts.Signal, portfolio Portfolio) error
}

// Portfolio is the risk engine's view of current exposure.
type Portfolio struct {
    Cash          float64
    RealisedPnL   float64
    PeakEquity    float64
    OpenPositions map[string]int
}
```

**Checks, evaluated in order. First failure rejects the signal.**

1. `max_position_pct` — `|qty * price| / equity <= max_position_pct`
2. `max_positions` — `len(OpenPositions) <= max_open_positions`
3. `max_drawdown_pct` — `(PeakEquity - Equity) / PeakEquity <= max_drawdown_pct`
4. `daily_loss_limit` — `RealisedPnL >= -daily_loss_limit`
5. `debounce` — no prior order for `(symbol, side)` within `debounce_ttl`
6. `quantity` — `1 <= qty <= max_order_quantity`, and `qty % lot_size == 0`
7. `market_hours` — within 09:15–15:30 IST, weekdays

**Rejection reason codes** (returned to the caller, counted as metrics):

`RISK_MAX_POSITION`, `RISK_MAX_POSITIONS`, `RISK_MAX_DRAWDOWN`,
`RISK_DAILY_LOSS`, `RISK_DEBOUNCED`, `RISK_BAD_QUANTITY`,
`RISK_MARKET_CLOSED`, `RISK_DUPLICATE`

**C6 owns this.** Debounce (check 5) uses `fluxKV` with a TTL key
`EXEC:<symbol>:<side>` — this is legitimate per-process use and is what
`fluxKV` is for.

---

## 7. HTTP APIs

All bodies are JSON. Errors return the code's HTTP status plus
`{"error": "<code>", "message": "<human readable>"}`.

### Execution — service 3, `:8080`

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/v1/signals` | Inference submits a signal. Runs the full risk gate, places if it passes. |
| `POST` | `/v1/orders` | Direct order placement. Same gate. |
| `GET` | `/v1/portfolio` | Current cash, PnL, open positions. |
| `GET` | `/v1/health` | Liveness + broker connectivity. |

`POST /v1/signals` request/response:

```json
// request — the contracts.Signal struct
{
  "symbol": "RELIANCE", "side": "BUY", "quantity": 10,
  "price": 2934.5, "score": 0.72, "model": "tabfm-v1.0.0",
  "as_of": "2026-09-29T10:31:00Z", "idempotency_key": "RELIANCE:BUY:20260929T1031"
}

// 202 Accepted — accepted and filled
{ "order_id": "4412", "status": "FILLED", "score": 0.72 }

// 200 OK — accepted, passed risk, order working
{ "order_id": "4412", "status": "OPEN", "score": 0.72 }

// 400 Bad Request — rejected by risk gate
{ "error": "RISK_MAX_POSITION",
  "message": "order value 29345 exceeds 10% of equity 250000" }

// 409 Conflict — duplicate idempotency key
{ "error": "RISK_DUPLICATE", "message": "key already processed", "order_id": "4412" }
```

**Idempotency** is mandatory. `idempotency_key` is required on
`/v1/signals`; replaying a key returns the original `order_id` with `409` and
must **not** place a second order. Inference retries, so this must hold.

**C7 owns this.**

### Inference — service 2, `:8000`

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/healthz` | Liveness + `model_loaded`. |
| `GET` | `/v1/context` | Last N feature rows for a symbol. Debugging. |
| `POST` | `/v1/predict` | Explicit prediction for a supplied context. Debugging. |

The **production** path is not an HTTP endpoint — it is an internal scheduler
that fires on each minute close, pulls context from DuckDB, runs the model, and
POSTs to execution. `/v1/predict` exists for debugging and tests.

```json
// POST /v1/predict
{ "symbol": "RELIANCE", "context_rows": 100 }
// 200
{ "symbol": "RELIANCE", "score": 0.72, "model": "tabfm-v1.0.0",
  "as_of": "2026-09-29T10:31:00Z" }
```

**C10 owns this.**

---

## 8. Config schema — frozen

`core/config/config.go` declares every key. **No component may add one.** If a
key you need is missing, it is a bug in the foundation, not a licence to edit.

```yaml
app:
  name: mft
  env: dev              # dev | staging | prod
  log_level: info       # debug | info | warn | error
  timezone: Asia/Kolkata

metrics:
  addr: ":9090"
  path: /metrics

broker:
  api_key: ""
  api_secret: ""
  access_token: ""
  product: MIS            # MIS | NRML | CNC
  instruments: [RELIANCE, TCS, INFY]
  reconnect_max_backoff_seconds: 60
  request_timeout_seconds: 10

storage:
  data_dir: data
  flush_interval_seconds: 300
  flush_max_rows: 10000
  partition_by: date

analytics:
  duckdb_path: data/mft.duckdb
  read_only: true

execution:
  addr: ":8080"
  capital: 1000000
  debounce_ttl_seconds: 300
  idempotency_ttl_seconds: 86400
  max_position_pct: 10.0
  max_open_positions: 10
  max_drawdown_pct: 5.0
  daily_loss_limit: 25000
  max_order_quantity: 500
  market_holidays: []      # NSE holidays, YYYY-MM-DD. Empty = every weekday trades.

inference:
  addr: ":8000"
  model: tabfm         # tabfm | heuristic
  execution_url: "http://localhost:8080"
  context_rows: 100
  horizon_bars: 1
  score_threshold: 0.55
  order_quantity: 10
  instruments: [RELIANCE, TCS, INFY]
  dry_run: true        # when true, never places a real order

jobs:
  schedule: "0 2 * * 6"
  rate_limit_per_second: 3
  historical_dir: data/historical
  backfill_lookback_days: 730
```

`inference.dry_run: true` is the default and should stay `true` until the
backtest numbers justify otherwise.
