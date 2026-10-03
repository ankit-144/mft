# High-level design (Mermaid)

These render **inline on GitHub** and in any Markdown viewer, with no tooling —
which is the reason to keep them in Mermaid even though the detailed set lives
in D2. `make mermaid` regenerates the `.svg`/`.png` files for anywhere that
cannot render Mermaid natively.

The D2 set in the parent directory carries the component detail. This file is
the read-it-on-GitHub overview.

## The platform

Four services and a shared `core/` Go module, running on one machine. Every
label below names something real: a type, a config key, a file path, an HTTP
route, or a reason code from [`docs/contracts.md`](../contracts.md).

```mermaid
flowchart TB
  classDef ext   fill:#eceff1,stroke:#90a4ae,color:#212121
  classDef go    fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
  classDef py    fill:#ede7f6,stroke:#5e35b1,color:#4a148c
  classDef lib   fill:#e0f2f1,stroke:#00897b,color:#004d40
  classDef store fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef mem   fill:#fff8e1,stroke:#f9a825,color:#e65100
  classDef bad   fill:#ffebee,stroke:#c62828,color:#b71c1c

  subgraph EXT["External"]
    direction LR
    KITE["<b>Zerodha Kite Connect</b><br/>WS quote stream · REST orders + history<br/><i>external system</i>"]:::ext
    HF["<b>HuggingFace Hub</b><br/>TabFM weights<br/><b>tabfm-non-commercial-v1.0</b><br/><i>non-commercial</i>"]:::ext
    OBS["<b>Prometheus + Grafana</b><br/>scrape :9090 / :9091 / :9092"]:::ext
  end

  subgraph DATA["Data plane — Go"]
    direction TB
    ING["<b>Service 1 · ingestion</b> <i>(Go)</i><br/><code>services/ingestion</code><br/>dedup · reconnect · 1m bars<br/>exactly-once completion"]:::go
    JOBS["<b>Service 4 · jobs</b> <i>(Go)</i><br/><code>services/jobs</code><br/>weekly backfill<br/>token bucket · resumable"]:::go
  end

  subgraph STORE["Durable store — survives restart"]
    direction TB
    PQ[("<b>Parquet · hive-partitioned</b><br/><code>data/{ticks,candles,historical}</code><br/><code>symbol=&lt;SYM&gt;/date=&lt;D&gt;/part-*.parquet</code><br/><i>.inflight-* then os.Rename</i>")]:::store
    DUCK[("<b>DuckDB</b><br/>in-memory over the immutable files<br/><code>read_parquet(glob)</code>")]:::store
  end

  subgraph MODEL["Service 2 · inference — Python"]
    direction TB
    LOOP["<b>minute-close loop</b><br/><code>app/loop.py</code><br/>pull, never pushed to<br/>fails closed"]:::py
    FEAT["<b>18-column feature table</b><br/><code>app/features.py</code><br/>rows[i] ↔ candles[i], no offset<br/>60 warm-up rows dropped"]:::py
    MODELBOX["<b>InferenceModel protocol</b><br/><code>model/base.py</code><br/>TabFM ↔ Heuristic, swappable"]:::py
    SIG["<b>Signal</b><br/><code>idempotency_key</code><br/>SYM:SIDE:YYYYMMDDTHHMM <i>UTC</i><br/><code>dry_run</code> = <b>true</b>"]:::py
  end

  subgraph RISK["Service 3 · execution — Go"]
    direction TB
    HTTP["<b>HTTP :8080</b><br/><code>POST /v1/signals</code> · <code>/v1/orders</code><br/>strict JSON · 64 KiB cap"]:::go
    GATE{"<b>Risk gate</b><br/>7 checks · first failure wins<br/><code>contracts.Checks</code>"}:::bad
    ORDER["<b>Order</b><br/><code>PlaceOrder</code><br/>MARKET→202 · LIMIT→200"]:::go
  end

  KV[("<b>fluxKV</b> — per-process, never a bus<br/><code>EXEC:sym:side</code> debounce · <code>IDEM:key</code> idempotency<br/>open bars · <code>Sweep</code> reclaims TTL")]:::mem

  KITE -- "ticks: binary LTP frames" --> ING
  ING -- "raw ticks" --> PQ
  ING -- "completed bars" --> PQ
  ING <-- "open bar, watermark" --> KV
  JOBS -- "segments, .partial then rename" --> PQ
  KITE -- "historical candles, 60-day segments" --> JOBS

  PQ --> DUCK
  DUCK -- "<b>pull</b> last context_rows bars" --> LOOP
  LOOP --> FEAT
  FEAT --> MODELBOX
  MODELBOX -- "score in [-1,1]" --> SIG
  SIG -. "when dry_run: false" .-> HTTP

  HTTP -- "claims IDEM key" --> KV
  HTTP --> GATE
  GATE -- "pass" --> ORDER
  GATE -- "RISK_* / RISK_DUPLICATE" --> REJ["<b>reject</b><br/>400 · or 409 with the<br/><i>first</i> order_id"]:::bad
  ORDER -- "orders, cancels, positions" --> KITE

  HF -. "loaded once, behind the protocol" .-> MODELBOX
  ING -.-> OBS
  HTTP -.-> OBS
  JOBS -.-> OBS

  NOTE["<b>fluxKV never crosses a process boundary</b> — only HTTP and the filesystem do.<br/>That is why inference <i>pulls</i> from Parquet instead of subscribing. See <code>Plan.md</code> §4."]:::lib
```

### The one decision that shapes everything

`fluxKV` is an in-memory `map`, and the original code had `ingestion` and
`execution` — separate processes — both reading it. Execution's debounce check
therefore always saw an empty cache, silently.

**`fluxKV` is a per-process cache, never a message bus.** Parquet is the durable
source of truth, and inference *pulls* at each minute close rather than
subscribing. At a 60-second decision cadence push latency buys nothing, and pull
survives a missed minute with no broker and no offset reconciliation. Only two
things cross a process boundary: **HTTP** and **the filesystem**.

## One minute close, end to end

The sequence below is the real order in the code, including the failure paths
and the idempotency handshake.

```mermaid
sequenceDiagram
    autonumber
    participant BR as broker.Kite
    participant ING as ingestion
    participant PQ as Parquet
    participant INF as inference
    participant EX as execution

    Note over ING,EX: One minute close.

    BR->>ING: tick (binary LTP frame, 68 bytes)
    ING->>ING: deliver() — three non-blocking selects
    Note over ING: full? evict the OLDEST tick.<br/>Blocking would stall the socket reader,<br/>stall Kite pings, and escalate a<br/>storage slowdown into a reconnect storm.
    ING->>ING: fold() then fluxKV.UpdateCandle
    ING->>PQ: Writer.Append(storage.NewTick(tick))

    Note over ING,PQ: Minute rolls over. Completion clears the open bar AND<br/>advances the per-symbol watermark in one step,<br/>so a bar is written exactly once.
    ING->>PQ: CandleWriter.Append(NewCandle(c))
    Note over PQ: flush() writes .inflight-* then os.Rename.<br/>Readers glob part-*.parquet and never see a partial file.

    INF->>PQ: tail(symbol, context_rows + WARMUP_ROWS)
    PQ-->>INF: last 160 bars (the newest 60 are context-only)
    INF->>INF: FeatureBuilder.build to 18 columns
    INF->>INF: context(rows) — warm-up dropped, NOT zero-padded
    INF->>INF: Predictor.predict(context, horizon)
    INF->>INF: side_for_score(score), compare to score_threshold
    alt dry_run is true (the default)
        INF-->>INF: log the signal it would have sent
    else dry_run is false
        INF->>EX: POST /v1/signals + idempotency_key
        EX->>EX: risk.ValidateSignal (shape only, claims nothing)
        EX->>EX: lock, then Get("IDEM:"+key)
        alt key already present — a prior attempt completed
            EX-->>INF: 409 RISK_DUPLICATE + the FIRST order_id
        else key absent
            EX->>EX: contracts.Checks — 7 checks, first failure wins
            alt a check fails
                EX-->>INF: 400 RISK_* with the reason code
            else all pass
                EX->>BR: PlaceOrder
                BR-->>EX: order_id
                EX->>EX: Set("IDEM:"+key, order_id) AFTER the broker returns
                EX-->>INF: 202 MARKET / 200 LIMIT
            end
        end
    end
```

## WebSocket lifecycle

`Stream` reconnects internally, but it can also **return** — and an unsupervised
return left the service reporting healthy while ingesting nothing. That is why
`Pipeline.supervise` re-invokes it on a second, independent backoff.

```mermaid
stateDiagram-v2
    direction TB
    [*] --> Idle

    Idle --> Preflight: Stream(ctx, symbols)
    Preflight --> Dialing: ok
    Preflight --> Failed: ErrAuth / bad config

    Dialing --> Subscribing: handshake ok
    Dialing --> Backoff: dial refused

    Subscribing --> Reading: subscribe frame sent, mode=full
    Reading --> Reading: binary LTP frame to contracts.Tick
    Reading --> Reading: JSON ping to pong, refresh read deadline

    Reading --> Backoff: read deadline exceeded
    Reading --> Stopped: ctx cancelled

    Backoff --> Dialing: delay elapsed (1s 2s 4s 8s 16s 32s 60s 60s...)
    Backoff --> Stopped: ctx cancelled

    Failed --> [*]
    Stopped --> [*]

    note right of Reading
        Deadline refreshes on ANY read.
        Sequence() survives reconnects.
    end note

    note right of Backoff
        Session errors never escape Stream.
        Pipeline.supervise re-invokes it.
    end note
```

Two details that are easy to get wrong and are load-bearing:

- The read deadline is refreshed on **any** read, not only on `pong`, so a
  silent socket is detected rather than waited on indefinitely.
- `Sequence()` is monotonic and does **not** reset across reconnects, which is
  what makes a duplicated or reordered frame detectable after a reconnect.

The backoff uses iterative doubling rather than `2^n` deliberately: a large
attempt count must clamp, not overflow into a hot loop.

## Licensing, because it constrains the architecture

TabFM's **source** is Apache-2.0. The **pretrained weights** are
`tabfm-non-commercial-v1.0` — non-commercial, non-production. A trading platform
is neither.

That is why the model sits behind the `InferenceModel` protocol: the decision has
to be reversible. The same reasoning drives the operational constraint — this
machine has 14GB with the OOM killer active, so CI never loads the weights and
`TabFMModel` is verified against a fake checkpoint. See
[`Plan.md`](../../Plan.md) §5 and §0.

## Rendering

```sh
make mermaid          # .mmd -> .svg and .png
```

`mmdc` 12.0.0 plus `google-chrome`. The D2 equivalent is `make diagrams`.
