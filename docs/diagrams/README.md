# Architecture diagrams

Twenty-seven diagrams describing the MFT platform. Written in [D2](https://d2lang.com),
rendered to SVG.

```sh
make diagrams          # render every .d2 to docs/diagrams/svg/
make diagrams-check    # parse-check every .d2 (the CI form)
```

Style and the d2 v0.9.0 syntax traps: [`STYLE.md`](./STYLE.md).

## How to read this set

**If you are new to the platform, read these five in order.** They build from
the outside in, and each one earns the next.

| # | Diagram | What it answers |
| :-- | :--- | :--- |
| [01](svg/01-system-context.svg) | System context | Who talks to this platform and why? |
| [02](svg/02-container-view.svg) | Container view | What runs where, and what does each service link? |
| [03](svg/03-data-flow-tick-to-order.svg) | Data flow | Trace one minute bar from the broker to an order. |
| [05](svg/05-state-stores.svg) | State stores | What is durable, what is in-memory, and what crosses a process boundary? |
| [06](svg/06-decision-and-risk.svg) | Decision & risk | A signal arrives. Which gate stops it, and why? |

**If you want to change something safely**, read 04 (ownership), 24 (the frozen
surface) and 26 (what is actually proven) first. Between them they will tell you
which files you may touch and which claims you should not trust.

**If you are about to go live**, read [26](svg/26-crosscut-verification.svg) on
its own. It is the honest ledger of what has been verified and what has never
been executed at all.

## High level

| # | Diagram | Question it answers |
| :-- | :--- | :--- |
| [01](svg/01-system-context.svg) | System context | External actors and systems, and the platform as one box. |
| [02](svg/02-container-view.svg) | Container view | The four services, their HTTP surfaces, and real dependency edges between the shared `core/` packages. |
| [03](svg/03-data-flow-tick-to-order.svg) | Data flow, tick to order | The end-to-end happy path with every hop named. |
| [04](svg/04-component-ownership.svg) | Component ownership | Which component owns which paths, and why the branches did not conflict. |
| [05](svg/05-state-stores.svg) | State stores | Durable vs process-local, and the fact that nothing crosses through `fluxKV`. |
| [06](svg/06-decision-and-risk.svg) | Decision & risk | The seven risk checks in evaluation order, and the HTTP status mapping. |
| [07](svg/07-pull-vs-push.svg) | Pull vs push | A decision record: why inference pulls at minute close instead of being pushed to. |
| [08](svg/08-build-and-branch-strategy.svg) | Build & branch strategy | How ten components were developed concurrently, and where that did not work. |
| [09](svg/09-runtime-topology.svg) | Runtime topology | Processes, ports, the fx graph, and the DI failure that unit tests could not catch. |

## Component detail

| # | Component | Diagram |
| :-- | :--- | :--- |
| 10 | C1 Kite broker | [Connector, instrument master, orders, error mapping](svg/10-lld-kite-broker.svg) |
| 11 | C1 Kite broker | [WebSocket stream as a reconnect state machine](svg/11-lld-stream-reconnect.svg) |
| 12 | C2 Storage | [Parquet writer and the atomic-publish invariant](svg/12-lld-storage-writer.svg) |
| 13 | C2 Storage | [Reader, dedup, and the DuckDB glob seam](svg/13-lld-storage-reader.svg) |
| 14 | C3 Features | [The 18-column table and the row-alignment guarantee](svg/14-lld-features.svg) |
| 15 | C4 Ingestion | [Pipeline, exactly-once bars, backpressure](svg/15-lld-ingestion.svg) |
| 16 | C6 Risk | [The risk gate in evaluation order](svg/16-lld-risk-engine.svg) |
| 17 | C7 Execution API | [HTTP boundary and the exactly-once guarantee](svg/17-lld-execution-api.svg) |
| 18 | C8 Jobs | [Backfill, rate limiting, resumability](svg/18-lld-jobs-backfill.svg) |
| 19 | C5 Inference model | [The swappable model layer, and why it exists](svg/19-lld-inference-model.svg) |
| 20 | C10 Inference loop | [The minute-close loop, and failing closed](svg/20-lld-inference-loop.svg) |
| 21 | C11 Backtest | [Walk-forward, and the no-lookahead guarantee](svg/21-lld-backtest.svg) |
| 22 | C9 Observability | [Metrics, logging, and the redaction rules](svg/22-lld-observability.svg) |

## Cross-cutting

| # | Diagram | Question it answers |
| :-- | :--- | :--- |
| [23](svg/23-crosscut-idempotency.svg) | Exactly-once | One idempotency key traced across Python, HTTP and Go into an in-memory map. |
| [24](svg/24-crosscut-contracts.svg) | The frozen surface | What is frozen, who depends on it, and where reality diverged. |
| [25](svg/25-crosscut-failure-modes.svg) | Failure modes | Retry, fail-fast, fail-closed, drop, refuse-to-start, bounded lifetime. |
| [26](svg/26-crosscut-verification.svg) | Verification | What is proven, what is proven by hand, and what has never run. |

## Things the diagrams record that the prose docs did not

These were found while drawing, by reading the code rather than the comments.
They are drawn truthfully in the diagrams above and are listed here so they are
not a surprise.

- **`core/features` and `core/analytics` have zero Go importers.** The Python
  inference service runs its own port of the 18-column feature schema. The two
  are pinned only by a hand-transcribed golden fixture, so changing one without
  the other passes both suites. → `24`
- **The feature builders' doc comments are wrong.** Both say a UTC
  `minute_of_session` at the 09:15 IST open reads `-345`; the arithmetic gives
  `-330`. The code is right and the comment is not. → `14`
- **`broker.product` and `analytics.duckdb_path` are dead config.** Both are
  declared, defaulted and cross-validated; neither is read by anything.
  → `24`
- **`inference.*` has no Go reader.** It is declared in the Go schema and
  consumed by exactly one Python process. → `24`
- **The broker connector hardcodes `product = "NRML"`**, so the config key is
  bypassed. → `10`
- **A socket outage never reaches the ingestion supervisor.** `Stream` swallows
  session errors and loops internally, so the restart counter stays flat during
  an outage and only `last_tick_age` moves. → `25`
- **The inference Prometheus target is permanently DOWN.** The Python service
  registers no `mft_` metrics, and `MFT_METRICS_ADDR` for it is inert. → `22`
- **No service calls `RegisterBuildInfo`,** so the build-info and uptime
  dashboard panels are empty. Two dashboard queries name metrics the code never
  registers. → `22`
- **The idempotency key is written only after the broker accepts.** The replay
  window is closed by `execution.idempotency_ttl_seconds` (24h), not by the
  cache itself. → `23`, `25`
- **`docs/contracts.md` §2 and `core/broker.Client` disagree** on
  `PlaceOrder`'s signature, and the test mock implements the deprecated form.
  → `24`
- **`core/fx.go` and `core/testutil/` are owned by nobody** and are not frozen,
  which is precisely the conflict the ownership table exists to prevent.
  → `04`, `24`