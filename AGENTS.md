# Repository working rules

## Resource safety

This machine has 14 GB RAM. Never load TabFM weights, read the Hugging Face
checkpoint cache, download weights, or load large assets. Do not run the full
`model/tests` directory or `test_tabfm_model.py` weight tests. Keep generated
data, credentials, local configuration, environments and journals outside Git.

## Architecture and ownership

| Area | Source |
| --- | --- |
| Broker, storage, features, observability | `core/` |
| Ingestion | `services/ingestion/` |
| Order lifecycle, risk, HTTP | `services/execution/` |
| Historical jobs | `services/jobs/` |
| Runtime and model contracts | `services/inference/app/`, `services/inference/model/` |
| Strategies, experiments, evaluators | `services/inference/app/research/` |

Coordinate shared-path ownership before parallel edits. Use isolated worktrees
for independent branches and preserve original component branches.

## Interface and code conventions

- Keep responsibility documentation at functional interfaces; remove narrative
  or redundant inline comments.
- Go: use `gofmt`, context-aware I/O, contextual errors, bounded queues and
  bounded concurrency.
- Python: use typed interfaces, injected time/storage/model dependencies,
  logging and finite numeric validation.
- Preserve the ordered 18-feature schema and UTC timestamps; sessions use
  Asia/Kolkata.
- Forecast targets are cumulative future log returns over the configured
  horizon.
- Register algorithms, strategies and evaluators independently; select
  parameters using validation data only.
- Metrics use isolated registries and the `mft_` prefix; readiness reflects
  usable dependencies.

## Validation

Run tests, builds, benchmarks or other validation only when requested. When
requested, the available targets are:

```sh
make lint
make test
make test-python
make integration
make benchmarks
```

Go uses two build workers by default. Python tests block real checkpoint access.
Socket fixtures skip only permission-denied errors in restricted sandboxes;
distinguish those skips from verified integration paths. The Python test-only
selector heartbeat works around forbidden socketpair writes.

## Operating defaults

`inference.dry_run` and `execution.paper_trading` default to true. Live execution
requires positive capital, an API token, reconciliation and durable state. No
test may place a real order. No return estimate should be presented as verified
market profitability.
