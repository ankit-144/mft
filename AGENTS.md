# AGENTS.md

Rules for any agent or human working on this repository.

## Read these first

| Document | What it gives you |
| :--- | :--- |
| [`Plan.md`](./Plan.md) | North star, architecture, the fluxKV decision, TabFM licensing |
| [`docs/contracts.md`](./docs/contracts.md) | **Frozen** types, interfaces, feature schema, config keys |

If your change is not described in `docs/contracts.md` and you are not certain
it belongs to you, stop and ask before writing code.

## Frozen files — do not edit

Editing any of these creates a merge conflict with every parallel branch.

- `core/config/config.go` — every key is already declared
- `core/contracts/` — shared domain types and interfaces
- `go.work`, `go.mod`, `go.sum` — dependencies are pre-resolved
- `Makefile` — all targets are pre-declared
- `opencode.json`, `AGENTS.md`, `docs/`, `Plan.md`

## Ownership

Each component owns an exclusive set of paths. Touch nothing else.

| Component | Owns |
| :--- | :--- |
| C1 kite-broker | `core/broker/**` |
| C2 storage-analytics | `core/storage/**`, `core/analytics/**` |
| C3 features | `core/features/**` |
| C4 ingestion | `services/ingestion/**` |
| C5 tabfm-model | `services/inference/model/**`, `services/inference/requirements*.txt` |
| C6 risk-engine | `services/execution/risk/**`, `services/execution/engine.go`, `core/fluxkv/**` |
| C7 execution-api | `services/execution/http.go` |
| C8 jobs-backfill | `services/jobs/**` |
| C9 observability | `core/metrics/**`, `core/log/**`, `observability/**` |
| C10 inference-api | `services/inference/app/**` |

If you need a change outside your ownership, it belongs in a different
component. Say so in your PR description instead of making the edit.

## Conventions

- Go 1.25, `gofmt` clean, `go vet` clean. Run `make lint` before pushing.
- Every exported symbol has a doc comment starting with its name.
- No comments that merely restate the code.
- Errors are wrapped with `%w` and given context.
- Prometheus: register every new metric through the provided `*prometheus.Registry`
  with the `mft_` prefix, and a `Help` string.
- Tests use `core/testutil` for mocks and metric assertions.
- Python: type hints, `logging` not `print`, no new top-level dependencies
  without adding them to `services/inference/requirements.txt`.

## Before you push

```sh
make lint
make test
```

Both must pass. For Python components, also confirm the venv imports cleanly.

## Safety

- `inference.dry_run` defaults to `true`. Do not change that default.
- Never commit credentials. `.env` and `configs/config.yaml` are gitignored.
- Never place a real order from a test.
