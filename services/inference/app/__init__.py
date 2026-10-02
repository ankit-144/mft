"""MFT inference service - the Python side of service 2.

The pieces, in the order a minute close runs through them:

| Module      | Responsibility                                                |
| :---------- | :------------------------------------------------------------ |
| `config`    | Reading the frozen YAML schema of `docs/contracts.md` §8.        |
| `candles`   | Pulling the last N bars of a symbol out of the Parquet store.   |
| `features`  | The 18-column table, ported from `core/features` (C3).          |
| `predictor` | Holding the model and calling it off the event loop.             |
| `signals`   | `contracts.Signal` and the idempotency key.                      |
| `execution` | Posting a signal to the execution service with a retry budget.  |
| `loop`      | The minute scheduler that ties the above together.               |
| `runtime`   | Assembling a service from configuration.                         |
| `main`      | FastAPI: `/healthz`, `/v1/context`, `/v1/predict`.               |
| `backtest`  | Walk-forward replay of the same chain, with no broker at all.    |
| `synth`     | Deterministic synthetic candles, so a backtest can run offline.  |

The production path is `loop`, not `main`. See `Plan.md` §4 for why inference
pulls at each minute close instead of being pushed to, and `docs/contracts.md`
§7 for the HTTP surface.

The model lives in the sibling `model` package, which C5 owns. This package
imports it as `from model import ...`, which resolves when
`services/inference` is the working root — what `make run-inference` and the
test suite both arrange. There is deliberately no `app.model` re-export: a
module named `model` inside this package used to shadow it and had to go.
"""

from __future__ import annotations

__all__ = [
    "backtest",
    "candles",
    "config",
    "execution",
    "features",
    "loop",
    "predictor",
    "runtime",
    "signals",
    "synth",
]
