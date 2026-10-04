"""Run a bounded research experiment with synthetic candles or Parquet."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

from ..candles import DuckDBCandleStore, candles_root
from ..synth import SynthSpec, generate
from .config import CostAssumptions, ExperimentConfig
from .adapters import available_strategies
from .engine import CandidateConfig, run_experiment, run_experiment_from_store
from .evaluators import available_evaluators
from model.algorithms import available_algorithms


def build_parser() -> argparse.ArgumentParser:
    """Build the offline research CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run deterministic, cost-aware walk-forward research")
    parser.add_argument("--source", choices=("synthetic", "parquet"), default="synthetic")
    parser.add_argument("--symbols", required=True, help="comma-separated instrument symbols")
    parser.add_argument("--data-dir", default="data", help="Parquet storage data_dir (candles is appended)")
    parser.add_argument("--bars", type=int, default=2000, help="synthetic bars per symbol")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--algorithm", action="append", choices=available_algorithms())
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--threshold-grid", help="comma-separated threshold candidates for validation selection")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--context-rows", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.35)
    parser.add_argument("--evaluator", choices=available_evaluators(), default="standard")
    parser.add_argument("--selection-metric", default="net_sharpe")
    parser.add_argument("--cost-preset", choices=("generic", "zerodha-mis", "zerodha-cnc"), default="zerodha-mis")
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--holding-policy", choices=("intraday", "delivery"))
    parser.add_argument("--strategy", default="threshold", help=f"registered strategy (available: {', '.join(available_strategies())})")
    parser.add_argument("--strategy-options", default="{}", help="JSON object passed as keyword options to the strategy")
    parser.add_argument("--max-bars-per-symbol", type=int, default=100_000)
    parser.add_argument("--output", type=Path, help="write the JSON artifact here; stdout if omitted")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="WARNING")
    return parser


def _candidates(args: argparse.Namespace) -> tuple[CandidateConfig, ...] | None:
    if not args.threshold_grid:
        return None
    try:
        thresholds = tuple(float(part.strip()) for part in args.threshold_grid.split(",") if part.strip())
    except ValueError as err:
        raise ValueError("--threshold-grid must contain comma-separated numbers") from err
    if not thresholds:
        raise ValueError("--threshold-grid must contain at least one threshold")
    names = tuple(args.algorithm or available_algorithms())
    return tuple(CandidateConfig(name, threshold=value) for name in names for value in thresholds)


def _strategy_options(raw: str) -> tuple[tuple[str, str | int | float | bool], ...]:
    """Parse JSON scalar keyword options for a registered strategy."""
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError("--strategy-options must be valid JSON") from err
    if not isinstance(options, dict):
        raise ValueError("--strategy-options must be a JSON object")
    if any(not isinstance(key, str) or not key.strip() for key in options):
        raise ValueError("--strategy-options keys must be nonempty strings")
    if any(value is None or not isinstance(value, (str, int, float, bool)) for value in options.values()):
        raise ValueError("--strategy-options values must be JSON strings, numbers, or booleans")
    return tuple(sorted(options.items()))


def resolve_data_dir(value: str | Path) -> Path:
    """Resolve a relative store path from the repository root, not service cwd."""
    data_dir = Path(value)
    return data_dir if data_dir.is_absolute() else Path(__file__).resolve().parents[4] / data_dir


async def _run(args: argparse.Namespace) -> dict[str, object]:
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol")
    costs = {
        "generic": lambda: CostAssumptions(spread_bps=args.spread_bps, slippage_bps=args.slippage_bps),
        "zerodha-mis": lambda: CostAssumptions.zerodha_equity_mis(
            spread_bps=args.spread_bps, slippage_bps=args.slippage_bps,
        ),
        "zerodha-cnc": lambda: CostAssumptions.zerodha_equity_cnc(
            spread_bps=args.spread_bps, slippage_bps=args.slippage_bps,
        ),
    }[args.cost_preset]()
    default_holding_policy = "delivery" if args.cost_preset == "zerodha-cnc" else "intraday"
    config = ExperimentConfig(
        horizon=args.horizon,
        context_rows=args.context_rows,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        threshold=args.threshold,
        evaluator=args.evaluator,
        selection_metric=args.selection_metric,
        costs=costs,
        holding_policy=args.holding_policy or default_holding_policy,
        strategy=args.strategy,
        strategy_options=_strategy_options(args.strategy_options),
        max_bars_per_symbol=args.max_bars_per_symbol,
    )
    candidates = _candidates(args)
    if args.source == "synthetic":
        if not 1 <= args.bars <= args.max_bars_per_symbol:
            raise ValueError("synthetic --bars must be between 1 and --max-bars-per-symbol")
        candles = {
            symbol: generate(symbol, SynthSpec(bars=args.bars, seed=args.seed))
            for symbol in symbols
        }
        result = await run_experiment(
            candles,
            config=config,
            algorithms=args.algorithm if candidates is None else None,
            candidates=candidates,
            source="synthetic",
        )
    else:
        store = DuckDBCandleStore(
            candles_root(resolve_data_dir(args.data_dir)),
            max_rows=config.max_bars_per_symbol + 1,
        )
        try:
            result = await run_experiment_from_store(
                store,
                symbols,
                config=config,
                algorithms=args.algorithm if candidates is None else None,
                candidates=candidates,
            )
        finally:
            await store.aclose()
    return result.to_json()


def main(argv: list[str] | None = None) -> int:
    """Run the selected source and emit a reproducible JSON document."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    try:
        document = asyncio.run(_run(args))
    except (ValueError, RuntimeError, OSError) as err:
        print(f"research: {err}", file=sys.stderr)
        return 2
    rendered = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
