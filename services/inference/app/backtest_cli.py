"""Command-line entry point for stored and synthetic backtest replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Sequence

from .candles import Candle
from .backtest import (
    DEFAULT_MAX_BARS, DEFAULT_MAX_TRADES_SHOWN, DEFAULT_MODEL, DEFAULT_START_PRICE,
    WEIGHTED_MODELS, WEIGHTS_WARNING, BacktestError, BacktestResult, Config, ConfigError,
    DuckDBCandleStore, Predictor, SynthSpec, candles_root, configure_logging, format_summary,
    generate_many, load_candles, load_config, load_execution_limits, run_backtest,
    trading_zone, window_candles,
)

logger = logging.getLogger("mft.inference.backtest.cli")


EXIT_OK: Final[int] = 0
EXIT_REFUSED: Final[int] = 2


def build_parser() -> argparse.ArgumentParser:
    """The `python -m app.backtest` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.backtest",
        description=(
            "Replay stored or synthetic candles through the live feature, model, decision "
            "and risk path. Places no order and contacts nothing."
        ),
        epilog=(
            "Defaults to the lightweight heuristic. Weighted research backends are optional. "
            "--model tabfm needs --allow-weights."
        ),
    )
    parser.add_argument("--config", default=None, help="config file (default: $MFT_CONFIG)")
    parser.add_argument(
        "--symbol", action="append", dest="symbols", metavar="SYM",
        help="instrument to run; repeatable. Default: every configured instrument.",
    )
    parser.add_argument("--from", dest="start", metavar="YYYY-MM-DD",
                        help="window start, inclusive, in IST")
    parser.add_argument("--to", dest="end", metavar="YYYY-MM-DD",
                        help="window end, inclusive of the whole day, in IST")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL,
                        help=f"model selector (default: {DEFAULT_MODEL}, not inference.model)")
    parser.add_argument(
        "--allow-weights", action="store_true",
        help=f"permit a weighted model such as {'/'.join(sorted(WEIGHTED_MODELS))}. "
             "It will load ~6.6 GB of non-commercial checkpoint.",
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="generate the input with app/synth.py instead of reading Parquet")
    parser.add_argument("--bars", type=int, default=2000, help="synthetic bars per symbol")
    parser.add_argument("--seed", type=int, default=20260929, help="synthetic generator seed")
    parser.add_argument("--start-price", type=float, default=DEFAULT_START_PRICE,
                        help="synthetic first price")
    parser.add_argument(
        "--feature-mode", choices=("rebuild", "incremental"), default="rebuild",
        help="rebuild (default, O(n^2), auditable) or incremental (O(n), identical numbers)",
    )
    parser.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS,
                        help=f"refuse a run longer than this per symbol "
                             f"(default: {DEFAULT_MAX_BARS})")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="write the full machine-readable report here")
    parser.add_argument("--log-level", default=None,
                        choices=("debug", "info", "warn", "error"))
    parser.add_argument("--max-trades-shown", type=int, default=DEFAULT_MAX_TRADES_SHOWN,
                        help="per-trade rows to print; --json always carries every trade")
    return parser


def parse_day(value: str | None, label: str) -> datetime | None:
    """Parse `YYYY-MM-DD` into midnight IST on that day."""
    if value is None:
        return None
    try:
        day = date.fromisoformat(value)
    except ValueError as err:
        raise BacktestError(f"--{label} {value!r} is not a YYYY-MM-DD date") from err
    return datetime(day.year, day.month, day.day, tzinfo=trading_zone("Asia/Kolkata"))


def parse_day_end(value: str | None) -> datetime | None:
    """The last instant of the day named by `--to`, IST: 23:59:59.999999."""
    start = parse_day(value, "to")
    return None if start is None else start + timedelta(days=1) - timedelta(microseconds=1)


def plan_model(args: argparse.Namespace, config: Config) -> str:
    """Decide the model, refusing a weighted one that was not asked for."""
    selector = (args.model or DEFAULT_MODEL).strip().lower()
    if selector in WEIGHTED_MODELS and not args.allow_weights:
        raise BacktestError(
            f"refusing to build model {selector!r}: it loads the TabFM checkpoint, ~6.6 GB, "
            "licensed non-commercial and non-production (Plan.md section 5). Re-run with "
            "--allow-weights if you have the memory and the licence, or use --model heuristic, "
            "which needs no weights and no licence."
        )
    if selector in WEIGHTED_MODELS:
        logger.warning("%s", WEIGHTS_WARNING)
    if config.inference.model in WEIGHTED_MODELS and selector != config.inference.model:
        logger.info(
            "config says inference.model: %s; running %s. The backtest does not follow that key "
            "by default — see the module docstring.",
            config.inference.model,
            selector,
        )
    return selector


def _synthetic_input(
    symbols: Sequence[str], args: argparse.Namespace, zone: str
) -> dict[str, list[Candle]]:
    if args.bars < 0:
        raise BacktestError(f"--bars must not be negative, got {args.bars}")
    logger.info(
        "synthetic input: seed=%d, %d bars per symbol — this proves the plumbing, not an edge",
        args.seed,
        args.bars,
    )
    spec = SynthSpec(bars=args.bars, seed=args.seed, start_price=args.start_price)
    return generate_many(symbols, spec, zone_name=zone)


async def _main_async(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    configure_logging(args.log_level or config.app.log_level)
    limits = load_execution_limits(args.config)

    symbols = [s.strip().upper() for s in (args.symbols or config.inference.instruments)]
    if not symbols:
        raise BacktestError(
            "no instruments to run: set inference.instruments in the config, or pass --symbol"
        )
    selector = plan_model(args, config)


    start = parse_day(args.start, "from")
    end = parse_day_end(args.end)
    if start is not None and end is not None and start > end:
        raise BacktestError(f"--from {args.start} is after --to {args.end}")
    if args.bars < 0 and args.synthetic:
        raise BacktestError(f"--bars must not be negative, got {args.bars}")
    if args.max_bars < 1:
        raise BacktestError(f"--max-bars must be at least 1, got {args.max_bars}")

    predictor = Predictor.from_name(selector)
    try:
        await predictor.load()
        logger.info("model loaded: selector=%s name=%s", selector, predictor.name)

        synthetic = bool(args.synthetic)
        if synthetic:
            raw = _synthetic_input(symbols, args, config.app.timezone)
        else:
            store = DuckDBCandleStore(
                candles_root(config.storage.data_dir), max_rows=args.max_bars + 1,
            )
            try:


                raw = await load_candles(store, symbols, max(args.max_bars, 1) + 1)
            finally:
                await store.aclose()
            if not raw:
                raise BacktestError(
                    f"no candles under {config.storage.data_dir}/candles for {', '.join(symbols)}. "
                    "Run ingestion to fill the store, or pass --synthetic to generate a series."
                )

        if start is not None or end is not None:
            windowed = {s: window_candles(c, start, end) for s, c in raw.items()}
            for symbol in sorted(s for s, c in windowed.items() if not c):
                logger.warning("%s: no bars in the requested window", symbol)
            raw = {s: c for s, c in windowed.items() if c}
            if not raw:
                raise BacktestError(f"no bars between {args.start} and {args.end} for any symbol")

        result = await run_backtest(
            raw,
            config=config,
            limits=limits,
            predictor=predictor,
            mode=args.feature_mode,
            max_bars=args.max_bars,
            synthetic=synthetic,
            seed=args.seed if synthetic else None,
            data_source="synthetic" if synthetic else "parquet",
        )
    finally:
        await predictor.aclose()

    logger.info("\n%s", format_summary(result, max_trades_shown=args.max_trades_shown))

    if args.json_path:
        path = Path(args.json_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result.to_json(), indent=2, allow_nan=False) + "\n")
        except OSError as err:
            raise BacktestError(f"cannot write {path}: {err}") from err
        logger.info("wrote %s (synthetic=%s)", path, result.synthetic)

    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m app.backtest`, and for the tests."""
    args = build_parser().parse_args(argv)

    configure_logging(args.log_level or "info")
    try:
        return asyncio.run(_main_async(args))
    except (BacktestError, ConfigError) as err:
        logger.error("backtest refused: %s", err)
        return EXIT_REFUSED
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("interrupted")
        return 130
