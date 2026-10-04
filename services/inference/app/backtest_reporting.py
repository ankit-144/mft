"""Human-readable backtest report rendering, separated from replay mechanics."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from .backtest import (
    BARS_PER_YEAR, CHARGE_DISCLAIMER, DEFAULT_MAX_TRADES_SHOWN, MIN_CONTEXT_ROWS,
    POSITION_MODEL_NOTE, REASON_CHECK, RISK_DRIFT_NOTES, RULE, THIN, BacktestResult,
    Reason, WARMUP_ROWS, rfc3339,
)

RULE: Final[str] = "=" * 78
THIN: Final[str] = "-" * 78


def format_summary(
    result: BacktestResult, *, max_trades_shown: int = DEFAULT_MAX_TRADES_SHOWN
) -> str:
    """Render the human report."""
    metrics = result.metrics
    lines: list[str] = []
    if result.synthetic:
        lines += [RULE, "MFT BACKTEST — SYNTHETIC INPUT. THIS IS NOT A STRATEGY RESULT.", RULE]
    else:
        lines += [RULE, "MFT BACKTEST — STORED CANDLES", RULE]
    lines += [
        "",
        _wrap(result.disclaimer),
        "",
        _wrap(CHARGE_DISCLAIMER),
        "",
        _wrap(POSITION_MODEL_NOTE),
        "",
        "RUN",
        THIN,
        f"  model              {result.model_selector} ({result.model_name})",
        f"  input              {result.data_source}"
        + (f", seed {result.seed}, SYNTHETIC" if result.synthetic else ", from Parquet"),
        f"  symbols            {', '.join(result.symbols) or '<none>'}",
        f"  bars               {_counts(result.bars_per_symbol)}",
        f"  horizon            {result.horizon} bar(s); "
        f"label = candles[i + {result.horizon}].close",
        f"  context rows       {result.context_rows} (after the {WARMUP_ROWS}-row warm-up)",
        f"  score threshold    {result.threshold:.4f}",
        f"  order quantity     {result.order_quantity} shares",
        f"  capital            Rs {result.limits.capital:,.2f}",
        f"  walk-forward       {result.feature_mode} — "
        + (
            "O(n^2) per symbol, one feature build per bar"
            if result.feature_mode == "rebuild"
            else "O(n) per symbol, identical numbers (proven in test_backtest.py)"
        ),
        f"  scored bars        {result.timing.bars_scored} "
        f"(first decision index {WARMUP_ROWS + MIN_CONTEXT_ROWS + result.horizon - 1})",
        f"  decision window    {_span(result.first_decision, result.last_decision)}",
        "",
        "RESULTS (net of charges unless the line says gross)",
        THIN,
        f"  trades             {metrics.trades}  ({metrics.wins} win / {metrics.losses} loss)",
        f"  hit rate           {_pct(metrics.hit_rate)}",
        f"  return, gross      {_pct(metrics.gross_return)}  (before charges, for comparison only)",
        f"  return, net        {_pct(metrics.net_return)}  "
        f"(Rs {metrics.net_pnl:+,.2f} on Rs {result.limits.capital:,.0f})",
        f"  per trade          mean {_pct(metrics.mean_trade_return)}  "
        f"median {_pct(metrics.median_trade_return)}  "
        f"best {_pct(metrics.best_trade_return)}  worst {_pct(metrics.worst_trade_return)}",
        f"  Sharpe, annualised {metrics.sharpe:.3f}   (per bar {metrics.sharpe_per_bar:+.4f}, "
        f"x sqrt({metrics.bars_per_year}))",
        f"                     annualisation is 375 bars/session x 252 sessions. A 1-minute",
        f"                     strategy's figure is enormous by construction; read the",
        f"                     per-bar number to see what the annualisation did.",
        f"  max drawdown       {metrics.max_drawdown_pct:.2f}%  "
        f"(limit {result.limits.max_drawdown_pct:.2f}%)"
        + ("  ← THE GATE HALTED TRADING" if result.drawdown_halted else ""),
        f"  turnover           {metrics.turnover:.2f}x capital  "
        f"(Rs {metrics.notional:,.0f} traded notional)",
        f"  exposure           {metrics.exposure_pct * 100.0:.2f}% of the timeline "
        "with a position open",
        f"  charges            Rs {metrics.charges:,.2f} total, "
        f"Rs {metrics.charges_per_trade:,.2f} per trade",
        f"  final equity       Rs {result.limits.capital + metrics.net_pnl:,.2f} "
        f"(capital + the sum of {metrics.trades} trade P&Ls)",
        "",
        "COST OF THE MODEL ITSELF",
        THIN,
        f"  wall time          {result.timing.total_seconds:.2f}s total",
        f"  model              {result.timing.model_seconds:.3f}s over "
        f"{result.timing.bars_scored} bars "
        f"= {result.timing.model_ms_per_bar:.3f} ms/bar",
        f"  features           {result.timing.feature_seconds:.3f}s = "
        f"{result.timing.feature_ms_per_bar:.3f} ms/bar  "
        + (
            "(walk-forward rebuild only; the live loop builds one bounded window a minute)"
            if result.feature_mode == "rebuild"
            else "(one build for the whole series)"
        ),
        f"  budget             a 1-minute cadence allows 60000 ms/bar; this run used "
        f"{result.timing.model_ms_per_bar + result.timing.feature_ms_per_bar:.3f} ms/bar",
        "",
        "PER SYMBOL",
        THIN,
        f"  {'symbol':<11}{'bars':>6}{'cand':>6}{'trades':>8}{'hit':>8}{'gross P&L':>12}"
        f"{'charges':>11}{'net P&L':>12}  reasons",
    ]
    for row in result.per_symbol:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(row.rejections.items())) or "-"
        lines.append(
            f"  {row.symbol:<11}{row.bars:>6}{row.candidates:>6}{row.trades:>8}"
            f"{row.hit_rate * 100.0:>7.1f}%{row.gross_pnl:>+12,.2f}{row.charges:>11,.2f}"
            f"{row.net_pnl:>+12,.2f}  {reasons}"
        )

    lines += ["", "PER-BAR OUTCOMES (before the risk gate)"]
    lines.append(
        "  "
        + (
            ", ".join(f"{k}={v}" for k, v in result.outcomes.items())
            if result.outcomes
            else "nothing"
        )
    )

    lines += ["", "RISK GATE — docs/contracts.md §6, in the contract's own order"]
    if result.rejections:
        ordered = sorted(
            result.rejections.items(),
            key=lambda kv: (REASON_CHECK.get(Reason(kv[0]), 99), kv[0]),
        )
        for code, count in ordered:
            lines.append(f"  {code:<22} {count:>7}   (check {REASON_CHECK[Reason(code)]})")
    else:
        lines.append("  nothing was rejected")
    lines += ["", "  these are the same reason codes the execution service returns"]
    for note in RISK_DRIFT_NOTES:
        lines.append(_wrap(f"- {note}", indent="    "))

    lines += ["", f"PER-TRADE TABLE ({len(result.trades)} trades, net of charges)"]
    if not result.trades:
        lines.append("  no trade was filled")
    else:
        lines.append(
            f"  {'entry':<21}{'exit':<21}{'sym':<10}{'side':<5}{'qty':>4}{'score':>8}"
            f"{'entry px':>11}{'exit px':>11}{'gross':>11}{'charges':>10}{'net':>11}{'net%':>9}"
        )
        for trade in result.trades[:max_trades_shown]:
            lines.append(
                f"  {rfc3339(trade.entry_as_of):<21}{rfc3339(trade.exit_as_of):<21}"
                f"{trade.symbol:<10}{trade.side:<5}{trade.quantity:>4}{trade.score:>+8.4f}"
                f"{trade.entry_price:>11.2f}{trade.exit_price:>11.2f}{trade.gross_pnl:>+11.2f}"
                f"{trade.charges:>10.2f}{trade.net_pnl:>+11.2f}{trade.net_return * 100.0:>+8.3f}%"
            )
        if len(result.trades) > max_trades_shown:
            lines.append(
                f"  ... {len(result.trades) - max_trades_shown} more; --json writes every trade"
            )

    lines += [
        "",
        "WHAT THIS RUN DOES NOT TELL YOU",
        THIN,
        "  * Out-of-sample performance. This is one history, scored once, by the",
        "    same model that would see it live. There is no held-out set, no",
        "    cross-validation and no parameter fitting, so a good number here is",
        "    not evidence of a good number anywhere else.",
        "  * Whether the model has any predictive power. On synthetic data it",
        "    provably does not: the series is a random walk. A positive Sharpe",
        "    here is a property of the cost and size assumptions, not of a signal.",
        "  * Anything about live execution. No fills, no slippage, no partial",
        "    rejects, no circuit limits, no broker downtime, no square-off.",
        "  * The real risk engine. The gate above is an offline re-implementation",
        "    of the checks that make sense without a broker; see the drift notes.",
        "",
        RULE,
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int = 76, indent: str = "  ") -> str:
    """Reflow a paragraph to `width`, every line carrying `indent`."""
    lines: list[str] = []
    current: str | None = None
    for word in text.split():
        if current is None:
            current = indent + word
        elif len(current) + 1 + len(word) > width:
            lines.append(current)
            current = indent + word
        else:
            current = f"{current} {word}"
    lines.append(current or indent.rstrip())
    return "\n".join(lines)


def _pct(fraction: float) -> str:
    return f"{fraction * 100.0:+.2f}%"


def _counts(bars_per_symbol: dict[str, int]) -> str:
    if not bars_per_symbol:
        return "<none>"
    return ", ".join(f"{symbol}={count}" for symbol, count in sorted(bars_per_symbol.items()))


def _span(first: datetime | None, last: datetime | None) -> str:
    if first is None or last is None:
        return "<no bar passed the threshold>"
    if first == last:
        return rfc3339(first)
    return f"{rfc3339(first)} .. {rfc3339(last)}"
