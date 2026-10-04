"""Typed assumptions for reproducible baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from datetime import date
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    """Per-leg cost assumptions; taxes and broker product stay distinct."""

    schedule: str = "generic_assumptions"
    effective_date: str | None = None
    source_url: str | None = None
    brokerage_bps: float = 3.0
    brokerage_cap: float | None = None
    stt_buy_bps: float = 0.0
    stt_sell_bps: float = 0.0
    exchange_bps: float = 0.0
    sebi_bps: float = 0.0
    ipft_bps: float = 0.0
    stamp_buy_bps: float = 0.0
    gst_pct: float = 0.0
    dp_sell_fee: float = 0.0
    spread_bps: float = 2.0
    slippage_bps: float = 2.0
    fixed_per_order: float = 0.0
    omitted_costs: tuple[str, ...] = ("IPFT is excluded unless ipft_bps is configured",)

    def __post_init__(self) -> None:
        rates = (
            self.brokerage_bps, self.stt_buy_bps, self.stt_sell_bps, self.exchange_bps,
            self.sebi_bps, self.ipft_bps, self.stamp_buy_bps, self.gst_pct, self.dp_sell_fee,
            self.spread_bps, self.slippage_bps, self.fixed_per_order,
        )
        if any(not math.isfinite(value) or value < 0 for value in rates) or (
            self.brokerage_cap is not None
            and (not math.isfinite(self.brokerage_cap) or self.brokerage_cap < 0)
        ):
            raise ValueError("cost assumptions must be finite and nonnegative")
        if not self.schedule.strip():
            raise ValueError("cost schedule must be nonempty")

    @classmethod
    def zerodha_equity_mis(cls, *, spread_bps: float = 2.0, slippage_bps: float = 2.0) -> CostAssumptions:
        """Dated NSE intraday estimate; spread/slippage remain user assumptions."""
        return cls(
            schedule="zerodha_equity_mis_nse",
            effective_date="2026-10-04",
            source_url="https://zerodha.com/charges",
            brokerage_bps=3.0,
            brokerage_cap=20.0,
            stt_sell_bps=2.5,
            exchange_bps=0.307,
            sebi_bps=0.01,
            stamp_buy_bps=0.3,
            gst_pct=18.0,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        )

    @classmethod
    def zerodha_equity_cnc(cls, *, spread_bps: float = 2.0, slippage_bps: float = 2.0) -> CostAssumptions:
        """Dated NSE delivery estimate with sell-side DP debit charge."""
        return cls(
            schedule="zerodha_equity_cnc_nse",
            effective_date="2026-10-04",
            source_url="https://zerodha.com/charges",
            brokerage_bps=0.0,
            brokerage_cap=0.0,
            stt_buy_bps=10.0,
            stt_sell_bps=10.0,
            exchange_bps=0.307,
            sebi_bps=0.01,
            stamp_buy_bps=1.5,
            gst_pct=18.0,
            dp_sell_fee=15.34,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        )

    def leg(self, notional: float, *, is_buy: bool, apply_dp_fee: bool = True) -> float:
        """Estimated currency cost for one order leg under this schedule."""
        if not math.isfinite(notional) or notional < 0.0:
            raise ValueError("order notional must be finite and nonnegative")
        brokerage = notional * self.brokerage_bps / 10_000.0
        if self.brokerage_cap is not None:
            brokerage = min(brokerage, self.brokerage_cap)
        exchange = notional * self.exchange_bps / 10_000.0
        sebi = notional * self.sebi_bps / 10_000.0
        stt = notional * (self.stt_buy_bps if is_buy else self.stt_sell_bps) / 10_000.0
        stamp = notional * self.stamp_buy_bps / 10_000.0 if is_buy else 0.0
        dp = 0.0 if is_buy or not apply_dp_fee else self.dp_sell_fee
        ipft = notional * self.ipft_bps / 10_000.0
        gst = (brokerage + exchange + sebi + ipft) * self.gst_pct / 100.0
        friction = notional * (self.spread_bps + self.slippage_bps) / 10_000.0
        return brokerage + exchange + sebi + ipft + stt + stamp + dp + gst + friction + self.fixed_per_order


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Split, prediction, execution, and cost assumptions for research."""

    horizon: int = 1
    context_rows: int = 100
    threshold: float = 0.25
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    purge_bars: int | None = None
    embargo_bars: int | None = None
    initial_capital: float = 1_000_000.0
    position_fraction: float = 0.10
    max_positions: int = 10
    long_only: bool = True
    max_position_pct: float = 10.0
    max_drawdown_pct: float = 5.0
    daily_loss_limit: float = 25_000.0
    debounce_ttl_seconds: int = 300
    holding_policy: Literal["intraday", "delivery"] = "intraday"
    strategy: str = "threshold"
    strategy_options: tuple[tuple[str, str | int | float | bool], ...] = ()
    timezone_name: str = "Asia/Kolkata"
    market_holidays: tuple[str, ...] = ()
    selection_metric: str = "net_sharpe"
    evaluator: str = "standard"
    annualization_bars: int = 94_500
    costs: CostAssumptions = field(default_factory=CostAssumptions.zerodha_equity_mis)
    max_bars_per_symbol: int = 100_000
    split_unit: Literal["bar"] = "bar"

    def __post_init__(self) -> None:
        integer_values = {
            "horizon": self.horizon,
            "context_rows": self.context_rows,
            "max_positions": self.max_positions,
            "debounce_ttl_seconds": self.debounce_ttl_seconds,
            "annualization_bars": self.annualization_bars,
            "max_bars_per_symbol": self.max_bars_per_symbol,
        }
        if self.purge_bars is not None:
            integer_values["purge_bars"] = self.purge_bars
        if self.embargo_bars is not None:
            integer_values["embargo_bars"] = self.embargo_bars
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values.values()):
            raise ValueError("bar counts, position counts, debounce, and split lengths must be finite integers")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.context_rows < 1:
            raise ValueError("context_rows must be positive")
        numeric = (
            self.threshold, self.train_fraction, self.validation_fraction,
            self.initial_capital, self.position_fraction, self.max_position_pct,
            self.max_drawdown_pct, self.daily_loss_limit,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("numeric experiment assumptions must be finite")
        if self.holding_policy not in {"intraday", "delivery"}:
            raise ValueError("holding_policy must be intraday or delivery")
        if not self.strategy.strip():
            raise ValueError("strategy name must be nonempty")
        for key, value in self.strategy_options:
            if not key.strip() or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError("strategy option names must be nonempty and numeric values finite")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if not 0.0 < self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be positive")
        if not 0.0 < self.max_position_pct <= 100.0:
            raise ValueError("max_position_pct must be in (0, 100]")
        if self.max_drawdown_pct < 0.0 or self.daily_loss_limit < 0.0 or self.debounce_ttl_seconds < 0:
            raise ValueError("drawdown, daily loss, and debounce assumptions must be nonnegative")
        try:
            ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as err:
            raise ValueError(f"unknown timezone {self.timezone_name!r}") from err
        for holiday in self.market_holidays:
            date.fromisoformat(holiday)
        if not self.long_only and self.costs.schedule == "zerodha_equity_cnc_nse":
            raise ValueError("CNC assumptions do not model intraday short positions; select MIS or custom costs")
        if self.annualization_bars < 1 or self.max_bars_per_symbol < 1:
            raise ValueError("annualization_bars and max_bars_per_symbol must be positive")
        if (self.purge_bars is not None and self.purge_bars < self.horizon) or (
            self.embargo_bars is not None and self.embargo_bars < 0
        ):
            raise ValueError("purge_bars must be at least horizon and embargo_bars nonnegative")

    @property
    def purge(self) -> int:
        """Default purge spans every overlapping label horizon."""
        return self.horizon if self.purge_bars is None else self.purge_bars

    @property
    def embargo(self) -> int:
        """Default post-boundary embargo spans one label horizon."""
        return self.horizon if self.embargo_bars is None else self.embargo_bars

    def to_json(self) -> dict[str, object]:
        """Return stable, JSON-compatible configuration data."""
        value = asdict(self)
        value["purge_bars"] = self.purge
        value["embargo_bars"] = self.embargo
        return value

    def canonical_json(self) -> str:
        """Stable compact encoding used to fingerprint an experiment."""
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), allow_nan=False)
