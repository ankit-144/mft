package risk

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/mft/core/contracts"
)

// MaxPositionCheck enforces |quantity * price| <= max_position_pct% of
// equity, and refuses an order larger than the entire account. A zero MaxPct
// disables the cap.
//
// The cash bound is a sanity net, not the affordability policy: the cash in
// contracts.Portfolio is total account cash, including notional still deployed
// in open positions (see Book), so this only catches an order bigger than
// everything the account owns. The real aggregate bound is this cap times
// max_open_positions — 100% of equity under the default config — and the
// free-cash guard is Book.Apply, which refuses to record a buy it cannot fund.
type MaxPositionCheck struct {
	MaxPct float64
}

// Check implements contracts.Checker.
func (c MaxPositionCheck) Check(_ context.Context, sig contracts.Signal, portfolio contracts.Portfolio) error {
	if c.MaxPct <= 0 {
		return nil
	}
	notional := math.Abs(float64(sig.Quantity) * sig.Price)
	equity := Equity(portfolio)
	if equity <= 0 {
		return contracts.Reject(contracts.ReasonMaxPosition,
			fmt.Sprintf("equity %.2f is not positive; refusing to size an order", equity))
	}
	limit := equity * c.MaxPct / 100
	if notional > limit {
		return contracts.Reject(contracts.ReasonMaxPosition,
			fmt.Sprintf("order value %.0f exceeds %.2f%% of equity %.0f", notional, c.MaxPct, equity))
	}
	if sig.Side == contracts.SideBuy && notional > portfolio.Cash {
		return contracts.Reject(contracts.ReasonMaxPosition,
			fmt.Sprintf("order value %.0f exceeds account cash %.0f", notional, portfolio.Cash))
	}
	return nil
}

// MaxPositionsCheck enforces the open-position count. A zero Max disables the
// cap.
//
// The contract states the bound as len(OpenPositions) <= max_open_positions,
// evaluated before the order lands. Read literally that admits one more than
// the limit, because a count of exactly the limit still passes and then grows.
// This check counts the position the signal would create, so a signal for a
// symbol already held adds nothing and a signal for a new symbol must fit
// inside the limit. That is the same formula, never more permissive.
type MaxPositionsCheck struct {
	Max int
}

// Check implements contracts.Checker.
func (c MaxPositionsCheck) Check(_ context.Context, sig contracts.Signal, portfolio contracts.Portfolio) error {
	if c.Max <= 0 {
		return nil
	}
	held := portfolio.OpenPositions[sig.Symbol]
	prospective := len(portfolio.OpenPositions)
	if held == 0 {
		prospective++
	}
	if prospective > c.Max {
		return contracts.Reject(contracts.ReasonMaxPositions,
			fmt.Sprintf("opening %s would hold %d positions, limit is %d", sig.Symbol, prospective, c.Max))
	}
	return nil
}

// DrawdownCheck enforces (peak - equity) / peak <= max_drawdown_pct. A zero
// MaxPct disables the cap.
//
// A non-positive peak means the account never had a positive high-water mark;
// the ratio is undefined there, and an undefined risk limit is not a limit, so
// the signal is refused.
type DrawdownCheck struct {
	MaxPct float64
}

// Check implements contracts.Checker.
func (c DrawdownCheck) Check(_ context.Context, _ contracts.Signal, portfolio contracts.Portfolio) error {
	if c.MaxPct <= 0 {
		return nil
	}
	peak := portfolio.PeakEquity
	if peak <= 0 {
		return contracts.Reject(contracts.ReasonMaxDrawdown,
			fmt.Sprintf("peak equity %.2f is not positive; drawdown is undefined", peak))
	}
	equity := Equity(portfolio)
	drawdown := (peak - equity) / peak * 100
	if drawdown > c.MaxPct {
		return contracts.Reject(contracts.ReasonMaxDrawdown,
			fmt.Sprintf("drawdown %.2f%% from peak equity %.0f exceeds %.2f%%", drawdown, peak, c.MaxPct))
	}
	return nil
}

// DailyLossCheck enforces realised PnL >= -daily_loss_limit. A non-positive
// Limit means no daily loss cap, which is how a config that never set
// daily_loss_limit reads.
type DailyLossCheck struct {
	Limit float64
}

// Check implements contracts.Checker.
func (c DailyLossCheck) Check(_ context.Context, _ contracts.Signal, portfolio contracts.Portfolio) error {
	if c.Limit <= 0 {
		return nil
	}
	if portfolio.RealisedPnL < -c.Limit {
		return contracts.Reject(contracts.ReasonDailyLoss,
			fmt.Sprintf("realised PnL %.2f breaches the daily loss limit of %.2f", portfolio.RealisedPnL, c.Limit))
	}
	return nil
}

// DebounceKey returns the fluxKV key that arms the debounce for one
// symbol/side pair, as fixed by docs/contracts.md §6.
func DebounceKey(symbol, side string) string {
	return fmt.Sprintf("EXEC:%s:%s", symbol, side)
}

// DebounceCheck refuses a signal for a symbol/side pair that already has an
// order inside the TTL window. Reading is all this check does: the engine arms
// the key after a successful placement, so a refused signal or a failed broker
// call never starts a window it cannot honour.
type DebounceCheck struct {
	Store TTLStore
	TTL   time.Duration
}

// Check implements contracts.Checker.
func (c DebounceCheck) Check(_ context.Context, sig contracts.Signal, _ contracts.Portfolio) error {
	if c.Store == nil {
		return nil
	}
	key := DebounceKey(sig.Symbol, sig.Side)
	if _, armed := c.Store.Get(key); armed {
		return contracts.Reject(contracts.ReasonDebounced,
			fmt.Sprintf("%s already ordered within %s", key, c.TTL))
	}
	return nil
}

// QuantityCheck enforces 1 <= quantity <= max_order_quantity, that quantity is
// a whole multiple of the symbol's lot size, and that the engine holds a long
// position large enough to sell.
//
// The sell bound is a platform limit, not a margin decision: v1 trades
// long-only cash equity and a SELL beyond the holding is a short, which the
// execution engine cannot price or square. Refusing it here means it can never
// reach the broker. A zero MaxQuantity disables the upper bound only.
type QuantityCheck struct {
	MaxQuantity int
	LotSizes    map[string]int
}

// Check implements contracts.Checker.
func (c QuantityCheck) Check(_ context.Context, sig contracts.Signal, portfolio contracts.Portfolio) error {
	if sig.Quantity < 1 {
		return contracts.Reject(contracts.ReasonBadQuantity,
			fmt.Sprintf("quantity %d is not at least 1", sig.Quantity))
	}
	if c.MaxQuantity > 0 && sig.Quantity > c.MaxQuantity {
		return contracts.Reject(contracts.ReasonBadQuantity,
			fmt.Sprintf("quantity %d exceeds max_order_quantity %d", sig.Quantity, c.MaxQuantity))
	}
	lot := c.lotSize(sig.Symbol)
	if lot < 1 {
		lot = 1
	}
	if sig.Quantity%lot != 0 {
		return contracts.Reject(contracts.ReasonBadQuantity,
			fmt.Sprintf("quantity %d is not a multiple of lot size %d for %s", sig.Quantity, lot, sig.Symbol))
	}
	if sig.Side == contracts.SideSell {
		held := portfolio.OpenPositions[sig.Symbol]
		if sig.Quantity > held {
			return contracts.Reject(contracts.ReasonBadQuantity,
				fmt.Sprintf("sell %d %s exceeds the long position of %d; shorting is not supported", sig.Quantity, sig.Symbol, held))
		}
	}
	return nil
}

// lotSize returns the configured lot size for symbol, defaulting to 1 for the
// cash-equity symbols that trade in units.
func (c QuantityCheck) lotSize(symbol string) int {
	if lot, ok := c.LotSizes[symbol]; ok && lot > 0 {
		return lot
	}
	return 1
}

// ValidateSignal reports whether a signal is well-formed enough to be risk
// checked at all: a symbol, a known side, a positive reference price, and a
// non-empty idempotency key. These are transport-level faults rather than
// policy decisions, so they return a plain error and no Rejection — there is
// no reason code for a malformed request, and inventing one would put an
// undeclared value in the rejection metric.
//
// The idempotency key is required. Inference retries, and a signal without a
// key cannot be made idempotent, so the engine refuses it rather than placing
// an order it could place twice.
func ValidateSignal(sig contracts.Signal) error {
	if strings.TrimSpace(sig.Symbol) == "" {
		return fmt.Errorf("risk: signal has no symbol")
	}
	switch sig.Side {
	case contracts.SideBuy, contracts.SideSell:
	default:
		return fmt.Errorf("risk: signal side %q must be %s or %s", sig.Side, contracts.SideBuy, contracts.SideSell)
	}
	if sig.Price <= 0 {
		return fmt.Errorf("risk: signal price %.4f must be positive", sig.Price)
	}
	if strings.TrimSpace(sig.IdempotencyKey) == "" {
		return fmt.Errorf("risk: signal %s has no idempotency key", sig.Symbol)
	}
	return nil
}
