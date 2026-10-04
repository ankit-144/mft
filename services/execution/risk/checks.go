package risk

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/mft/core/contracts"
)

// MaxPositionCheck enforces a per-symbol marked exposure cap and available-cash bound.
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
	current := portfolio.PositionValues[sig.Symbol] + portfolio.ReservedValues[sig.Symbol]
	if sig.Side == contracts.SideBuy && current+notional > limit {
		return contracts.Reject(contracts.ReasonMaxPosition,
			fmt.Sprintf("position value %.0f including order %.0f exceeds %.2f%% of equity %.0f", current, notional, c.MaxPct, equity))
	}
	if sig.Side == contracts.SideBuy && notional > portfolio.Cash-portfolio.ReservedCash {
		return contracts.Reject(contracts.ReasonMaxPosition,
			fmt.Sprintf("order value %.0f exceeds available cash %.0f", notional, portfolio.Cash-portfolio.ReservedCash))
	}
	return nil
}

// MaxPositionsCheck enforces the open-position count.
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
	if portfolio.ReservedPositions[sig.Symbol] > 0 && held == 0 {
		prospective++
	}
	if held == 0 && portfolio.ReservedPositions[sig.Symbol] == 0 {
		prospective++
	}
	if prospective > c.Max {
		return contracts.Reject(contracts.ReasonMaxPositions,
			fmt.Sprintf("opening %s would hold %d positions, limit is %d", sig.Symbol, prospective, c.Max))
	}
	return nil
}

// DrawdownCheck enforces (peak - equity) / peak <= max_drawdown_pct.
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

// DailyLossCheck enforces realised PnL >= -daily_loss_limit.
type DailyLossCheck struct {
	Limit float64
}

// Check implements contracts.Checker.
func (c DailyLossCheck) Check(_ context.Context, _ contracts.Signal, portfolio contracts.Portfolio) error {
	if c.Limit <= 0 {
		return nil
	}
	if portfolio.RealisedPnLToday < -c.Limit {
		return contracts.Reject(contracts.ReasonDailyLoss,
			fmt.Sprintf("today's realised PnL %.2f breaches the daily loss limit of %.2f", portfolio.RealisedPnLToday, c.Limit))
	}
	return nil
}

// DebounceKey returns the fluxKV key that arms the debounce for one symbol/side pair,
// as fixed by docs/contracts.md §6.
func DebounceKey(symbol, side string) string {
	return fmt.Sprintf("EXEC:%s:%s", symbol, side)
}

// DebounceCheck refuses a symbol/side pair already claimed during the TTL window.
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

// QuantityCheck enforces 1 <= quantity <= max_order_quantity, that quantity is a whole
// multiple of the symbol's lot size, and that the engine holds a long position large
// enough to sell.
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
		held := portfolio.OpenPositions[sig.Symbol] - portfolio.ReservedSells[sig.Symbol]
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

// ValidateSignal reports whether a signal is well-formed enough to be risk checked at
// all: a symbol, a known side, a positive reference price, and a non-empty idempotency
// key.
func ValidateSignal(sig contracts.Signal) error {
	if strings.TrimSpace(sig.Symbol) == "" {
		return fmt.Errorf("risk: signal has no symbol")
	}
	switch sig.Side {
	case contracts.SideBuy, contracts.SideSell:
	default:
		return fmt.Errorf("risk: signal side %q must be %s or %s", sig.Side, contracts.SideBuy, contracts.SideSell)
	}
	if sig.Price <= 0 || math.IsNaN(sig.Price) || math.IsInf(sig.Price, 0) {
		return fmt.Errorf("risk: signal price %.4f must be positive", sig.Price)
	}
	if math.IsNaN(sig.Score) || math.IsInf(sig.Score, 0) {
		return fmt.Errorf("risk: signal score must be finite")
	}
	if strings.TrimSpace(sig.IdempotencyKey) == "" {
		return fmt.Errorf("risk: signal %s has no idempotency key", sig.Symbol)
	}
	if sig.AsOf.IsZero() {
		return fmt.Errorf("risk: signal %s has no timestamp", sig.Symbol)
	}
	return nil
}
