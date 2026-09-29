package risk

import (
	"fmt"
	"sync"

	"github.com/mft/core/contracts"
)

// Book is the risk engine's running exposure state: the contracts.Portfolio
// handed to every check, plus the per-symbol cost basis that the frozen
// Portfolio has no field for.
//
// # Cash is total account cash
//
// Snapshot reports Cash as the account's whole cash balance — free cash plus
// the notional still deployed in open positions — because that is the only
// reading under which the frozen Portfolio can express equity. Equity is
// Cash + RealisedPnL, and cash that is sitting in an open position has not
// left the account. Reporting free cash alone would make every buy look like a
// drawdown the instant it filled, and the drawdown check would refuse the
// second order of the day.
//
// Free cash is tracked internally, and Book.Apply refuses a buy it cannot fund
// rather than letting cash go negative. That is the affordability backstop;
// the exposure bound itself is the contract's own: max_position_pct against
// equity times max_open_positions distinct symbols, which is 100% of equity
// under the default configuration.
//
// # Fill accounting
//
//   - BUY:  free cash -= quantity*price; the position's average cost is
//     reweighted. Equity is unchanged by the trade itself.
//   - SELL: free cash += closedQuantity*averageCost, so the entry cost comes
//     back; realisedPnL += (price-averageCost)*closedQuantity. The sale
//     proceeds are split this way deliberately. Crediting cash with the full
//     proceeds *and* booking the difference against realised PnL would
//     double-count the profit.
//
// Book is safe for concurrent use; the execution engine additionally
// serialises whole signals around it, so Apply and Snapshot are normally
// called one at a time.
type Book struct {
	mu       sync.Mutex
	cash     float64
	realised float64
	peak     float64
	held     map[string]int
	cost     map[string]float64
}

// NewBook returns an empty book funded with capital. A non-positive capital
// starts a book with zero equity, which every exposure check refuses — the
// engine logs a warning for that configuration rather than inventing a
// default.
func NewBook(capital float64) *Book {
	return &Book{
		cash: capital,
		peak: capital,
		held: make(map[string]int),
		cost: make(map[string]float64),
	}
}

// Snapshot returns the current portfolio. Cash is total account cash, as
// described on Book, and OpenPositions is a copy — a caller can mutate
// neither the book's arithmetic nor its state through the result.
func (b *Book) Snapshot() contracts.Portfolio {
	b.mu.Lock()
	defer b.mu.Unlock()

	positions := make(map[string]int, len(b.held))
	deployed := 0.0
	for symbol, qty := range b.held {
		if qty == 0 {
			continue
		}
		positions[symbol] = qty
		deployed += float64(qty) * b.cost[symbol]
	}
	return contracts.Portfolio{
		Cash:          b.cash + deployed,
		RealisedPnL:   b.realised,
		PeakEquity:    b.peak,
		OpenPositions: positions,
	}
}

// FreeCash returns the cash not committed to an open position. It is the
// engine's affordability figure; the risk checks see total cash, because
// contracts.Portfolio cannot carry both.
func (b *Book) FreeCash() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cash
}

// Apply records a filled order against the book. It fails closed: an order it
// cannot represent — a short, a nonsensical price, an unaffordable buy — is
// refused with an error and the book is left untouched. Apply is called after
// the broker has filled the order, so a silently dropped fill would corrupt
// every subsequent limit; the engine logs that case as an error.
func (b *Book) Apply(sig contracts.Signal) error {
	if sig.Price <= 0 {
		return fmt.Errorf("book: cannot apply %s %s at price %.4f", sig.Symbol, sig.Side, sig.Price)
	}
	if sig.Quantity < 1 {
		return fmt.Errorf("book: cannot apply %s %s of quantity %d", sig.Symbol, sig.Side, sig.Quantity)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	switch sig.Side {
	case contracts.SideBuy:
		notional := float64(sig.Quantity) * sig.Price
		if notional > b.cash {
			return fmt.Errorf("book: buy of %.0f exceeds free cash %.0f", notional, b.cash)
		}
		held := b.held[sig.Symbol]
		b.cost[sig.Symbol] = (float64(held)*b.cost[sig.Symbol] + notional) / float64(held+sig.Quantity)
		b.held[sig.Symbol] = held + sig.Quantity
		b.cash -= notional

	case contracts.SideSell:
		held := b.held[sig.Symbol]
		if sig.Quantity > held {
			return fmt.Errorf("book: sell of %d exceeds the long position of %d in %s", sig.Quantity, held, sig.Symbol)
		}
		avg := b.cost[sig.Symbol]
		closed := float64(sig.Quantity)
		b.cash += closed * avg
		b.realised += (sig.Price - avg) * closed
		b.held[sig.Symbol] = held - sig.Quantity
		if b.held[sig.Symbol] == 0 {
			delete(b.held, sig.Symbol)
			delete(b.cost, sig.Symbol)
		}

	default:
		return fmt.Errorf("book: unknown side %q", sig.Side)
	}

	if equity := b.cash + b.deployedLocked() + b.realised; equity > b.peak {
		b.peak = equity
	}
	return nil
}

// deployedLocked returns the notional committed to open positions. The caller
// must hold b.mu.
func (b *Book) deployedLocked() float64 {
	deployed := 0.0
	for symbol, qty := range b.held {
		deployed += float64(qty) * b.cost[symbol]
	}
	return deployed
}
