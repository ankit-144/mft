package risk

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/mft/core/contracts"
)

// BookState is the durable accounting snapshot needed to rebuild a Book.
type BookState struct {
	Cash          float64              `json:"cash"`
	RealisedPnL   float64              `json:"realised_pnl"`
	RealisedByDay map[string]float64   `json:"realised_by_day"`
	PeakEquity    float64              `json:"peak_equity"`
	Held          map[string]int       `json:"held"`
	Cost          map[string]float64   `json:"cost"`
	Marks         map[string]float64   `json:"marks"`
	MarksAt       map[string]time.Time `json:"marks_at"`
}

// Book keeps actual cash, filled positions and their current marks.
type Book struct {
	mu          sync.RWMutex
	cash        float64
	realised    float64
	realisedDay map[string]float64
	peak        float64
	held        map[string]int
	cost        map[string]float64
	marks       map[string]float64
	marksAt     map[string]time.Time
}

// NewBook creates a flat account funded with capital.
func NewBook(capital float64) *Book {
	return &Book{
		cash: capital, peak: capital,
		realisedDay: make(map[string]float64),
		held:        make(map[string]int), cost: make(map[string]float64),
		marks: make(map[string]float64), marksAt: make(map[string]time.Time),
	}
}

// RestoreBook rebuilds a book from a previously persisted snapshot.
func RestoreBook(state BookState) (*Book, error) {
	if !finite(state.Cash) || !finite(state.RealisedPnL) || !finite(state.PeakEquity) || state.PeakEquity < 0 {
		return nil, fmt.Errorf("risk: invalid persisted peak equity")
	}
	b := &Book{
		cash: state.Cash, realised: state.RealisedPnL, peak: state.PeakEquity,
		realisedDay: cloneMap(state.RealisedByDay), held: cloneMap(state.Held),
		cost: cloneMap(state.Cost), marks: cloneMap(state.Marks), marksAt: cloneMap(state.MarksAt),
	}
	if b.realisedDay == nil {
		b.realisedDay = make(map[string]float64)
	}
	if b.held == nil {
		b.held = make(map[string]int)
	}
	if b.cost == nil {
		b.cost = make(map[string]float64)
	}
	if b.marks == nil {
		b.marks = make(map[string]float64)
	}
	if b.marksAt == nil {
		b.marksAt = make(map[string]time.Time)
	}
	for day, pnl := range b.realisedDay {
		if day == "" || !finite(pnl) {
			return nil, fmt.Errorf("risk: invalid persisted daily PnL")
		}
	}
	for symbol, qty := range b.held {
		if symbol == "" || qty <= 0 || b.cost[symbol] <= 0 || !finite(b.cost[symbol]) ||
			(b.marks[symbol] != 0 && (!finite(b.marks[symbol]) || b.marks[symbol] <= 0)) {
			return nil, fmt.Errorf("risk: invalid persisted position %q", symbol)
		}
		if b.marks[symbol] <= 0 {
			b.marks[symbol] = b.cost[symbol]
		}
	}
	return b, nil
}

// State returns a detached accounting snapshot for durable persistence.
func (b *Book) State() BookState {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return BookState{
		Cash: b.cash, RealisedPnL: b.realised, RealisedByDay: cloneMap(b.realisedDay),
		PeakEquity: b.peak, Held: cloneMap(b.held), Cost: cloneMap(b.cost),
		Marks: cloneMap(b.marks), MarksAt: cloneMap(b.marksAt),
	}
}

// Snapshot returns a copied portfolio measured using current marks.
func (b *Book) Snapshot() contracts.Portfolio { return b.SnapshotAt(time.Now()) }

// SnapshotAt includes the realized PnL for the supplied IST trading date.
func (b *Book) SnapshotAt(now time.Time) contracts.Portfolio {
	b.mu.RLock()
	defer b.mu.RUnlock()
	positions := cloneMap(b.held)
	values := make(map[string]float64, len(b.held))
	equity := b.cash
	for symbol, qty := range b.held {
		mark := b.marks[symbol]
		if mark <= 0 {
			mark = b.cost[symbol]
		}
		values[symbol] = float64(qty) * mark
		equity += values[symbol]
	}
	return contracts.Portfolio{
		Cash: b.cash, Equity: equity, RealisedPnL: b.realised,
		RealisedPnLToday: b.realisedDay[now.In(IST()).Format(HolidayLayout)],
		PeakEquity:       b.peak, OpenPositions: positions, PositionValues: values,
	}
}

// FreeCash returns the spendable cash after filled trades and realized PnL.
func (b *Book) FreeCash() float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.cash
}

// Mark updates one held position's current price and mark timestamp.
func (b *Book) Mark(symbol string, price float64, at time.Time) error {
	if price <= 0 || symbol == "" || math.IsNaN(price) || math.IsInf(price, 0) {
		return fmt.Errorf("risk: invalid mark for %q at %v", symbol, price)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.held[symbol] == 0 {
		return nil
	}
	b.marks[symbol], b.marksAt[symbol] = price, at.UTC()
	b.updatePeakLocked()
	return nil
}

// MarkAge reports the oldest mark age and whether all held symbols are marked.
func (b *Book) MarkAge(now time.Time) (time.Duration, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	var oldest time.Duration
	for symbol := range b.held {
		at := b.marksAt[symbol]
		if at.IsZero() {
			return 0, false
		}
		age := now.Sub(at)
		if age > oldest {
			oldest = age
		}
	}
	return oldest, true
}

// ApplyFill accounts only the quantity and average price actually filled.
func (b *Book) ApplyFill(sig contracts.Signal, quantity int, price float64, at time.Time) error {
	if price <= 0 || quantity < 1 || math.IsNaN(price) || math.IsInf(price, 0) {
		return fmt.Errorf("risk: invalid fill %s %s x%d at %.4f", sig.Symbol, sig.Side, quantity, price)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	switch sig.Side {
	case contracts.SideBuy:
		notional := float64(quantity) * price
		held := b.held[sig.Symbol]
		b.cost[sig.Symbol] = (float64(held)*b.cost[sig.Symbol] + notional) / float64(held+quantity)
		b.held[sig.Symbol] = held + quantity
		b.cash -= notional
		b.marks[sig.Symbol], b.marksAt[sig.Symbol] = price, at.UTC()
	case contracts.SideSell:
		held := b.held[sig.Symbol]
		if quantity > held {
			return fmt.Errorf("risk: sell of %d exceeds position %d in %s", quantity, held, sig.Symbol)
		}
		avg := b.cost[sig.Symbol]
		b.cash += float64(quantity) * price
		pnl := (price - avg) * float64(quantity)
		b.realised += pnl
		day := at.In(IST()).Format(HolidayLayout)
		b.realisedDay[day] += pnl
		b.held[sig.Symbol] = held - quantity
		if b.held[sig.Symbol] == 0 {
			delete(b.held, sig.Symbol)
			delete(b.cost, sig.Symbol)
			delete(b.marks, sig.Symbol)
			delete(b.marksAt, sig.Symbol)
		} else {
			b.marks[sig.Symbol], b.marksAt[sig.Symbol] = price, at.UTC()
		}
	default:
		return fmt.Errorf("risk: unknown fill side %q", sig.Side)
	}
	b.updatePeakLocked()
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// Apply records an immediate synthetic fill at the signal reference price.
func (b *Book) Apply(sig contracts.Signal) error {
	return b.ApplyFill(sig, sig.Quantity, sig.Price, time.Now().UTC())
}

func (b *Book) updatePeakLocked() {
	equity := b.cash
	for symbol, qty := range b.held {
		mark := b.marks[symbol]
		if mark <= 0 {
			mark = b.cost[symbol]
		}
		equity += float64(qty) * mark
	}
	if equity > b.peak {
		b.peak = equity
	}
}

func cloneMap[K comparable, V any](in map[K]V) map[K]V {
	if in == nil {
		return nil
	}
	out := make(map[K]V, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
