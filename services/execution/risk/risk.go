// Package risk implements the MFT risk engine: the ordered chain of checks that every
// trading signal must pass before an order reaches the broker, and the exposure book
// those checks are measured against.
package risk

import (
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
)

// TTLStore is the slice of the fluxKV cache the risk engine depends on.
type TTLStore interface {
	// Get returns the value stored under key, reporting false when the key is missing or
	// expired.
	Get(key string) (any, bool)
	// Set stores value under key for ttl.
	Set(key string, value any, ttl time.Duration)
}

// Policy is the risk budget for one engine: the shared contracts.Policy plus the three
// inputs the shared contract does not carry — per-symbol lot sizes, the holiday
// calendar, and the clock.
type Policy struct {
	contracts.Policy

	// LotSizes maps symbol to its exchange lot size.
	LotSizes map[string]int

	// Calendar reports NSE trading holidays.
	Calendar Calendar

	// Clock supplies the current time.
	Clock Clock
}

// FromConfig builds a Policy from the shared execution configuration.
func FromConfig(cfg config.ExecutionConfig) Policy {
	return Policy{
		Policy: contracts.Policy{
			Capital:          cfg.Capital,
			MaxPositionPct:   cfg.MaxPositionPct,
			MaxOpenPositions: cfg.MaxOpenPositions,
			MaxDrawdownPct:   cfg.MaxDrawdownPct,
			DailyLossLimit:   cfg.DailyLossLimit,
			MaxOrderQuantity: cfg.MaxOrderQuantity,
			DebounceTTL:      time.Duration(cfg.DebounceTTLSeconds) * time.Second,
		},
		Clock:    SystemClock,
		Calendar: NewStaticCalendar(cfg.MarketHolidays...),
	}
}

// New composes the seven checks in the order fixed by docs/contracts.md §6.
func (p Policy) New(store TTLStore) contracts.Checks {
	return contracts.Checks{
		MaxPositionCheck{MaxPct: p.MaxPositionPct},
		MaxPositionsCheck{Max: p.MaxOpenPositions},
		DrawdownCheck{MaxPct: p.MaxDrawdownPct},
		DailyLossCheck{Limit: p.DailyLossLimit},
		DebounceCheck{Store: store, TTL: p.DebounceTTL},
		QuantityCheck{MaxQuantity: p.MaxOrderQuantity, LotSizes: p.LotSizes},
		MarketHoursCheck{Clock: p.Clock, Calendar: p.Calendar},
	}
}

// Equity returns marked account value, falling back to actual cash for a flat legacy
// portfolio.
func Equity(p contracts.Portfolio) float64 {
	if p.Equity != 0 || p.PositionValues != nil {
		return p.Equity
	}
	return p.Cash
}
