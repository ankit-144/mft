// Package risk implements the MFT risk engine: the ordered chain of checks
// that every trading signal must pass before an order reaches the broker, and
// the exposure book those checks are measured against.
//
// # Order
//
// docs/contracts.md §6 fixes the order and it is load-bearing: a cheap,
// hard bound (quantity, session) must not be reported when the reason the
// signal was refused is that the account is already at its loss limit. Policy
// composes the chain through contracts.Checks, so the order is one list
// literal and is asserted by test. The first failure rejects the signal with
// a *contracts.Rejection carrying a Reason* code; no broker call is made.
//
// # Portfolio arithmetic
//
// contracts.Portfolio is frozen and carries no per-position cost basis, so the
// engine derives account equity as Cash + RealisedPnL and maintains cash with
// two rules that make that identity exact:
//
//   - opening a position moves entry notional out of Cash and into the
//     position, so equity is unchanged by the trade itself;
//   - closing one returns the entry cost to Cash and books the difference
//     against RealisedPnL, exactly once.
//
// The consequence is that equity is exact for a flat book and carries an open
// position at its entry cost rather than its mark. The book tracks cost basis
// internally (see Book) so that a round trip realises the right PnL; surfacing
// marked equity needs a field the frozen contract does not have yet.
//
// # Market holidays
//
// The session check uses an injected Calendar. A nil or empty calendar means
// "assume every weekday is a trading day": it never calls out to the network
// and never blocks a signal on a stale local file. NSE publishes a holiday
// list annually, so an empty calendar is a real operational gap, not a
// modelling nicety — see NewStaticCalendar for how to close it without a
// config key.
package risk

import (
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
)

// TTLStore is the slice of the fluxKV cache the risk engine depends on.
// fluxkv.KV satisfies it. The interface keeps this package independent of the
// cache implementation and lets tests substitute a store that is already
// primed with the state under test.
type TTLStore interface {
	// Get returns the value stored under key, reporting false when the key is
	// missing or expired.
	Get(key string) (any, bool)
	// Set stores value under key for ttl.
	Set(key string, value any, ttl time.Duration)
}

// Policy is the risk budget for one engine: the frozen contracts.Policy plus
// the three inputs the frozen contract does not carry — per-symbol lot sizes,
// the holiday calendar, and the clock.
//
// A zero cap means the cap is not enforced. config.Validate fills in every
// execution default except daily_loss_limit, so in production only that one
// can legitimately arrive as zero, where it reads as "no daily loss cap".
type Policy struct {
	contracts.Policy

	// LotSizes maps symbol to its exchange lot size. A symbol that is absent
	// is treated as lot size 1, which is correct for NSE cash equity delivery.
	// The frozen config schema has no lot-size key, so F&O symbols must be
	// registered here by the caller.
	LotSizes map[string]int

	// Calendar reports NSE trading holidays. nil means "assume every weekday
	// trades"; see the package doc.
	Calendar Calendar

	// Clock supplies the current time. nil means wall clock.
	Clock Clock
}

// FromConfig builds a Policy from the frozen execution configuration. The
// clock defaults to SystemClock; LotSizes and Calendar are left for the
// caller to set.
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
		Clock: SystemClock,
	}
}

// New composes the seven checks in the order fixed by docs/contracts.md §6.
// The returned chain is a value: build it once at startup and share it.
//
// The order is exposure-first (how much room is left), then hygiene (was this
// just said, is the order well-formed), then the session itself. Debounce sits
// before quantity so that a repeated signal is reported as a repeat rather
// than as whatever happens to be wrong with its size.
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

// Equity returns the account value that every exposure check is measured
// against: free cash plus cumulative realised PnL. It is negative or zero only
// if the account is blown, which the exposure checks treat as a refusal
// rather than as an infinite limit.
func Equity(p contracts.Portfolio) float64 {
	return p.Cash + p.RealisedPnL
}
