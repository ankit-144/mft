package risk

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/mft/core/contracts"
)

// fakeStore is a TTLStore that the tests prime directly. fluxkv.KV satisfies
// TTLStore in production; here the state under test is set up by hand so the
// expiry cases do not depend on a sleep.
type fakeStore struct {
	mu    sync.Mutex
	items map[string]struct{}
}

func newFakeStore(keys ...string) *fakeStore {
	s := &fakeStore{items: make(map[string]struct{}, len(keys))}
	for _, k := range keys {
		s.items[k] = struct{}{}
	}
	return s
}

func (s *fakeStore) Get(key string) (any, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.items[key]
	return true, ok
}

func (s *fakeStore) Set(key string, _ any, _ time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.items[key] = struct{}{}
}

// openTime is a Tuesday at 10:30 IST, inside the session, in UTC. Every test
// that needs "a valid moment" derives from it so a timezone mistake shows up
// as a failure rather than as a pass that only happens in IST.
var openTime = time.Date(2026, 9, 29, 5, 0, 0, 0, time.UTC) // 10:30 IST

func atIST(hour, minute int) time.Time {
	return time.Date(2026, 9, 29, hour, minute, 0, 0, IST())
}

func clockAt(t time.Time) Clock {
	return func() time.Time { return t }
}

// basePortfolio is a flat book with 100k of equity, which is the starting
// point for every boundary case below.
func basePortfolio() contracts.Portfolio {
	return contracts.Portfolio{
		Cash:          100_000,
		RealisedPnL:   0,
		PeakEquity:    100_000,
		OpenPositions: map[string]int{},
	}
}

func signal(mutators ...func(*contracts.Signal)) contracts.Signal {
	sig := contracts.Signal{
		Symbol:         "RELIANCE",
		Side:           contracts.SideBuy,
		Quantity:       10,
		Price:          1000,
		AsOf:           openTime,
		IdempotencyKey: "RELIANCE:BUY:1",
	}
	for _, m := range mutators {
		m(&sig)
	}
	return sig
}

// TestChecks exercises all seven checks in docs/contracts.md §6. Every check
// appears twice against its own boundary: once just outside it, which must
// produce the documented reason code, and once exactly on it, which must pass.
func TestChecks(t *testing.T) {
	cases := []struct {
		name      string
		check     contracts.Checker
		portfolio contracts.Portfolio
		signal    contracts.Signal
		wantCode  string // "" means the signal must pass
	}{
		// 1. max_position_pct — 10% of 100k equity is a 10,000 order value.
		{
			name:      "max_position_pct rejects above the cap",
			check:     MaxPositionCheck{MaxPct: 10},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Price = 1000.01 }),
			wantCode:  contracts.ReasonMaxPosition,
		},
		{
			name:      "max_position_pct accepts exactly the cap",
			check:     MaxPositionCheck{MaxPct: 10},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
		{
			name:  "max_position_pct rejects on non-positive equity",
			check: MaxPositionCheck{MaxPct: 10}, portfolio: contracts.Portfolio{
				Cash: 0, RealisedPnL: -1_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{},
			},
			signal:   signal(),
			wantCode: contracts.ReasonMaxPosition,
		},
		{
			name:  "max_position_pct rejects a buy larger than the whole account",
			check: MaxPositionCheck{MaxPct: 50}, portfolio: contracts.Portfolio{
				Cash: 5_000, PeakEquity: 5_000, OpenPositions: map[string]int{},
			},
			signal:   signal(func(s *contracts.Signal) { s.Quantity = 10; s.Price = 1000 }),
			wantCode: contracts.ReasonMaxPosition,
		},
		{
			name:      "max_position_pct disabled when unset",
			check:     MaxPositionCheck{},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Quantity = 100_000 }),
		},

		// 2. max_positions — a signal for a held symbol does not add a slot.
		{
			name:  "max_positions rejects a new symbol at the limit",
			check: MaxPositionsCheck{Max: 2}, portfolio: contracts.Portfolio{
				Cash: 100_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{"RELIANCE": 10, "TCS": 5},
			},
			signal:   signal(func(s *contracts.Signal) { s.Symbol = "INFY" }),
			wantCode: contracts.ReasonMaxPositions,
		},
		{
			name:  "max_positions accepts a new symbol one below the limit",
			check: MaxPositionsCheck{Max: 2}, portfolio: contracts.Portfolio{
				Cash: 100_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{"RELIANCE": 10},
			},
			signal: signal(func(s *contracts.Signal) { s.Symbol = "INFY" }),
		},
		{
			name:  "max_positions accepts a held symbol at the limit",
			check: MaxPositionsCheck{Max: 2}, portfolio: contracts.Portfolio{
				Cash: 100_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{"RELIANCE": 10, "TCS": 5},
			},
			signal: signal(),
		},
		{
			name:      "max_positions disabled when unset",
			check:     MaxPositionsCheck{},
			portfolio: basePortfolio(),
			signal:    signal(),
		},

		// 3. max_drawdown_pct — (peak - equity) / peak * 100.
		{
			name:  "max_drawdown_pct rejects one unit past the cap",
			check: DrawdownCheck{MaxPct: 5}, portfolio: contracts.Portfolio{
				Cash: 94_999, PeakEquity: 100_000, OpenPositions: map[string]int{},
			},
			signal:   signal(),
			wantCode: contracts.ReasonMaxDrawdown,
		},
		{
			name:  "max_drawdown_pct accepts exactly the cap",
			check: DrawdownCheck{MaxPct: 5}, portfolio: contracts.Portfolio{
				Cash: 95_000, PeakEquity: 100_000, OpenPositions: map[string]int{},
			},
			signal: signal(),
		},
		{
			name:  "max_drawdown_pct rejects an undefined drawdown",
			check: DrawdownCheck{MaxPct: 5}, portfolio: contracts.Portfolio{
				Cash: -1, PeakEquity: 0, OpenPositions: map[string]int{},
			},
			signal:   signal(),
			wantCode: contracts.ReasonMaxDrawdown,
		},
		{
			name:  "max_drawdown_pct is satisfied by realised gains",
			check: DrawdownCheck{MaxPct: 5}, portfolio: contracts.Portfolio{
				Cash: 90_000, RealisedPnL: 15_000, PeakEquity: 110_000, OpenPositions: map[string]int{},
			},
			signal: signal(),
		},
		{
			name:      "max_drawdown_pct disabled when unset",
			check:     DrawdownCheck{},
			portfolio: basePortfolio(),
			signal:    signal(),
		},

		// 4. daily_loss_limit — realised PnL >= -limit.
		{
			name:  "daily_loss_limit rejects one unit past the limit",
			check: DailyLossCheck{Limit: 25_000}, portfolio: contracts.Portfolio{
				Cash: 75_000, RealisedPnL: -25_000.01, PeakEquity: 100_000,
				OpenPositions: map[string]int{},
			},
			signal:   signal(),
			wantCode: contracts.ReasonDailyLoss,
		},
		{
			name:  "daily_loss_limit accepts exactly the limit",
			check: DailyLossCheck{Limit: 25_000}, portfolio: contracts.Portfolio{
				Cash: 75_000, RealisedPnL: -25_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{},
			},
			signal: signal(),
		},
		{
			name:      "daily_loss_limit disabled when unset",
			check:     DailyLossCheck{},
			portfolio: basePortfolio(),
			signal:    signal(),
		},

		// 5. debounce — the TTL key EXEC:<symbol>:<side>.
		{
			name:      "debounce rejects an armed key",
			check:     DebounceCheck{Store: newFakeStore("EXEC:RELIANCE:BUY"), TTL: 5 * time.Minute},
			portfolio: basePortfolio(),
			signal:    signal(),
			wantCode:  contracts.ReasonDebounced,
		},
		{
			name:      "debounce accepts an unarmed key",
			check:     DebounceCheck{Store: newFakeStore(), TTL: 5 * time.Minute},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
		{
			name:      "debounce is per side, not per symbol",
			check:     DebounceCheck{Store: newFakeStore("EXEC:RELIANCE:SELL"), TTL: 5 * time.Minute},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
		{
			name:      "debounce accepts a nil store",
			check:     DebounceCheck{},
			portfolio: basePortfolio(),
			signal:    signal(),
		},

		// 6. quantity — bounds, lot multiple, and no naked shorts.
		{
			name:      "quantity rejects zero",
			check:     QuantityCheck{MaxQuantity: 500},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Quantity = 0 }),
			wantCode:  contracts.ReasonBadQuantity,
		},
		{
			name:      "quantity rejects above max_order_quantity",
			check:     QuantityCheck{MaxQuantity: 500},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Quantity = 501 }),
			wantCode:  contracts.ReasonBadQuantity,
		},
		{
			name:      "quantity accepts exactly max_order_quantity",
			check:     QuantityCheck{MaxQuantity: 500},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Quantity = 500; s.Price = 20 }),
		},
		{
			name: "quantity rejects a non-multiple of the lot size",
			check: QuantityCheck{MaxQuantity: 500, LotSizes: map[string]int{
				"NIFTY26OCTFUT": 75,
			}},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Symbol = "NIFTY26OCTFUT"; s.Quantity = 76; s.Price = 100 }),
			wantCode:  contracts.ReasonBadQuantity,
		},
		{
			name: "quantity accepts a multiple of the lot size",
			check: QuantityCheck{MaxQuantity: 500, LotSizes: map[string]int{
				"NIFTY26OCTFUT": 75,
			}},
			portfolio: basePortfolio(),
			signal:    signal(func(s *contracts.Signal) { s.Symbol = "NIFTY26OCTFUT"; s.Quantity = 75; s.Price = 100 }),
		},
		{
			name:  "quantity rejects a naked short",
			check: QuantityCheck{MaxQuantity: 500}, portfolio: contracts.Portfolio{
				Cash: 100_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{"RELIANCE": 4},
			},
			signal:   signal(func(s *contracts.Signal) { s.Side = contracts.SideSell; s.Quantity = 10 }),
			wantCode: contracts.ReasonBadQuantity,
		},
		{
			name:  "quantity accepts a sell within the long position",
			check: QuantityCheck{MaxQuantity: 500}, portfolio: contracts.Portfolio{
				Cash: 100_000, PeakEquity: 100_000,
				OpenPositions: map[string]int{"RELIANCE": 10},
			},
			signal: signal(func(s *contracts.Signal) { s.Side = contracts.SideSell; s.Quantity = 10 }),
		},

		// 7. market_hours — 09:15-15:30 IST, weekdays, non-holidays.
		{
			name:      "market_hours rejects before the open",
			check:     MarketHoursCheck{Clock: clockAt(atIST(9, 14))},
			portfolio: basePortfolio(),
			signal:    signal(),
			wantCode:  contracts.ReasonMarketClosed,
		},
		{
			name:      "market_hours accepts the open",
			check:     MarketHoursCheck{Clock: clockAt(atIST(9, 15))},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
		{
			name:      "market_hours accepts the close",
			check:     MarketHoursCheck{Clock: clockAt(atIST(15, 30))},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
		{
			name:      "market_hours rejects after the close",
			check:     MarketHoursCheck{Clock: clockAt(atIST(15, 31))},
			portfolio: basePortfolio(),
			signal:    signal(),
			wantCode:  contracts.ReasonMarketClosed,
		},
		{
			name:      "market_hours rejects a weekend",
			check:     MarketHoursCheck{Clock: clockAt(time.Date(2026, 10, 3, 5, 0, 0, 0, time.UTC))},
			portfolio: basePortfolio(),
			signal:    signal(),
			wantCode:  contracts.ReasonMarketClosed,
		},
		{
			name: "market_hours rejects a holiday",
			check: MarketHoursCheck{
				Clock:    clockAt(atIST(10, 0)),
				Calendar: NewStaticCalendar("2026-09-29"),
			},
			portfolio: basePortfolio(),
			signal:    signal(),
			wantCode:  contracts.ReasonMarketClosed,
		},
		{
			name: "market_hours assumes open on a holiday set that omits today",
			check: MarketHoursCheck{
				Clock:    clockAt(atIST(10, 0)),
				Calendar: NewStaticCalendar("2026-10-02"),
			},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
		{
			name:      "market_hours assumes open with no calendar at all",
			check:     MarketHoursCheck{Clock: clockAt(atIST(10, 0))},
			portfolio: basePortfolio(),
			signal:    signal(),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.check.Check(context.Background(), tc.signal, tc.portfolio)

			if tc.wantCode == "" {
				if err != nil {
					t.Fatalf("Check() = %v, want pass", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Check() = nil, want rejection %s", tc.wantCode)
			}
			var rej *contracts.Rejection
			if !errors.As(err, &rej) {
				t.Fatalf("Check() = %T(%v), want *contracts.Rejection", err, err)
			}
			if rej.Code != tc.wantCode {
				t.Fatalf("rejection code = %s, want %s", rej.Code, tc.wantCode)
			}
		})
	}
}

// TestPolicyChainOrder asserts the composition order of docs/contracts.md §6:
// the first failure wins, so a portfolio that violates several limits at once
// reports the earliest one.
func TestPolicyChainOrder(t *testing.T) {
	policy := Policy{
		Policy: contracts.Policy{
			MaxPositionPct:   10,
			MaxOpenPositions: 2,
			MaxDrawdownPct:   5,
			DailyLossLimit:   25_000,
			MaxOrderQuantity: 500,
			DebounceTTL:      5 * time.Minute,
		},
		Clock: clockAt(openTime),
	}

	// A portfolio that is simultaneously over its position cap, over its
	// position count, at its drawdown limit, and past its daily loss, for a
	// zero-quantity signal on a debounced key outside market hours. Only the
	// first check in the chain gets a say.
	portfolio := contracts.Portfolio{
		Cash: 1_000, RealisedPnL: -30_000, PeakEquity: 100_000,
		OpenPositions: map[string]int{"RELIANCE": 10, "TCS": 5, "INFY": 5},
	}
	store := newFakeStore("EXEC:RELIANCE:BUY")
	sig := signal(func(s *contracts.Signal) { s.Quantity = 0 })

	err := policy.New(store).Check(context.Background(), sig, portfolio)
	var rej *contracts.Rejection
	if !errors.As(err, &rej) {
		t.Fatalf("Check() = %v, want *contracts.Rejection", err)
	}
	if rej.Code != contracts.ReasonMaxPosition {
		t.Fatalf("first failing check = %s, want %s", rej.Code, contracts.ReasonMaxPosition)
	}

	// A clean portfolio with an already-armed debounce key and a valid order
	// must be stopped by check 5, not pass.
	portfolio = basePortfolio()
	sig = signal()
	err = policy.New(store).Check(context.Background(), sig, portfolio)
	if !errors.As(err, &rej) {
		t.Fatalf("Check() = %v, want *contracts.Rejection", err)
	}
	if rej.Code != contracts.ReasonDebounced {
		t.Fatalf("rejection code = %s, want %s", rej.Code, contracts.ReasonDebounced)
	}
}

// TestPolicyChainPasses walks a valid signal through the full chain.
func TestPolicyChainPasses(t *testing.T) {
	policy := Policy{
		Policy: contracts.Policy{
			MaxPositionPct:   10,
			MaxOpenPositions: 2,
			MaxDrawdownPct:   5,
			DailyLossLimit:   25_000,
			MaxOrderQuantity: 500,
			DebounceTTL:      5 * time.Minute,
		},
		Clock: clockAt(openTime),
	}
	if err := policy.New(newFakeStore()).Check(context.Background(), signal(), basePortfolio()); err != nil {
		t.Fatalf("Check() = %v, want pass", err)
	}
}

func TestEquity(t *testing.T) {
	cases := []struct {
		name string
		p    contracts.Portfolio
		want float64
	}{
		{"flat book is cash", contracts.Portfolio{Cash: 100_000}, 100_000},
		{"realised loss reduces equity", contracts.Portfolio{Cash: 90_000, RealisedPnL: -10_000}, 80_000},
		{"realised gain raises equity", contracts.Portfolio{Cash: 110_000, RealisedPnL: 10_000}, 120_000},
		{"peak is not part of equity", contracts.Portfolio{Cash: 100_000, PeakEquity: 250_000}, 100_000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Equity(tc.p); got != tc.want {
				t.Fatalf("Equity() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestValidateSignal(t *testing.T) {
	cases := []struct {
		name    string
		signal  contracts.Signal
		wantErr bool
	}{
		{"valid", signal(), false},
		{"no symbol", signal(func(s *contracts.Signal) { s.Symbol = "" }), true},
		{"blank symbol", signal(func(s *contracts.Signal) { s.Symbol = "  " }), true},
		{"unknown side", signal(func(s *contracts.Signal) { s.Side = "HOLD" }), true},
		{"lower-case side", signal(func(s *contracts.Signal) { s.Side = "buy" }), true},
		{"zero price", signal(func(s *contracts.Signal) { s.Price = 0 }), true},
		{"negative price", signal(func(s *contracts.Signal) { s.Price = -1 }), true},
		{"no idempotency key", signal(func(s *contracts.Signal) { s.IdempotencyKey = "" }), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateSignal(tc.signal)
			if tc.wantErr != (err != nil) {
				t.Fatalf("ValidateSignal() error = %v, wantErr %v", err, tc.wantErr)
			}
			// A malformed signal must not be reported as a risk rejection:
			// there is no reason code for it and the metric is labelled by code.
			var rej *contracts.Rejection
			if errors.As(err, &rej) {
				t.Fatalf("ValidateSignal() = %v, want a plain error", err)
			}
		})
	}
}

func TestStaticCalendar(t *testing.T) {
	c := NewStaticCalendar("2026-10-02", "not-a-date")
	if !c.IsHoliday(atIST(10, 0).AddDate(0, 0, 3)) {
		t.Fatal("2026-10-02 should be a holiday")
	}
	if c.IsHoliday(atIST(10, 0)) {
		t.Fatal("2026-09-29 should not be a holiday")
	}
	if (StaticCalendar{}).IsHoliday(atIST(10, 0)) {
		t.Fatal("the zero calendar should report no holidays")
	}
}

func TestClockDefaultsToSystemClock(t *testing.T) {
	var c Clock
	if c.now().IsZero() {
		t.Fatal("a nil clock should read wall time, not the zero time")
	}
}
