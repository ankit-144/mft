package execution

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/testutil"
	"github.com/mft/services/execution/risk"
	"github.com/prometheus/client_golang/prometheus"
)

// openTime is a Tuesday at 10:30 IST, inside the NSE session. Every test
// injects it so nothing here depends on when the suite happens to run.
var openTime = time.Date(2026, 9, 29, 5, 0, 0, 0, time.UTC)

func testConfig() *config.Config {
	cfg := &config.Config{}
	cfg.Execution.Addr = ":0"
	cfg.Execution.Capital = 1_000_000
	cfg.Execution.DebounceTTLSeconds = 300
	cfg.Execution.MaxPositionPct = 10
	cfg.Execution.MaxOpenPositions = 10
	cfg.Execution.MaxDrawdownPct = 5
	cfg.Execution.DailyLossLimit = 25_000
	cfg.Execution.MaxOrderQuantity = 500
	return cfg
}

// newEngineWithRegistry builds an engine on its own Prometheus registry with
// the clock pinned to now, which is the injection point that keeps the market
// hours check out of wall-clock time.
func newEngineWithRegistry(t *testing.T, client *countingClient, cfg *config.Config, now time.Time) (*Engine, *prometheus.Registry) {
	t.Helper()
	policy := risk.FromConfig(cfg.Execution)
	policy.Clock = func() time.Time { return now }
	reg := testutil.NewRegistry()
	return newEngine(client, fluxkv.New(), cfg, reg, testutil.NewLogger(), policy, policy.Clock), reg
}

// rejectionCount reads one label of the reason-code counter vec.
func rejectionCount(t *testing.T, reg *prometheus.Registry, reason string) int {
	t.Helper()
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() != "execution_rejections_total" {
			continue
		}
		for _, m := range f.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == "reason" && l.GetValue() == reason {
					return int(m.GetCounter().GetValue())
				}
			}
		}
	}
	return 0
}

func validSignal(mutators ...func(*contracts.Signal)) contracts.Signal {
	sig := contracts.Signal{
		Symbol:         "RELIANCE",
		Side:           contracts.SideBuy,
		Quantity:       10,
		Price:          1000,
		Model:          "tabfm-v1.0.0",
		AsOf:           openTime,
		IdempotencyKey: "RELIANCE:BUY:20260929T1031",
	}
	for _, m := range mutators {
		m(&sig)
	}
	return sig
}

// countingClient records every PlaceOrder call. The risk tests need the exact
// call count, which testutil.MockClient only gives indirectly by slicing.
type countingClient struct {
	mu     sync.Mutex
	calls  int
	orders []testutil.OrderCall
	err    error
}

func (c *countingClient) PlaceOrder(_ context.Context, symbol, side string, quantity int, price float64) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return "", c.err
	}
	c.calls++
	c.orders = append(c.orders, testutil.OrderCall{
		Symbol: symbol, Side: side, Quantity: quantity, Price: price,
	})
	return "mock-order", nil
}

func (c *countingClient) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

func TestExecuteSignalPlacesOrder(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	orderID, err := engine.ExecuteSignal(context.Background(), validSignal())
	if err != nil {
		t.Fatalf("ExecuteSignal() error = %v", err)
	}
	if orderID != "mock-order" {
		t.Fatalf("order id = %q, want %q", orderID, "mock-order")
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}
}

// TestExecuteSignalIdempotent is the load-bearing test for the retry path:
// inference retries, and a replayed key must place exactly one order.
func TestExecuteSignalIdempotent(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	first, err := engine.ExecuteSignal(context.Background(), validSignal())
	if err != nil {
		t.Fatalf("first ExecuteSignal() error = %v", err)
	}

	second, err := engine.ExecuteSignal(context.Background(), validSignal())
	if err == nil {
		t.Fatal("a replayed key must be rejected")
	}
	var rej *contracts.Rejection
	if !errors.As(err, &rej) {
		t.Fatalf("error = %T(%v), want *contracts.Rejection", err, err)
	}
	if rej.Code != contracts.ReasonDuplicate {
		t.Fatalf("rejection code = %s, want %s", rej.Code, contracts.ReasonDuplicate)
	}
	if second != first {
		t.Fatalf("replay returned order %q, want the original %q", second, first)
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times for a repeated key, want 1", client.count())
	}
}

// TestExecuteSignalIdempotentConcurrent proves the exactly-once property holds
// when a retry storm arrives in parallel, not just sequentially.
func TestExecuteSignalIdempotentConcurrent(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	const racers = 16
	var wg sync.WaitGroup
	orderIDs := make([]string, racers)
	errs := make([]error, racers)
	for i := range racers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			orderIDs[i], errs[i] = engine.ExecuteSignal(context.Background(), validSignal())
		}()
	}
	wg.Wait()

	if client.count() != 1 {
		t.Fatalf("broker called %d times for one key under %d concurrent retries, want 1", client.count(), racers)
	}

	// Exactly one racer wins the key; every other one is told it is a
	// duplicate and is handed the winner's order id. The winner is whatever
	// the broker returned, since the losers are defined by comparison to it.
	winner := ""
	for _, err := range errs {
		if err == nil {
			winner = "mock-order"
		}
	}
	for i, err := range errs {
		if err == nil {
			continue
		}
		var rej *contracts.Rejection
		if !errors.As(err, &rej) || rej.Code != contracts.ReasonDuplicate {
			t.Fatalf("goroutine %d error = %v, want a RISK_DUPLICATE rejection", i, err)
		}
		if orderIDs[i] != winner {
			t.Fatalf("goroutine %d got order %q, want the winner's %q", i, orderIDs[i], winner)
		}
	}
}

// TestExecuteSignalRejectedNeverReachesBroker walks every reason code and
// asserts the broker is untouched for each. Several cases need a portfolio
// that is already in breach, which is seeded through a real Book so the state
// under test is the state production would see.
func TestExecuteSignalRejectedNeverReachesBroker(t *testing.T) {
	cases := []struct {
		name     string
		policy   func(*risk.Policy)
		now      time.Time
		signal   func(*contracts.Signal)
		seed     func(capital float64) *risk.Book
		preArm   string
		preClaim string
		wantCode string
	}{
		{
			// 1% of 1,000,000 is a 10,000 order value; this one is 10,050.
			name:     contracts.ReasonMaxPosition,
			policy:   func(p *risk.Policy) { p.MaxPositionPct = 1 },
			signal:   func(s *contracts.Signal) { s.Price = 1005 },
			wantCode: contracts.ReasonMaxPosition,
		},
		{
			// One symbol is held and a second would breach a limit of one.
			name:     contracts.ReasonMaxPositions,
			policy:   func(p *risk.Policy) { p.MaxOpenPositions = 1 },
			signal:   func(s *contracts.Signal) { s.Symbol = "INFY" },
			seed:     seedLong("TCS", 10, 1000),
			wantCode: contracts.ReasonMaxPositions,
			// The seeded TCS position has to be affordable under the
			// tightened policy, or check 1 would fire first.
		},
		{
			// A 90,000 loss on a 1,000,000 peak is a 9% drawdown against a 5%
			// limit. The same loss also breaches the daily limit, but the
			// drawdown check is third in the chain and reports first.
			name:     contracts.ReasonMaxDrawdown,
			policy:   func(p *risk.Policy) { p.MaxDrawdownPct = 5 },
			seed:     seedLoss("SEED", 300, 1000, 700),
			wantCode: contracts.ReasonMaxDrawdown,
		},
		{
			// The same round trip realised -90,000 against a 25,000 limit, with
			// the drawdown cap lifted so check 4 is what fires.
			name:     contracts.ReasonDailyLoss,
			policy:   func(p *risk.Policy) { p.DailyLossLimit = 25_000; p.MaxDrawdownPct = 50 },
			seed:     seedLoss("SEED", 300, 1000, 700),
			wantCode: contracts.ReasonDailyLoss,
		},
		{
			name:     contracts.ReasonMarketClosed,
			now:      openTime.Add(8 * time.Hour),
			wantCode: contracts.ReasonMarketClosed,
		},
		{
			name:     contracts.ReasonBadQuantity,
			signal:   func(s *contracts.Signal) { s.Quantity = 0 },
			wantCode: contracts.ReasonBadQuantity,
		},
		{
			// A sell with no long position is a short, which v1 does not place.
			name:     contracts.ReasonBadQuantity + " short",
			signal:   func(s *contracts.Signal) { s.Side = contracts.SideSell; s.Quantity = 10 },
			wantCode: contracts.ReasonBadQuantity,
		},
		{
			// Check 5: the debounce key is already armed for this symbol and
			// side from an earlier successful order.
			name:     contracts.ReasonDebounced,
			signal:   func(s *contracts.Signal) { s.Symbol = "TCS" },
			preArm:   risk.DebounceKey("TCS", contracts.SideBuy),
			wantCode: contracts.ReasonDebounced,
		},
		{
			// The key was claimed by an earlier attempt, so this is a replay
			// and never reaches the chain at all.
			name:     contracts.ReasonDuplicate,
			preClaim: contracts.ReasonDuplicate + ":key",
			wantCode: contracts.ReasonDuplicate,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			policy := risk.FromConfig(testConfig().Execution)
			if tc.policy != nil {
				tc.policy(&policy)
			}
			now := tc.now
			if now.IsZero() {
				now = openTime
			}
			policy.Clock = func() time.Time { return now }

			client := &countingClient{}
			cache := fluxkv.New()
			cfg := testConfig()
			engine := newEngine(client, cache, cfg, testutil.NewRegistry(), testutil.NewLogger(), policy, policy.Clock)

			if tc.seed != nil {
				engine.book = tc.seed(cfg.Execution.Capital)
			}
			if tc.preArm != "" {
				cache.Set(tc.preArm, "mock-order", time.Hour)
			}
			if tc.preClaim != "" {
				cache.Set("IDEM:"+tc.preClaim, "mock-order", time.Hour)
			}

			before := client.count()

			sig := validSignal(func(s *contracts.Signal) {
				s.IdempotencyKey = tc.name + ":key"
			})
			if tc.signal != nil {
				tc.signal(&sig)
			}

			_, err := engine.ExecuteSignal(context.Background(), sig)
			if err == nil {
				t.Fatal("ExecuteSignal() = nil, want a rejection")
			}
			var rej *contracts.Rejection
			if !errors.As(err, &rej) {
				t.Fatalf("error = %T(%v), want *contracts.Rejection", err, err)
			}
			if rej.Code != tc.wantCode {
				t.Fatalf("rejection code = %s, want %s", rej.Code, tc.wantCode)
			}
			if got := client.count() - before; got != 0 {
				t.Fatalf("broker was called %d times for a rejected signal, want 0", got)
			}
		})
	}
}

func TestExecuteSignalDebounced(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	first := validSignal()
	if _, err := engine.ExecuteSignal(context.Background(), first); err != nil {
		t.Fatalf("first ExecuteSignal() error = %v", err)
	}

	// A different key, same symbol and side: this is a fresh decision that the
	// debounce window must still refuse.
	second := validSignal(func(s *contracts.Signal) { s.IdempotencyKey = "RELIANCE:BUY:20260929T1032" })
	_, err := engine.ExecuteSignal(context.Background(), second)
	if err == nil {
		t.Fatal("a second order for the same symbol and side must be debounced")
	}
	var rej *contracts.Rejection
	if !errors.As(err, &rej) || rej.Code != contracts.ReasonDebounced {
		t.Fatalf("error = %v, want a RISK_DEBOUNCED rejection", err)
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}
}

func TestExecuteSignalDebounceIsPerSide(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	buy := validSignal()
	if _, err := engine.ExecuteSignal(context.Background(), buy); err != nil {
		t.Fatalf("buy error = %v", err)
	}

	sell := validSignal(func(s *contracts.Signal) {
		s.Side = contracts.SideSell
		s.Quantity = 4
		s.IdempotencyKey = "RELIANCE:SELL:20260929T1031"
	})
	if _, err := engine.ExecuteSignal(context.Background(), sell); err != nil {
		t.Fatalf("the opposite side must not be debounced: %v", err)
	}
	if client.count() != 2 {
		t.Fatalf("broker called %d times, want 2", client.count())
	}
}

func TestExecuteSignalBrokerErrorLeavesNoClaim(t *testing.T) {
	client := &countingClient{err: errors.New("broker down")}
	engine, reg := newEngineWithRegistry(t, client, testConfig(), openTime)

	_, err := engine.ExecuteSignal(context.Background(), validSignal())
	if err == nil {
		t.Fatal("a broker failure must propagate")
	}

	// The failed attempt must not have claimed the key or armed the debounce:
	// a retry after the broker recovers has to be able to place the order.
	client.err = nil
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatalf("retry after a broker failure error = %v, want success", err)
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}
	if got := testutil.MetricValue(t, "execution_orders_placed_total", reg, nil); got != 1 {
		t.Fatalf("orders_placed_total = %d, want 1", got)
	}
}

// TestExecuteSignalRejectionMetrics asserts the reason-code counter vec is
// labelled, and that a replay is counted as a duplicate rather than silently
// absorbed.
func TestExecuteSignalRejectionMetrics(t *testing.T) {
	client := &countingClient{}
	engine, reg := newEngineWithRegistry(t, client, testConfig(), openTime)

	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatalf("first ExecuteSignal() error = %v", err)
	}
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err == nil {
		t.Fatal("the replay should have been rejected")
	}

	if got := rejectionCount(t, reg, contracts.ReasonDuplicate); got != 1 {
		t.Fatalf("%s = %d, want 1", contracts.ReasonDuplicate, got)
	}
	if got := testutil.MetricValue(t, "execution_orders_rejected_total", reg, nil); got != 1 {
		t.Fatalf("orders_rejected_total = %d, want 1", got)
	}
	if got := testutil.MetricValue(t, "execution_orders_placed_total", reg, nil); got != 1 {
		t.Fatalf("orders_placed_total = %d, want 1", got)
	}

	// A risk rejection labels the metric with its own code.
	engine2, reg2 := newEngineWithRegistry(t, &countingClient{}, testConfig(), openTime.Add(8*time.Hour))
	if _, err := engine2.ExecuteSignal(context.Background(), validSignal()); err == nil {
		t.Fatal("the signal should have been rejected outside market hours")
	}
	if got := rejectionCount(t, reg2, contracts.ReasonMarketClosed); got != 1 {
		t.Fatalf("%s = %d, want 1", contracts.ReasonMarketClosed, got)
	}
	if got := testutil.MetricValue(t, "execution_orders_placed_total", reg2, nil); got != 0 {
		t.Fatalf("orders_placed_total = %d, want 0", got)
	}
}

// TestExecuteSignalMaintainsPortfolio walks a full round trip and checks the
// engine's exposure view tracks it.
func TestExecuteSignalMaintainsPortfolio(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	if p := engine.Portfolio(); p.Cash != 1_000_000 || len(p.OpenPositions) != 0 {
		t.Fatalf("initial portfolio = %+v, want 1000000 cash and no positions", p)
	}

	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatalf("buy error = %v", err)
	}
	p := engine.Portfolio()
	if p.OpenPositions["RELIANCE"] != 10 {
		t.Fatalf("positions after buy = %+v, want RELIANCE 10", p.OpenPositions)
	}
	if p.Cash != 1_000_000 {
		t.Fatalf("account cash after buy = %v, want 1000000", p.Cash)
	}
	if p.PeakEquity != 1_000_000 {
		t.Fatalf("peak equity after buy = %v, want 1000000", p.PeakEquity)
	}

	sell := validSignal(func(s *contracts.Signal) {
		s.Side = contracts.SideSell
		s.Quantity = 10
		s.Price = 1100
		s.IdempotencyKey = "RELIANCE:SELL:20260929T1031"
	})
	if _, err := engine.ExecuteSignal(context.Background(), sell); err != nil {
		t.Fatalf("sell error = %v", err)
	}

	p = engine.Portfolio()
	if len(p.OpenPositions) != 0 {
		t.Fatalf("positions after close = %+v, want none", p.OpenPositions)
	}
	if p.RealisedPnL != 1_000 {
		t.Fatalf("realised PnL = %v, want 1000", p.RealisedPnL)
	}
	if got := risk.Equity(p); got != 1_001_000 {
		t.Fatalf("equity after a +1000 round trip = %v, want 1001000", got)
	}
	if p.PeakEquity != 1_001_000 {
		t.Fatalf("peak equity = %v, want 1001000", p.PeakEquity)
	}
}

// TestExecuteSignalPositionLimitClosesTheBook proves the position check reads
// real engine state: ten filled symbols, then an eleventh is refused.
func TestExecuteSignalPositionLimitClosesTheBook(t *testing.T) {
	client := &countingClient{}
	cfg := testConfig()
	cfg.Execution.MaxOpenPositions = 2
	engine, _ := newEngineWithRegistry(t, client, cfg, openTime)

	for i, symbol := range []string{"RELIANCE", "TCS"} {
		sig := validSignal(func(s *contracts.Signal) {
			s.Symbol = symbol
			s.IdempotencyKey = string(rune('a'+i)) + ":1"
		})
		if _, err := engine.ExecuteSignal(context.Background(), sig); err != nil {
			t.Fatalf("%s error = %v", symbol, err)
		}
	}

	third := validSignal(func(s *contracts.Signal) {
		s.Symbol = "INFY"
		s.IdempotencyKey = "third:1"
	})
	_, err := engine.ExecuteSignal(context.Background(), third)
	var rej *contracts.Rejection
	if !errors.As(err, &rej) || rej.Code != contracts.ReasonMaxPositions {
		t.Fatalf("error = %v, want a RISK_MAX_POSITIONS rejection", err)
	}
	if client.count() != 2 {
		t.Fatalf("broker called %d times, want 2", client.count())
	}
}

// TestExecuteSignalMalformed checks that a signal which is not well-formed is
// refused before the chain, and that it is not counted as a risk rejection —
// there is no reason code for a malformed request.
func TestExecuteSignalMalformed(t *testing.T) {
	cases := []struct {
		name   string
		signal func(*contracts.Signal)
	}{
		{"no symbol", func(s *contracts.Signal) { s.Symbol = "" }},
		{"bad side", func(s *contracts.Signal) { s.Side = "HOLD" }},
		{"no price", func(s *contracts.Signal) { s.Price = 0 }},
		{"no idempotency key", func(s *contracts.Signal) { s.IdempotencyKey = "" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := &countingClient{}
			engine, reg := newEngineWithRegistry(t, client, testConfig(), openTime)

			_, err := engine.ExecuteSignal(context.Background(), validSignal(tc.signal))
			if err == nil {
				t.Fatal("ExecuteSignal() = nil, want an error")
			}
			var rej *contracts.Rejection
			if errors.As(err, &rej) {
				t.Fatalf("error = %v, want a plain error, not a risk rejection", err)
			}
			if client.count() != 0 {
				t.Fatalf("broker was called %d times, want 0", client.count())
			}
			if got := testutil.MetricValue(t, "execution_orders_rejected_total", reg, nil); got != 1 {
				t.Fatalf("orders_rejected_total = %d, want 1", got)
			}
		})
	}
}

// TestExecute covers the loose-field entry point that the HTTP layer calls. It
// must run the same gate; the only difference is a derived, debounce-scoped
// idempotency key.
func TestExecute(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 1000); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}

	// The same order inside the debounce window is refused.
	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 1000); err == nil {
		t.Fatal("an identical order inside the debounce window must be refused")
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}
}

func TestExecuteOutsideMarketHours(t *testing.T) {
	client := &countingClient{}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime.Add(9*time.Hour))

	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 1000); err == nil {
		t.Fatal("an order outside the session must be refused")
	}
	if client.count() != 0 {
		t.Fatalf("broker was called %d times, want 0", client.count())
	}
}

func TestExecuteBrokerError(t *testing.T) {
	client := &countingClient{err: errors.New("kite 502")}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)

	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 1000); err == nil {
		t.Fatal("a broker failure must propagate")
	}
	if client.count() != 0 {
		t.Fatalf("a failed placement must not count as an order, got %d", client.count())
	}
}

// seedLong returns a book already holding a long position in symbol.
func seedLong(symbol string, quantity int, price float64) func(float64) *risk.Book {
	return func(capital float64) *risk.Book {
		book := risk.NewBook(capital)
		mustBookApply(book, contracts.Signal{
			Symbol: symbol, Side: contracts.SideBuy, Quantity: quantity, Price: price,
		})
		return book
	}
}

// seedLoss returns a book that has round-tripped a losing position, leaving
// both a drawdown from the peak and a realised daily loss on the books.
func seedLoss(symbol string, quantity, buyPrice, sellPrice int) func(float64) *risk.Book {
	return func(capital float64) *risk.Book {
		book := risk.NewBook(capital)
		mustBookApply(book, contracts.Signal{
			Symbol: symbol, Side: contracts.SideBuy, Quantity: quantity, Price: float64(buyPrice),
		})
		mustBookApply(book, contracts.Signal{
			Symbol: symbol, Side: contracts.SideSell, Quantity: quantity, Price: float64(sellPrice),
		})
		return book
	}
}

func mustBookApply(book *risk.Book, sig contracts.Signal) {
	if err := book.Apply(sig); err != nil {
		panic("seed fill could not be applied: " + err.Error())
	}
}
