package execution

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/mft/core/broker"
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
	journalDir, _ := os.MkdirTemp("", "mft-execution-test-")
	cfg.Execution.JournalPath = filepath.Join(journalDir, "state.json")
	cfg.Execution.PaperTrading = false
	cfg.Execution.APIToken = "test-token"
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
	engine := newEngine(client, fluxkv.New(), cfg, reg, testutil.NewLogger(), policy, policy.Clock)
	// Unit tests exercise submission paths directly; Fx startup performs the
	// real broker reconciliation before setting this flag in production.
	engine.ready = true
	return engine, reg
}

// rejectionCount reads one label of the reason-code counter vec.
func rejectionCount(t *testing.T, reg *prometheus.Registry, reason string) int {
	t.Helper()
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() != "mft_execution_rejections_total" {
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
	mu          sync.Mutex
	calls       int
	orders      []testutil.OrderCall
	err         error
	states      map[string]broker.Order
	state       broker.OrderState
	acceptedErr error
	started     chan struct{}
	release     chan struct{}
	quoteCalls  int
	quoteErr    error
}

func (c *countingClient) PlaceOrder(ctx context.Context, req contracts.OrderRequest) (string, error) {
	return c.place(ctx, req, "")
}

func (c *countingClient) PlaceOrderTagged(ctx context.Context, req contracts.OrderRequest, tag string) (string, error) {
	return c.place(ctx, req, tag)
}

func (c *countingClient) place(_ context.Context, req contracts.OrderRequest, tag string) (string, error) {
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return "", c.err
	}
	c.calls++
	c.orders = append(c.orders, testutil.OrderCall{
		Symbol: req.Symbol, Side: req.Side, Quantity: req.Quantity, Price: req.Price,
	})
	if c.states == nil {
		c.states = make(map[string]broker.Order)
	}
	state := c.state
	if state == "" {
		state = broker.OrderFilled
	}
	filled := 0
	if state == broker.OrderFilled {
		filled = req.Quantity
	}
	avg := req.Price
	if avg == 0 {
		avg = 1000
	}
	orderID := "mock-order"
	c.states[orderID] = broker.Order{ID: orderID, Tag: tag, Status: state, Symbol: req.Symbol, Side: req.Side, Quantity: req.Quantity, Price: req.Price, FilledQuantity: filled, AverageFillPrice: avg, UpdatedAt: openTime}
	started, release, acceptedErr := c.started, c.release, c.acceptedErr
	c.mu.Unlock()
	if started != nil {
		started <- struct{}{}
	}
	if release != nil {
		<-release
	}
	if acceptedErr != nil {
		return "", acceptedErr
	}
	return orderID, nil
}

func (c *countingClient) CancelOrder(context.Context, string) error { return nil }
func (c *countingClient) GetPositions(context.Context) ([]contracts.Position, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	quantities := map[string]int{}
	for _, o := range c.states {
		if o.FilledQuantity > 0 {
			q := o.FilledQuantity
			if o.Side == contracts.SideSell {
				q = -q
			}
			quantities[o.Symbol] += q
		}
	}
	var out []contracts.Position
	for symbol, quantity := range quantities {
		if quantity != 0 {
			out = append(out, contracts.Position{Symbol: symbol, Quantity: quantity})
		}
	}
	return out, nil
}
func (c *countingClient) GetOrder(_ context.Context, id string) (broker.Order, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	o, ok := c.states[id]
	if !ok {
		return broker.Order{}, errors.New("unknown order")
	}
	return o, nil
}
func (c *countingClient) GetOrders(context.Context) ([]broker.Order, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]broker.Order, 0, len(c.states))
	for _, o := range c.states {
		out = append(out, o)
	}
	return out, nil
}
func (c *countingClient) GetLastPrices(_ context.Context, symbols []string) (map[string]float64, error) {
	c.mu.Lock()
	c.quoteCalls++
	err := c.quoteErr
	c.mu.Unlock()
	if err != nil {
		return nil, err
	}
	prices := make(map[string]float64, len(symbols))
	for _, symbol := range symbols {
		prices[symbol] = 1000
	}
	return prices, nil
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

func TestPaperModeBooksSimulatedFillWithoutBrokerCall(t *testing.T) {
	client := &countingClient{quoteErr: errors.New("quotes offline")}
	cfg := testConfig()
	cfg.Execution.PaperTrading = true
	policy := risk.FromConfig(cfg.Execution)
	policy.Clock = func() time.Time { return openTime }
	engine, err := buildEngine(client, fluxkv.New(), cfg, testutil.NewRegistry(), testutil.NewLogger(), policy, policy.Clock)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatal(err)
	}
	if client.count() != 0 {
		t.Fatalf("paper mode sent %d broker orders", client.count())
	}
	if got := engine.OrderStatus("RELIANCE:BUY:20260929T1031"); got.Status != contracts.OrderStatusPaperFilled || got.FilledQuantity != 10 {
		t.Fatalf("paper order result = %+v", got)
	}
	stale := validSignal(func(s *contracts.Signal) { s.AsOf = openTime.Add(-24 * time.Hour) })
	if _, err := engine.ExecuteSignal(context.Background(), stale); err == nil {
		t.Fatal("duplicate request should still report the existing claim")
	} else if rej, ok := err.(*contracts.Rejection); !ok || rej.Code != contracts.ReasonDuplicate {
		t.Fatalf("duplicate error = %T(%v), want duplicate rejection", err, err)
	}
	client.mu.Lock()
	quoteCalls := client.quoteCalls
	client.mu.Unlock()
	if quoteCalls != 0 {
		t.Fatalf("paper execution made %d quote calls, want none", quoteCalls)
	}
}

func TestReconcileRunsIdempotentExitPolicyAfterLossLimit(t *testing.T) {
	client := &countingClient{}
	cfg := testConfig()
	cfg.Execution.PaperTrading = true
	cfg.Execution.MaxDrawdownPct = 1
	cfg.Execution.DailyLossLimit = 100
	policy := risk.FromConfig(cfg.Execution)
	policy.Clock = func() time.Time { return openTime }
	engine, err := buildEngine(client, fluxkv.New(), cfg, testutil.NewRegistry(), testutil.NewLogger(), policy, policy.Clock)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	book := risk.NewBook(cfg.Execution.Capital)
	if err := book.ApplyFill(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy}, 100, 1000, openTime); err != nil {
		t.Fatal(err)
	}
	if err := book.ApplyFill(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell}, 10, 500, openTime); err != nil {
		t.Fatal(err)
	}
	engine.book = book
	policyCalls := 0
	engine.SetExitPolicy(ExitPolicyFunc(func(_ context.Context, p contracts.Portfolio) ([]contracts.OrderRequest, error) {
		policyCalls++
		if policyCalls == 1 && p.OpenPositions["RELIANCE"] != 90 {
			t.Fatalf("exit policy saw %d shares, want 90", p.OpenPositions["RELIANCE"])
		}
		return []contracts.OrderRequest{{
			Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 1,
			Price: 500, Type: contracts.OrderTypeLimit, IdempotencyKey: "strategy:exit:RELIANCE:1",
		}}, nil
	}))
	if err := engine.Reconcile(context.Background()); err != nil {
		t.Fatalf("reconcile/exit failed: %v", err)
	}
	result := engine.OrderStatus("strategy:exit:RELIANCE:1")
	if result.Status != contracts.OrderStatusPaperFilled || result.FilledQuantity != 1 {
		t.Fatalf("exit result = %+v", result)
	}
	if err := engine.Reconcile(context.Background()); err != nil {
		t.Fatalf("repeat reconcile failed: %v", err)
	}
	if got := engine.Portfolio().OpenPositions["RELIANCE"]; got != 89 {
		t.Fatalf("repeated exit policy changed position to %d, want 89", got)
	}
	if client.count() != 0 {
		t.Fatalf("paper exit called broker %d times", client.count())
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

func TestDebounceReservationSurvivesRestart(t *testing.T) {
	client := &countingClient{}
	cfg := testConfig()
	engine, _ := newEngineWithRegistry(t, client, cfg, openTime)
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	restarted, _ := newEngineWithRegistry(t, client, cfg, openTime)
	defer restarted.Close()
	sig := validSignal(func(s *contracts.Signal) { s.IdempotencyKey = "different-key-same-decision" })
	_, err := restarted.ExecuteSignal(context.Background(), sig)
	var rejection *contracts.Rejection
	if !errors.As(err, &rejection) || rejection.Code != contracts.ReasonDebounced {
		t.Fatalf("error = %v, want restored debounce rejection", err)
	}
	if client.count() != 1 {
		t.Fatalf("broker placements after restart = %d, want 1", client.count())
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

	// Exactly one racer wins the key. A loser that arrives before the broker
	// acknowledges sees a truthful empty order id and SUBMITTING state.
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
		if orderIDs[i] != "" && orderIDs[i] != winner {
			t.Fatalf("goroutine %d got order %q, want pending or the winner's %q", i, orderIDs[i], winner)
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
				engine.state.Orders[tc.name+":key"] = OrderRecord{Key: tc.name + ":key", OrderID: "prior", Status: contracts.OrderStatusFilled, CreatedAt: now}
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

func TestExecuteSignalBrokerErrorFailsClosed(t *testing.T) {
	client := &countingClient{err: errors.New("broker down")}
	engine, reg := newEngineWithRegistry(t, client, testConfig(), openTime)

	_, err := engine.ExecuteSignal(context.Background(), validSignal())
	if err == nil {
		t.Fatal("a broker failure must propagate")
	}

	// A network error may hide an accepted order. The stable key stays claimed
	// and readiness remains closed until reconciliation resolves it.
	client.err = nil
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err == nil {
		t.Fatal("retry after an indeterminate broker failure must fail closed")
	}
	if client.count() != 0 {
		t.Fatalf("broker calls = %d, want no accepted placement", client.count())
	}
	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg, nil); got != 0 {
		t.Fatalf("orders_placed_total = %d, want 0", got)
	}
	_ = reg
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
	if got := testutil.MetricValue(t, "mft_execution_orders_rejected_total", reg, nil); got != 1 {
		t.Fatalf("orders_rejected_total = %d, want 1", got)
	}
	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg, nil); got != 1 {
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
	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg2, nil); got != 0 {
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
	if p.Cash != 990_000 {
		t.Fatalf("account cash after buy = %v, want 990000", p.Cash)
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

func TestAcknowledgementReservesUntilFill(t *testing.T) {
	client := &countingClient{state: broker.OrderOpen}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)
	id, err := engine.ExecuteSignal(context.Background(), validSignal())
	if err != nil {
		t.Fatal(err)
	}
	if got := engine.OrderStatus("RELIANCE:BUY:20260929T1031"); got.Status != contracts.OrderStatusOpen || got.FilledQuantity != 0 || got.OrderID != id {
		t.Fatalf("order result = %+v, want acknowledged OPEN with no fill", got)
	}
	p := engine.Portfolio()
	if p.Cash != 1_000_000 || p.OpenPositions["RELIANCE"] != 0 || p.ReservedCash != 10_000 || p.ReservedPositions["RELIANCE"] != 10 {
		t.Fatalf("portfolio after broker acknowledgement = %+v", p)
	}
}

func TestUnknownBrokerStateIsReportedAsIndeterminate(t *testing.T) {
	client := &countingClient{state: broker.OrderUnknown}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)
	_, err := engine.ExecuteSignal(context.Background(), validSignal())
	var indeterminate *IndeterminateError
	if !errors.As(err, &indeterminate) {
		t.Fatalf("error = %v, want indeterminate", err)
	}
	if got := engine.OrderStatus("RELIANCE:BUY:20260929T1031").Status; got != contracts.OrderStatusUnknown {
		t.Fatalf("status = %s", got)
	}
	if ready, _ := engine.Ready(); ready {
		t.Fatal("unknown broker state left engine ready")
	}
}

func TestPartialFillThenCancelReleasesOnlyUnfilledReservation(t *testing.T) {
	client := &countingClient{state: broker.OrderPartial}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatal(err)
	}
	client.mu.Lock()
	o := client.states["mock-order"]
	o.FilledQuantity, o.AverageFillPrice = 4, 1000
	client.states[o.ID] = o
	client.mu.Unlock()
	if err := engine.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	p := engine.Portfolio()
	if p.OpenPositions["RELIANCE"] != 4 || p.Cash != 996_000 || p.ReservedCash != 6_000 || p.ReservedPositions["RELIANCE"] != 6 {
		t.Fatalf("portfolio after partial fill = %+v", p)
	}
	client.mu.Lock()
	o.Status = broker.OrderCancelled
	client.states[o.ID] = o
	client.mu.Unlock()
	if err := engine.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	rec := engine.state.Orders["RELIANCE:BUY:20260929T1031"]
	client.mu.Lock()
	delayedOpen := client.states[rec.OrderID]
	delayedOpen.Status = broker.OrderOpen
	client.mu.Unlock()
	if err := engine.applyBrokerOrder("RELIANCE:BUY:20260929T1031", delayedOpen); err != nil {
		t.Fatal(err)
	}
	p = engine.Portfolio()
	if p.OpenPositions["RELIANCE"] != 4 || p.ReservedCash != 0 || p.ReservedPositions["RELIANCE"] != 0 || engine.OrderStatus("RELIANCE:BUY:20260929T1031").Status != contracts.OrderStatusCancelled {
		t.Fatalf("portfolio/order after cancellation = %+v / %+v", p, engine.OrderStatus("RELIANCE:BUY:20260929T1031"))
	}
}

func TestBrokerUpdatesCannotDowngradeTerminalStatus(t *testing.T) {
	client := &countingClient{state: broker.OrderFilled}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)
	key := "RELIANCE:BUY:20260929T1031"
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatal(err)
	}
	rec := engine.state.Orders[key]
	staleOpen := broker.Order{ID: rec.OrderID, Tag: rec.Tag, Status: broker.OrderOpen,
		Symbol: rec.Request.Symbol, Side: rec.Request.Side, Quantity: rec.Request.Quantity,
		FilledQuantity: rec.Request.Quantity, AverageFillPrice: 1000, UpdatedAt: openTime}
	if err := engine.applyBrokerOrder(key, staleOpen); err != nil {
		t.Fatal(err)
	}
	if got := engine.OrderStatus(key); got.Status != contracts.OrderStatusFilled {
		t.Fatalf("stale OPEN downgraded FILLED: %+v", got)
	}
}

func TestBrokerCannotMarkPartialOrderFilledOrReduceCumulativeNotional(t *testing.T) {
	client := &countingClient{state: broker.OrderPartial}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)
	key := "RELIANCE:BUY:20260929T1031"
	if _, err := engine.ExecuteSignal(context.Background(), validSignal()); err != nil {
		t.Fatal(err)
	}
	rec := engine.state.Orders[key]
	partial := broker.Order{ID: rec.OrderID, Tag: rec.Tag, Status: broker.OrderPartial,
		Symbol: rec.Request.Symbol, Side: rec.Request.Side, Quantity: rec.Request.Quantity,
		FilledQuantity: 4, AverageFillPrice: 1000, UpdatedAt: openTime}
	if err := engine.applyBrokerOrder(key, partial); err != nil {
		t.Fatal(err)
	}
	partial.FilledQuantity = 3
	if err := engine.applyBrokerOrder(key, partial); err == nil {
		t.Fatal("decreasing cumulative fill quantity was accepted")
	}
	partial.FilledQuantity = 4
	partial.Status = broker.OrderFilled
	if err := engine.applyBrokerOrder(key, partial); err == nil {
		t.Fatal("incomplete cumulative quantity was accepted as FILLED")
	}
	partial.Status = broker.OrderPartial
	partial.FilledQuantity = 5
	partial.AverageFillPrice = 700 // cumulative notional falls from 4,000 to 3,500
	if err := engine.applyBrokerOrder(key, partial); err == nil {
		t.Fatal("decreasing cumulative notional was accepted")
	}
	if got := engine.Portfolio().OpenPositions["RELIANCE"]; got != 4 {
		t.Fatalf("invalid broker update changed held quantity to %d, want 4", got)
	}
	if got := engine.OrderStatus(key); got.Status != contracts.OrderStatusPartial || got.FilledQuantity != 4 {
		t.Fatalf("invalid broker update changed order state: %+v", got)
	}
}

func TestRejectedOrderReleasesCashAndDebounceReservation(t *testing.T) {
	client := &countingClient{state: broker.OrderRejected}
	engine, _ := newEngineWithRegistry(t, client, testConfig(), openTime)
	first := validSignal(func(s *contracts.Signal) { s.IdempotencyKey = "rejected-first" })
	if _, err := engine.ExecuteSignal(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	if got := engine.OrderStatus(first.IdempotencyKey); got.Status != contracts.OrderStatusRejected {
		t.Fatalf("first status = %+v", got)
	}
	p := engine.Portfolio()
	if p.ReservedCash != 0 || p.OpenPositions["RELIANCE"] != 0 {
		t.Fatalf("rejected order retained reservation: %+v", p)
	}
	client.state = broker.OrderFilled
	second := validSignal(func(s *contracts.Signal) { s.IdempotencyKey = "rejected-retry" })
	if _, err := engine.ExecuteSignal(context.Background(), second); err != nil {
		t.Fatalf("retry after definitive broker rejection: %v", err)
	}
	if client.count() != 2 {
		t.Fatalf("broker placements = %d, want 2", client.count())
	}
}

func TestAcceptedButTimedOutReconcilesByDurableTag(t *testing.T) {
	client := &countingClient{acceptedErr: errors.New("reply lost after broker acceptance")}
	cfg := testConfig()
	policy := risk.FromConfig(cfg.Execution)
	policy.Clock = func() time.Time { return openTime }
	engine, err := buildEngine(client, fluxkv.New(), cfg, testutil.NewRegistry(), testutil.NewLogger(), policy, policy.Clock)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Initialize(context.Background()); err != nil {
		t.Fatal(err)
	}
	key := "timeout-key"
	sig := validSignal(func(s *contracts.Signal) { s.IdempotencyKey = key })
	if _, err := engine.ExecuteSignal(context.Background(), sig); err == nil {
		t.Fatal("lost broker reply must be reported as indeterminate")
	}
	if ready, _ := engine.Ready(); ready {
		t.Fatal("engine became ready with unresolved submission")
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	recovered, err := buildEngine(client, fluxkv.New(), cfg, testutil.NewRegistry(), testutil.NewLogger(), policy, policy.Clock)
	if err != nil {
		t.Fatal(err)
	}
	defer recovered.Close()
	if ready, _ := recovered.Ready(); ready {
		t.Fatal("restart became ready before broker reconciliation")
	}
	if err := recovered.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := recovered.OrderStatus(key); got.Status != contracts.OrderStatusFilled || got.FilledQuantity != 10 {
		t.Fatalf("reconciled result = %+v, record=%+v, broker=%+v", got, recovered.state.Orders[key], client.states)
	}
	if _, err := recovered.ExecuteSignal(context.Background(), sig); err == nil {
		t.Fatal("replay after recovery must be rejected as duplicate")
	}
	if ready, reason := recovered.Ready(); !ready {
		t.Fatalf("engine remains unready: %s", reason)
	}
	if client.count() != 1 {
		t.Fatalf("broker placement calls = %d, want exactly 1", client.count())
	}
}

func TestConcurrentReservationProtectsAvailableCash(t *testing.T) {
	client := &countingClient{started: make(chan struct{}, 1), release: make(chan struct{})}
	cfg := testConfig()
	cfg.Execution.Capital = 10_000
	cfg.Execution.MaxPositionPct = 100
	engine, _ := newEngineWithRegistry(t, client, cfg, openTime)
	firstDone := make(chan error, 1)
	go func() {
		_, err := engine.ExecuteSignal(context.Background(), validSignal(func(s *contracts.Signal) { s.Quantity = 6; s.Price = 1000; s.IdempotencyKey = "reserve-a" }))
		firstDone <- err
	}()
	<-client.started
	_, err := engine.ExecuteSignal(context.Background(), validSignal(func(s *contracts.Signal) {
		s.Symbol = "TCS"
		s.Quantity = 6
		s.Price = 1000
		s.IdempotencyKey = "reserve-b"
	}))
	var rejection *contracts.Rejection
	if !errors.As(err, &rejection) || rejection.Code != contracts.ReasonMaxPosition {
		t.Fatalf("second order error = %v, want cash rejection", err)
	}
	close(client.release)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	if client.count() != 1 {
		t.Fatalf("broker calls = %d, want 1", client.count())
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
			if got := testutil.MetricValue(t, "mft_execution_orders_rejected_total", reg, nil); got != 1 {
				t.Fatalf("orders_rejected_total = %d, want 1", got)
			}
		})
	}
}

func TestExecuteSignalRejectsStaleTimestamp(t *testing.T) {
	client := &countingClient{}
	cfg := testConfig()
	cfg.Execution.MaxSignalAgeSeconds = 60
	engine, reg := newEngineWithRegistry(t, client, cfg, openTime)
	sig := validSignal(func(s *contracts.Signal) { s.AsOf = openTime.Add(-time.Minute - time.Second) })
	_, err := engine.ExecuteSignal(context.Background(), sig)
	var rejection *contracts.Rejection
	if !errors.As(err, &rejection) || rejection.Code != contracts.ReasonStaleSignal {
		t.Fatalf("error = %v, want stale signal rejection", err)
	}
	if client.count() != 0 {
		t.Fatalf("stale request reached broker %d times", client.count())
	}
	if got := rejectionCount(t, reg, contracts.ReasonStaleSignal); got != 1 {
		t.Fatalf("stale rejection count = %d, want 1", got)
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
		if err := book.ApplyFill(contracts.Signal{Symbol: symbol, Side: contracts.SideBuy}, quantity, float64(buyPrice), openTime); err != nil {
			panic(err)
		}
		if err := book.ApplyFill(contracts.Signal{Symbol: symbol, Side: contracts.SideSell}, quantity, float64(sellPrice), openTime); err != nil {
			panic(err)
		}
		return book
	}
}

func mustBookApply(book *risk.Book, sig contracts.Signal) {
	if err := book.Apply(sig); err != nil {
		panic("seed fill could not be applied: " + err.Error())
	}
}
