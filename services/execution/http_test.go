package execution

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/log"
	"github.com/mft/core/testutil"
	"github.com/mft/services/execution/risk"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
)

// newHandlerFor builds the execution handler over a real engine whose risk
// clock is pinned to now, plus the counting broker in front of it and the
// registry the handler registered its own metrics on.
//
// Pinned time is what keeps the market-hours check out of wall-clock time:
// without it the suite passes or fails depending on when it happens to run.
func newHandlerFor(
	t *testing.T,
	client *countingClient,
	cfg *config.Config,
	now time.Time,
	mutate func(*risk.Policy),
) (http.Handler, *prometheus.Registry) {
	t.Helper()
	policy := risk.FromConfig(cfg.Execution)
	if mutate != nil {
		mutate(&policy)
	}
	policy.Clock = func() time.Time { return now }

	reg := testutil.NewRegistry()
	engine := newEngine(client, fluxkv.New(), cfg, reg, testutil.NewLogger(), policy, policy.Clock)
	return NewHandler(engine, &healthBroker{}, reg, testutil.NewLogger()), reg
}

// post sends a JSON body to path and returns the recorded response.
func post(t *testing.T, h http.Handler, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// get sends a bodyless GET to path and returns the recorded response.
func get(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec
}

// decodeBody unmarshals a recorded response body, failing the test if it is
// not the JSON the endpoint promises.
func decodeBody(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body %q: %v", rec.Body.String(), err)
	}
	return body
}

// signalJSON renders a signal as the wire body of POST /v1/signals.
func signalJSON(t *testing.T, sig contracts.Signal) string {
	t.Helper()
	body, err := json.Marshal(sig)
	if err != nil {
		t.Fatalf("marshal signal: %v", err)
	}
	return string(body)
}

// labelledCounter reads one label set of a counter vec, so a test can assert
// on a specific route or error code rather than on whichever series the
// registry happens to order first. Label sets are matched by name, not by
// position, because Prometheus sorts them.
func labelledCounter(t *testing.T, reg *prometheus.Registry, name string, labels ...string) int {
	t.Helper()
	if len(labels)%2 != 0 {
		t.Fatal("labels must be name/value pairs")
	}
	want := map[string]string{}
	for i := 0; i < len(labels); i += 2 {
		want[labels[i]] = labels[i+1]
	}

	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() != name {
			continue
		}
		for _, m := range f.GetMetric() {
			got := map[string]string{}
			for _, l := range m.GetLabel() {
				got[l.GetName()] = l.GetValue()
			}
			if maps.Equal(got, want) {
				return int(m.GetCounter().GetValue())
			}
		}
	}
	return 0
}

// placedOrders reads the orders the counting broker recorded.
func placedOrders(c *countingClient) []testutil.OrderCall {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]testutil.OrderCall(nil), c.orders...)
}

func TestOrderTypeFor(t *testing.T) {
	cases := []struct {
		name     string
		declared string
		price    float64
		want     string
	}{
		{"no type, no price is market", "", 0, contracts.OrderTypeMarket},
		{"no type, a price is limit", "", 2934.5, contracts.OrderTypeLimit},
		{"declared market", "MARKET", 0, contracts.OrderTypeMarket},
		{"declared market, lower case", "market", 0, contracts.OrderTypeMarket},
		{"declared limit", "limit", 2934.5, contracts.OrderTypeLimit},
		{"declared market wins over price", "MARKET", 100, contracts.OrderTypeMarket},
		{"declared limit wins over a zero price", "LIMIT", 0, contracts.OrderTypeLimit},
		{"unrecognised falls back to the price", "SLIPPAGE", 0, contracts.OrderTypeMarket},
		{"unrecognised falls back to the price, priced", "SLIPPAGE", 10, contracts.OrderTypeLimit},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := orderTypeFor(tc.declared, tc.price); got != tc.want {
				t.Fatalf("orderTypeFor(%q, %v) = %s, want %s", tc.declared, tc.price, got, tc.want)
			}
		})
	}
}

// TestResponseCode maps only actual observed order state to HTTP acceptance.
func TestResponseCode(t *testing.T) {
	cases := []struct {
		state    string
		wantCode int
	}{
		{contracts.OrderStatusOpen, http.StatusAccepted},
		{contracts.OrderStatusPartial, http.StatusAccepted},
		{contracts.OrderStatusUnknown, http.StatusServiceUnavailable},
		{contracts.OrderStatusFilled, http.StatusOK},
		{contracts.OrderStatusCancelled, http.StatusOK},
	}
	for _, tc := range cases {
		t.Run(tc.state, func(t *testing.T) {
			if code := responseCode(tc.state); code != tc.wantCode {
				t.Fatalf("responseCode(%s) = %d, want %d", tc.state, code, tc.wantCode)
			}
		})
	}
}

// TestHandlerSignalAccepted is the happy path: risk passes, one order is
// placed, and the body is the shape docs/contracts.md §7 documents.
func TestHandlerSignalAccepted(t *testing.T) {
	client := &countingClient{}
	h, reg := newHandlerFor(t, client, testConfig(), openTime, nil)

	sig := validSignal(func(s *contracts.Signal) { s.Score = 0.72 })
	rec := post(t, h, "/v1/signals", signalJSON(t, sig))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
	body := decodeBody(t, rec)
	if body["order_id"] != "mock-order" {
		t.Fatalf("order_id = %v, want mock-order", body["order_id"])
	}
	if body["status"] != contracts.OrderStatusFilled {
		t.Fatalf("status = %v, want %s", body["status"], contracts.OrderStatusFilled)
	}
	if body["score"] != 0.72 {
		t.Fatalf("score = %v, want 0.72", body["score"])
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}
	if got := labelledCounter(t, reg, "mft_execution_http_requests_total",
		"route", "signals", "method", http.MethodPost, "status", "200"); got != 1 {
		t.Fatalf("signals 200 counter = %d, want 1", got)
	}
	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg, nil); got != 1 {
		t.Fatalf("orders_placed_total = %d, want 1", got)
	}
}

func TestHandlerAcknowledgementIsNotReportedAsFill(t *testing.T) {
	client := &countingClient{state: broker.OrderOpen}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)
	rec := post(t, h, "/v1/signals", signalJSON(t, validSignal()))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202; body %s", rec.Code, rec.Body.String())
	}
	body := decodeBody(t, rec)
	if body["order_id"] != "mock-order" || body["status"] != contracts.OrderStatusOpen {
		t.Fatalf("body = %v, want OPEN", body)
	}
}

// TestHandlerOrdersAccepted walks the shape of a direct placement: a limit
// order is working at the exchange the moment the broker acknowledges it, so
// the contract's answer is 200 OPEN.
func TestHandlerOrdersAccepted(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		wantCode  int
		wantState string
	}{
		{"explicit limit", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":2934.5,"type":"LIMIT","idempotency_key":"http-limit-1"}`,
			http.StatusOK, contracts.OrderStatusFilled},
		{"limit inferred from the price", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":2934.5,"idempotency_key":"http-limit-2"}`,
			http.StatusOK, contracts.OrderStatusFilled},
		{"type is not required", `{"symbol":"RELIANCE","side":"BUY","quantity":4,"price":2900,"idempotency_key":"http-limit-3"}`,
			http.StatusOK, contracts.OrderStatusFilled},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := &countingClient{}
			h, reg := newHandlerFor(t, client, testConfig(), openTime, nil)

			rec := post(t, h, "/v1/orders", tc.body)
			if rec.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d; body %s", rec.Code, tc.wantCode, rec.Body.String())
			}
			body := decodeBody(t, rec)
			if body["order_id"] != "mock-order" || body["status"] != tc.wantState {
				t.Fatalf("body = %v, want order_id mock-order and status %s", body, tc.wantState)
			}
			if client.count() != 1 {
				t.Fatalf("broker called %d times, want 1", client.count())
			}
			if got := labelledCounter(t, reg, "mft_execution_http_requests_total",
				"route", "orders", "method", http.MethodPost, "status", "200"); got != 1 {
				t.Fatalf("orders 200 counter = %d, want 1", got)
			}
		})
	}
}

func TestHandlerMarketOrderUsesCurrentMarkForRisk(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)
	rec := post(t, h, "/v1/orders", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"type":"MARKET","idempotency_key":"market-1"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want filled 200; body %s", rec.Code, rec.Body.String())
	}
	if got := placedOrders(client); len(got) != 1 || got[0].Price != 0 {
		t.Fatalf("market order sent to broker = %+v", got)
	}
}

// TestHandlerSignalRejections walks every reason code in docs/contracts.md §6
// through the transport. Each must surface its own code on a 400, because the
// inference loop branches on it and the rejection metric is keyed on it, and
// each must leave the broker untouched.
func TestHandlerSignalRejections(t *testing.T) {
	cases := []struct {
		name     string
		policy   func(*risk.Policy)
		now      time.Time
		signal   func(*contracts.Signal)
		seed     func(float64) *risk.Book
		preArm   string
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
			name:     contracts.ReasonMaxPositions,
			policy:   func(p *risk.Policy) { p.MaxOpenPositions = 1 },
			signal:   func(s *contracts.Signal) { s.Symbol = "INFY" },
			seed:     seedLong("TCS", 10, 1000),
			wantCode: contracts.ReasonMaxPositions,
		},
		{
			name:     contracts.ReasonMaxDrawdown,
			policy:   func(p *risk.Policy) { p.MaxDrawdownPct = 5 },
			seed:     seedLoss("SEED", 300, 1000, 700),
			wantCode: contracts.ReasonMaxDrawdown,
		},
		{
			name:     contracts.ReasonDailyLoss,
			policy:   func(p *risk.Policy) { p.DailyLossLimit = 25_000; p.MaxDrawdownPct = 50 },
			seed:     seedLoss("SEED", 300, 1000, 700),
			wantCode: contracts.ReasonDailyLoss,
		},
		{
			name:     contracts.ReasonBadQuantity,
			signal:   func(s *contracts.Signal) { s.Quantity = 0 },
			wantCode: contracts.ReasonBadQuantity,
		},
		{
			// A sell with no long position is a short, which v1 does not place.
			name:     contracts.ReasonBadQuantity + " short",
			signal:   func(s *contracts.Signal) { s.Side = contracts.SideSell },
			wantCode: contracts.ReasonBadQuantity,
		},
		{
			name:     contracts.ReasonMarketClosed,
			now:      openTime.Add(8 * time.Hour),
			wantCode: contracts.ReasonMarketClosed,
		},
		{
			name:     contracts.ReasonDebounced,
			signal:   func(s *contracts.Signal) { s.Symbol = "TCS" },
			preArm:   risk.DebounceKey("TCS", contracts.SideBuy),
			wantCode: contracts.ReasonDebounced,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := &countingClient{}
			cfg := testConfig()
			policy := risk.FromConfig(cfg.Execution)
			if tc.policy != nil {
				tc.policy(&policy)
			}
			now := tc.now
			if now.IsZero() {
				now = openTime
			}
			policy.Clock = func() time.Time { return now }

			cache := fluxkv.New()
			reg := testutil.NewRegistry()
			engine := newEngine(client, cache, cfg, reg, testutil.NewLogger(), policy, policy.Clock)
			if tc.seed != nil {
				engine.book = tc.seed(cfg.Execution.Capital)
			}
			if tc.preArm != "" {
				cache.Set(tc.preArm, "mock-order", time.Hour)
			}
			h := NewHandler(engine, &healthBroker{}, reg, testutil.NewLogger())

			sig := validSignal(func(s *contracts.Signal) { s.IdempotencyKey = tc.name + ":key" })
			if tc.signal != nil {
				tc.signal(&sig)
			}
			rec := post(t, h, "/v1/signals", signalJSON(t, sig))

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body %s", rec.Code, rec.Body.String())
			}
			body := decodeBody(t, rec)
			if body["error"] != tc.wantCode {
				t.Fatalf("error = %v, want %s", body["error"], tc.wantCode)
			}
			if msg, _ := body["message"].(string); msg == "" {
				t.Fatalf("body %v carries no message", body)
			}
			if client.count() != 0 {
				t.Fatalf("broker was called %d times for a rejected signal, want 0", client.count())
			}
			if got := rejectionCount(t, reg, tc.wantCode); got != 1 {
				t.Fatalf("%s = %d, want 1", tc.wantCode, got)
			}
			if got := labelledCounter(t, reg, "mft_execution_http_faults_total", "code", tc.wantCode); got != 1 {
				t.Fatalf("http_faults_total{%s} = %d, want 1", tc.wantCode, got)
			}
		})
	}
}

// TestHandlerSignalDuplicate is the load-bearing test for this component:
// inference retries, and a replayed key must place exactly one order, answer
// 409, and hand back the order id the first attempt produced.
func TestHandlerSignalDuplicate(t *testing.T) {
	client := &countingClient{}
	h, reg := newHandlerFor(t, client, testConfig(), openTime, nil)
	body := signalJSON(t, validSignal())

	first := post(t, h, "/v1/signals", body)
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200; body %s", first.Code, first.Body.String())
	}
	firstID := decodeBody(t, first)["order_id"]

	for i := range 3 {
		rec := post(t, h, "/v1/signals", body)
		if rec.Code != http.StatusConflict {
			t.Fatalf("replay %d status = %d, want 409; body %s", i, rec.Code, rec.Body.String())
		}
		replay := decodeBody(t, rec)
		if replay["error"] != contracts.ReasonDuplicate {
			t.Fatalf("replay %d error = %v, want %s", i, replay["error"], contracts.ReasonDuplicate)
		}
		if replay["order_id"] != firstID {
			t.Fatalf("replay %d order_id = %v, want the original %v", i, replay["order_id"], firstID)
		}
	}

	if client.count() != 1 {
		t.Fatalf("broker called %d times for one key replayed four times, want 1", client.count())
	}
	if got := rejectionCount(t, reg, contracts.ReasonDuplicate); got != 3 {
		t.Fatalf("%s = %d, want 3", contracts.ReasonDuplicate, got)
	}
	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg, nil); got != 1 {
		t.Fatalf("orders_placed_total = %d, want 1", got)
	}
	if got := labelledCounter(t, reg, "mft_execution_http_requests_total",
		"route", "signals", "method", http.MethodPost, "status", "409"); got != 3 {
		t.Fatalf("signals 409 counter = %d, want 3", got)
	}
}

// TestHandlerSignalDuplicateConcurrent proves the exactly-once property holds
// through the transport under a retry storm, not just for sequential retries.
func TestHandlerSignalDuplicateConcurrent(t *testing.T) {
	client := &countingClient{}
	h, reg := newHandlerFor(t, client, testConfig(), openTime, nil)
	body := signalJSON(t, validSignal())

	const racers = 16
	var wg sync.WaitGroup
	codes := make([]int, racers)
	for i := range racers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			codes[i] = post(t, h, "/v1/signals", body).Code
		}()
	}
	wg.Wait()
	accepted, conflicts := 0, 0
	for _, code := range codes {
		switch code {
		case http.StatusOK:
			accepted++
		case http.StatusConflict:
			conflicts++
		default:
			t.Fatalf("unexpected status %d", code)
		}
	}
	if accepted != 1 || conflicts != racers-1 {
		t.Fatalf("accepted %d and conflicted %d, want 1 and %d", accepted, conflicts, racers-1)
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times under %d concurrent retries of one key, want 1",
			client.count(), racers)
	}
	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg, nil); got != 1 {
		t.Fatalf("orders_placed_total = %d, want 1", got)
	}
}

// TestHandlerSignalMalformed covers everything the transport owns: a body it
// cannot parse, a field it does not recognise, and a required field that is
// missing. None of them may reach the risk gate, and none of them may appear
// in the rejection metric.
func TestHandlerSignalMalformed(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"not json", `{"symbol":`},
		{"empty body", ``},
		{"a bare string", `"RELIANCE"`},
		{"two objects", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"idempotency_key":"a"}{"symbol":"TCS"}`},
		{"unknown field", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"idempotency_key":"a","conviction":0.9}`},
		{"misspelled idempotency key", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"idempotency_key":"a","idem_key":"b"}`},
		{"fractional quantity", `{"symbol":"RELIANCE","side":"BUY","quantity":10.5,"price":1000,"idempotency_key":"a"}`},
		{"unparseable timestamp", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"as_of":"yesterday","idempotency_key":"a"}`},
		{"no symbol", `{"side":"BUY","quantity":10,"price":1000,"idempotency_key":"a"}`},
		{"no side", `{"symbol":"RELIANCE","quantity":10,"price":1000,"idempotency_key":"a"}`},
		{"unknown side", `{"symbol":"RELIANCE","side":"HOLD","quantity":10,"price":1000,"idempotency_key":"a"}`},
		{"no price", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"idempotency_key":"a"}`},
		{"negative price", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":-1,"idempotency_key":"a"}`},
		{"no idempotency key", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000}`},
		{"blank idempotency key", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"idempotency_key":"   "}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := &countingClient{}
			h, reg := newHandlerFor(t, client, testConfig(), openTime, nil)

			rec := post(t, h, "/v1/signals", tc.body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body %s", rec.Code, rec.Body.String())
			}
			body := decodeBody(t, rec)
			if body["error"] != codeBadRequest {
				t.Fatalf("error = %v, want %s", body["error"], codeBadRequest)
			}
			if client.count() != 0 {
				t.Fatalf("broker was called %d times, want 0", client.count())
			}
			// There is no reason code for a request that was never a decision,
			// so nothing may land in the rejection metric.
			families, err := reg.Gather()
			if err != nil {
				t.Fatalf("gather: %v", err)
			}
			for _, f := range families {
				if f.GetName() == "mft_execution_rejections_total" {
					t.Fatalf("a malformed request was counted as a risk rejection: %v", f.GetMetric())
				}
			}
		})
	}
}

// TestHandlerSignalNormalisesWireForms checks the one liberty the transport
// takes: instruments and sides are matched case-insensitively by the broker
// and exactly by the risk gate, so the transport is where the two meet.
func TestHandlerSignalNormalisesWireForms(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

	rec := post(t, h, "/v1/signals",
		`{"symbol":" reliance ","side":"buy","quantity":10,"price":1000,"as_of":"2026-09-29T05:00:00Z","idempotency_key":" k "}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
	orders := placedOrders(client)
	if len(orders) != 1 {
		t.Fatalf("broker called %d times, want 1", len(orders))
	}
	if orders[0].Symbol != "RELIANCE" || orders[0].Side != contracts.SideBuy {
		t.Fatalf("broker saw %+v, want RELIANCE BUY", orders[0])
	}
}

func TestHandlerOrdersMalformed(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"not json", `{"symbol":`},
		{"unknown field", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"product":"MIS"}`},
		{"no symbol", `{"side":"BUY","quantity":10,"price":1000}`},
		{"unknown side", `{"symbol":"RELIANCE","side":"HOLD","quantity":10,"price":1000}`},
		{"zero quantity", `{"symbol":"RELIANCE","side":"BUY","quantity":0,"price":1000}`},
		{"negative quantity", `{"symbol":"RELIANCE","side":"BUY","quantity":-10,"price":1000}`},
		{"negative price", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":-5}`},
		{"unknown type", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"type":"SLIPPAGE"}`},
		{"market carrying a price", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"type":"MARKET"}`},
		{"limit without a price", `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":0,"type":"LIMIT"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := &countingClient{}
			h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

			rec := post(t, h, "/v1/orders", tc.body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body %s", rec.Code, rec.Body.String())
			}
			if body := decodeBody(t, rec); body["error"] != codeBadRequest {
				t.Fatalf("error = %v, want %s", body["error"], codeBadRequest)
			}
			if client.count() != 0 {
				t.Fatalf("broker was called %d times, want 0", client.count())
			}
		})
	}
}

// TestHandlerOrdersDuplicate proves the derived key on the legacy loose-field
// entry point is surfaced the same way: a replay is a 409 carrying the original
// order id, not a second order.
func TestHandlerOrdersDuplicate(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)
	const body = `{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,"idempotency_key":"same-order-key"}`

	if rec := post(t, h, "/v1/orders", body); rec.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
	rec := post(t, h, "/v1/orders", body)
	if rec.Code != http.StatusConflict {
		t.Fatalf("replay status = %d, want 409; body %s", rec.Code, rec.Body.String())
	}
	replay := decodeBody(t, rec)
	if replay["error"] != contracts.ReasonDuplicate || replay["order_id"] != "mock-order" {
		t.Fatalf("replay body = %v, want a RISK_DUPLICATE carrying the original order id", replay)
	}
	if client.count() != 1 {
		t.Fatalf("broker called %d times, want 1", client.count())
	}
}

// TestHandlerBrokerFailure checks the classification of a non-rejection: a
// placement that never happened must not read as a 400, or inference will
// retry a request that was not its fault.
func TestHandlerBrokerFailure(t *testing.T) {
	client := &countingClient{err: errors.New("kite 502")}
	h, reg := newHandlerFor(t, client, testConfig(), openTime, nil)

	rec := post(t, h, "/v1/signals", signalJSON(t, validSignal()))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body %s", rec.Code, rec.Body.String())
	}
	if body := decodeBody(t, rec); body["error"] != codeOutcomeUnknown {
		t.Fatalf("error = %v, want %s", body["error"], codeOutcomeUnknown)
	}
	if client.count() != 0 {
		t.Fatalf("a failed placement was counted as an order: %d calls", client.count())
	}
	if got := labelledCounter(t, reg, "mft_execution_http_faults_total", "code", codeOutcomeUnknown); got != 1 {
		t.Fatalf("http_faults_total{%s} = %d, want 1", codeOutcomeUnknown, got)
	}
}

// TestHandlerBodyLimit proves the body cap is enforced before anything is
// decoded or risk-checked.
func TestHandlerBodyLimit(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

	oversized := fmt.Sprintf(`{"symbol":"RELIANCE","side":"BUY","quantity":10,"price":1000,`+
		`"model":%q,"idempotency_key":"big"}`, strings.Repeat("x", maxRequestBody+1))
	rec := post(t, h, "/v1/signals", oversized)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413; body %s", rec.Code, rec.Body.String())
	}
	if body := decodeBody(t, rec); body["error"] != codePayloadTooBig {
		t.Fatalf("error = %v, want %s", body["error"], codePayloadTooBig)
	}
	if client.count() != 0 {
		t.Fatalf("broker was called %d times, want 0", client.count())
	}
}

// TestHandlerBodyAtLimit proves the cap is a ceiling and not a rejection of
// the documented request shape: a body that fills it is still served.
func TestHandlerBodyAtLimit(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

	rec := post(t, h, "/v1/signals", signalJSON(t, validSignal(func(s *contracts.Signal) {
		s.Model = strings.Repeat("m", maxRequestBody/2)
	})))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
	if rec.Body.Len() > maxRequestBody {
		t.Fatalf("response is %d bytes, over the %d byte body cap", rec.Body.Len(), maxRequestBody)
	}
}

// TestHandlerRecoversFromPanic proves the barrier is scoped to one connection:
// a panic inside the broker is answered with the API's error shape, the
// process survives, and the next request is served normally.
func TestHandlerRecoversFromPanic(t *testing.T) {
	client := &panicClient{}
	policy := risk.FromConfig(testConfig().Execution)
	policy.Clock = func() time.Time { return openTime }
	reg := testutil.NewRegistry()
	engine := newEngine(client, fluxkv.New(), testConfig(), reg, testutil.NewLogger(), policy, policy.Clock)
	h := NewHandler(engine, &healthBroker{}, reg, testutil.NewLogger())

	rec := post(t, h, "/v1/signals", signalJSON(t, validSignal()))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body %s", rec.Code, rec.Body.String())
	}
	body := decodeBody(t, rec)
	if body["error"] != codeOutcomeUnknown || body["status"] != contracts.OrderStatusUnknown {
		t.Fatalf("error = %v / status %v, want unknown broker outcome", body["error"], body["status"])
	}
	// The panic value is internal detail; it is logged, never returned.
	if strings.Contains(rec.Body.String(), "broker exploded") {
		t.Fatalf("the panic value leaked into the response: %s", rec.Body.String())
	}
	if got := labelledCounter(t, reg, "mft_execution_http_faults_total", "code", codeOutcomeUnknown); got != 1 {
		t.Fatalf("http_faults_total{%s} = %d, want 1", codeOutcomeUnknown, got)
	}

	// The process is still serving.
	if rec := get(t, h, "/v1/portfolio"); rec.Code != http.StatusOK {
		t.Fatalf("status after a panic = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
}

func TestHandlerPortfolio(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

	rec := get(t, h, "/v1/portfolio")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
	var flat contracts.Portfolio
	if err := json.Unmarshal(rec.Body.Bytes(), &flat); err != nil {
		t.Fatalf("decode %s: %v", rec.Body.String(), err)
	}
	if flat.Cash != 1_000_000 || len(flat.OpenPositions) != 0 {
		t.Fatalf("flat portfolio = %+v, want 1000000 cash and no positions", flat)
	}
	if !strings.Contains(rec.Body.String(), `"open_positions":{}`) {
		t.Fatalf("a flat book should serialise its positions as {}, got %s", rec.Body.String())
	}

	if rec := post(t, h, "/v1/signals", signalJSON(t, validSignal())); rec.Code != http.StatusOK {
		t.Fatalf("signal status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}

	rec = get(t, h, "/v1/portfolio")
	var held contracts.Portfolio
	if err := json.Unmarshal(rec.Body.Bytes(), &held); err != nil {
		t.Fatalf("decode %s: %v", rec.Body.String(), err)
	}
	if held.OpenPositions["RELIANCE"] != 10 {
		t.Fatalf("positions = %+v, want RELIANCE 10", held.OpenPositions)
	}
}

func TestLiveHTTPRequiresBearerToken(t *testing.T) {
	client := &countingClient{}
	cfg := testConfig()
	cfg.Execution.APIToken = "local-secret"
	policy := risk.FromConfig(cfg.Execution)
	policy.Clock = func() time.Time { return openTime }
	reg := testutil.NewRegistry()
	engine := newEngine(client, fluxkv.New(), cfg, reg, testutil.NewLogger(), policy, policy.Clock)
	h := NewHandler(engine, client, reg, testutil.NewLogger(), &cfg.Execution)
	if rec := get(t, h, "/v1/portfolio"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated live request status = %d, want 401", rec.Code)
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/portfolio", nil)
	req.Header.Set("Authorization", "Bearer local-secret")
	resp := httptest.NewRecorder()
	h.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("authenticated live request status = %d, want 200", resp.Code)
	}
}

func TestHandlerHealth(t *testing.T) {
	cases := []struct {
		name     string
		broker   broker.OrderClient
		wantCode int
	}{
		{"broker reachable", &healthBroker{}, http.StatusOK},
		{"broker session refused", &healthBroker{err: errors.New("kite: access_token is not configured")}, http.StatusServiceUnavailable},
		{"no broker bound", nil, http.StatusServiceUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := NewHandler(&apiStub{}, tc.broker, testutil.NewRegistry(), testutil.NewLogger())

			rec := get(t, h, "/v1/health")
			if rec.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d; body %s", rec.Code, tc.wantCode, rec.Body.String())
			}
			var report struct {
				Status string `json:"status"`
				Checks []struct {
					Name   string `json:"name"`
					Status string `json:"status"`
				} `json:"checks"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &report); err != nil {
				t.Fatalf("decode %s: %v", rec.Body.String(), err)
			}
			if report.Status == "" || len(report.Checks) != 1 || report.Checks[0].Name != "broker" {
				t.Fatalf("report = %+v, want a status and one broker check", report)
			}
		})
	}
}

func TestPaperHealthSkipsBrokerAndUsesConfiguredToken(t *testing.T) {
	for _, tc := range []struct {
		name  string
		token string
	}{
		{name: "local paper without token"},
		{name: "token protected paper", token: "paper-secret"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.ExecutionConfig{PaperTrading: true, APIToken: tc.token}
			h := NewHandler(&apiStub{}, forbiddenHealthBroker{}, testutil.NewRegistry(), testutil.NewLogger(), cfg)
			req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
			if tc.token != "" {
				unauthorized := httptest.NewRecorder()
				h.ServeHTTP(unauthorized, req)
				if unauthorized.Code != http.StatusUnauthorized {
					t.Fatalf("paper health without token = %d, want 401", unauthorized.Code)
				}
				req.Header.Set("Authorization", "Bearer "+tc.token)
			}
			resp := httptest.NewRecorder()
			h.ServeHTTP(resp, req)
			if resp.Code != http.StatusOK {
				t.Fatalf("paper health status = %d, want 200; body=%s", resp.Code, resp.Body.String())
			}
			var report struct {
				Checks []struct {
					Name string `json:"name"`
				} `json:"checks"`
			}
			if err := json.Unmarshal(resp.Body.Bytes(), &report); err != nil {
				t.Fatal(err)
			}
			if len(report.Checks) != 0 {
				t.Fatalf("paper health performed dependency probes: %+v", report.Checks)
			}
		})
	}
}

func TestHandlerUnknownRoute(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

	rec := get(t, h, "/v1/nope")
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body %s", rec.Code, rec.Body.String())
	}
	if body := decodeBody(t, rec); body["error"] != codeNotFound {
		t.Fatalf("error = %v, want %s", body["error"], codeNotFound)
	}
}

// TestHandlerPropagatesRequestID checks C9's middleware is wired in: a
// correlation id sent by inference is echoed back and stamped on the log, so
// one rejected signal is one story across two log streams.
func TestHandlerPropagatesRequestID(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)
	const id = "RELIANCE:BUY:20260929T1031"

	req := httptest.NewRequest(http.MethodPost, "/v1/signals", strings.NewReader(signalJSON(t, validSignal())))
	req.Header.Set(log.RequestIDHeader, id)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get(log.RequestIDHeader); got != id {
		t.Fatalf("response %s = %q, want %q", log.RequestIDHeader, got, id)
	}
}

// TestStartHTTPServerIsConstructibleByFX guards the constructor signature.
// StartHTTPServer grew a broker client and a registry when the API gained a
// health probe and metrics, and a dependency the graph cannot supply is a
// panic at startup rather than a compile error — the app builds fine until
// someone runs it.
func TestStartHTTPServerIsConstructibleByFX(t *testing.T) {
	cfg := testConfig()
	cfg.Execution.Addr = "127.0.0.1:0"
	reg := testutil.NewRegistry()
	policy := risk.FromConfig(cfg.Execution)
	engine := newEngine(&countingClient{}, fluxkv.New(), cfg, reg, testutil.NewLogger(), policy, policy.Clock)

	app := fx.New(
		fx.NopLogger,
		fx.Provide(
			func() *config.Config { return cfg },
			func() *Engine { return engine },
			func() broker.OrderClient { return &healthBroker{} },
			func() *prometheus.Registry { return reg },
			testutil.NewLogger,
		),
		fx.Invoke(StartHTTPServer),
	)
	if err := app.Err(); err != nil {
		t.Fatalf("execution module graph does not resolve: %v", err)
	}
}

func TestHandlerMethodNotAllowed(t *testing.T) {
	client := &countingClient{}
	h, _ := newHandlerFor(t, client, testConfig(), openTime, nil)

	rec := get(t, h, "/v1/signals")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
	if client.count() != 0 {
		t.Fatalf("broker was called %d times, want 0", client.count())
	}
}

// apiStub is a minimal Executor. Only the health endpoint is exercised through
// it, so it does not need the machinery the real engine already has under
// test.
type apiStub struct {
	orderID string
	err     error
}

func (s *apiStub) ExecuteSignal(context.Context, contracts.Signal) (string, error) {
	return s.orderID, s.err
}

func (s *apiStub) Execute(context.Context, string, string, int, float64) (string, error) {
	return s.orderID, s.err
}

func (s *apiStub) ExecuteOrder(context.Context, contracts.OrderRequest) (string, error) {
	return s.orderID, s.err
}
func (s *apiStub) OrderStatus(string) contracts.OrderResult {
	return contracts.OrderResult{OrderID: s.orderID, Status: contracts.OrderStatusFilled, FilledQuantity: 1}
}

func (s *apiStub) Portfolio() contracts.Portfolio {
	return contracts.Portfolio{OpenPositions: map[string]int{}}
}

// healthBroker is a broker.OrderClient that answers GetPositions from a
// scripted error. The API never places through it; the engine does, through
// the counting client.
type healthBroker struct {
	err error
}

type forbiddenHealthBroker struct{}

func (forbiddenHealthBroker) PlaceOrder(context.Context, contracts.OrderRequest) (string, error) {
	panic("paper health must not place orders")
}
func (forbiddenHealthBroker) CancelOrder(context.Context, string) error {
	panic("paper health must not call the broker")
}
func (forbiddenHealthBroker) GetPositions(context.Context) ([]contracts.Position, error) {
	panic("paper health must not read broker positions")
}

func (b *healthBroker) PlaceOrder(context.Context, contracts.OrderRequest) (string, error) {
	return "stub-order", b.err
}

func (b *healthBroker) CancelOrder(context.Context, string) error { return b.err }

func (b *healthBroker) GetPositions(context.Context) ([]contracts.Position, error) {
	return []contracts.Position{}, b.err
}

// panicClient is a broker.Client that panics on placement. The placement runs
// on the request goroutine, so a panic inside the connector unwinds through
// the handler exactly as a panic in a handler would.
type panicClient struct {
	calls atomic.Int64
}

func (p *panicClient) PlaceOrder(context.Context, contracts.OrderRequest) (string, error) {
	p.calls.Add(1)
	panic("broker exploded")
}

func (p *panicClient) CancelOrder(context.Context, string) error                  { return nil }
func (p *panicClient) GetPositions(context.Context) ([]contracts.Position, error) { return nil, nil }
