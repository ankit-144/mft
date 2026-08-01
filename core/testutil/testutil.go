// Package testutil provides shared helpers for MFT unit and integration
// tests: in-memory logger, mock broker streamer/client, and FX test app
// utilities.
package testutil

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// NewLogger returns a discard zap logger for tests.
func NewLogger() *zap.Logger {
	return zap.NewNop()
}

// NewRegistry returns an isolated Prometheus registry for tests.
func NewRegistry() *prometheus.Registry {
	return prometheus.NewRegistry()
}

// MockStreamer implements broker.Streamer, emitting a scripted sequence of
// ticks and recording the subscribed symbols.
type MockStreamer struct {
	Ticks   []broker.Tick
	Symbols []string
}

// Stream emits all queued ticks then blocks until ctx is cancelled.
func (m *MockStreamer) Stream(ctx context.Context, symbols []string, out chan<- broker.Tick) error {
	m.Symbols = symbols
	for _, t := range m.Ticks {
		select {
		case out <- t:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	<-ctx.Done()
	return ctx.Err()
}

// Tick returns a broker.Tick with sensible defaults.
func Tick(symbol string, price float64, ts time.Time) broker.Tick {
	return broker.Tick{Symbol: symbol, Price: price, Timestamp: ts, Volume: 1}
}

// MockClient implements broker.Client for testing order placement.
type MockClient struct {
	// Orders records every PlaceOrder call.
	Orders []OrderCall
	// FailErr, when non-nil, is returned by PlaceOrder.
	FailErr error
	// ID is returned as the order id; defaults to "mock-order".
	ID string
	// counter provides unique order ids per call.
	counter atomic.Int64
}

// OrderCall captures a single PlaceOrder invocation.
type OrderCall struct {
	Symbol   string
	Side     string
	Quantity int
	Price    float64
}

// PlaceOrder records the call and returns a mock order id.
func (m *MockClient) PlaceOrder(_ context.Context, symbol, side string, quantity int, price float64) (string, error) {
	if m.FailErr != nil {
		return "", m.FailErr
	}
	m.counter.Add(1)
	m.Orders = append(m.Orders, OrderCall{
		Symbol:   symbol,
		Side:     side,
		Quantity: quantity,
		Price:    price,
	})
	oid := m.ID
	if oid == "" {
		oid = "mock-order"
	}
	return oid, nil
}

// Reset clears recorded orders and the id counter.
func (m *MockClient) Reset() {
	m.Orders = nil
	m.counter.Store(0)
}

// MetricValue gathers the current integer value of the metric named name from
// reg after fn runs, returning 0 when the metric is absent.
func MetricValue(t *testing.T, name string, reg *prometheus.Registry, fn func()) int {
	t.Helper()
	if fn != nil {
		fn()
	}
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() != name || len(f.GetMetric()) == 0 {
			continue
		}
		m := f.GetMetric()[0]
		if c := m.GetCounter(); c != nil {
			return int(c.GetValue())
		}
		if g := m.GetGauge(); g != nil {
			return int(g.GetValue())
		}
	}
	return 0
}
