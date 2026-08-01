// Package broker contains interfaces and stub implementations for broker
// connectivity (Zerodha Kite Connect).
package broker

import (
	"context"
	"time"
)

// Tick is a raw market data tick delivered by the broker stream.
type Tick struct {
	Symbol      string
	Price       float64
	Volume      int64
	Timestamp   time.Time
}

// Streamer connects to the broker WebSocket and delivers ticks.
type Streamer interface {
	Stream(ctx context.Context, symbols []string, out chan<- Tick) error
}

// Client executes orders on the broker.
type Client interface {
	PlaceOrder(ctx context.Context, symbol string, side string, quantity int, price float64) (string, error)
}

// Kite is a placeholder implementation of the Zerodha Kite Connect connector.
// Real integration will use the Kite Connect WebSocket and REST APIs.
type Kite struct{}

// NewKite returns a Kite connector.
func NewKite() *Kite {
	return &Kite{}
}

// Stream is a stub that emits no ticks until the real Kite WebSocket is wired.
func (k *Kite) Stream(ctx context.Context, symbols []string, out chan<- Tick) error {
	<-ctx.Done()
	return ctx.Err()
}

// PlaceOrder is a stub that always fails until the Kite REST API is wired.
func (k *Kite) PlaceOrder(ctx context.Context, symbol string, side string, quantity int, price float64) (string, error) {
	return "", ctx.Err()
}
