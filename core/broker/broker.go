// Package broker contains the Zerodha Kite Connect connector: the instrument master,
// the WebSocket tick streamer, and the REST order client.
package broker

import (
	"context"
	"fmt"
	"strings"

	"github.com/mft/core/contracts"
)

// Tick is a raw market data tick delivered by the broker stream.
type Tick = contracts.Tick

// Streamer connects to the broker WebSocket and delivers ticks.
type Streamer interface {
	Stream(ctx context.Context, symbols []string, out chan<- Tick) error
}

// Client executes orders on the broker.
type Client interface {
	PlaceOrder(ctx context.Context, symbol string, side string, quantity int, price float64) (string, error)
}

// OrderClient is the order surface defined in docs/contracts.md §2.
type OrderClient interface {
	// PlaceOrder submits an order and returns the broker order id.
	PlaceOrder(ctx context.Context, req contracts.OrderRequest) (string, error)
	// CancelOrder withdraws a working order by id.
	CancelOrder(ctx context.Context, orderID string) error
	// GetPositions returns the current net positions.
	GetPositions(ctx context.Context) ([]contracts.Position, error)
}

// orderClient adapts the Kite connector to OrderClient.
type orderClient struct {
	kite *Kite
}

// NewOrderClient adapts a Kite connector to the docs/contracts.md §2 order surface,
// translating contracts.OrderRequest into the connector call.
func NewOrderClient(kite *Kite) OrderClient { return &orderClient{kite: kite} }

// PlaceOrder implements OrderClient.
func (o *orderClient) PlaceOrder(ctx context.Context, req contracts.OrderRequest) (string, error) {
	typ := strings.ToUpper(strings.TrimSpace(req.Type))
	if typ == "" {
		typ = contracts.OrderTypeMarket
		if req.Price > 0 {
			typ = contracts.OrderTypeLimit
		}
	}
	return o.kite.placeOrder(ctx, req, typ, "")
}

// PlaceOrderTagged submits an order with a caller-supplied reconciliation tag.
func (o *orderClient) PlaceOrderTagged(ctx context.Context, req contracts.OrderRequest, tag string) (string, error) {
	return o.kite.PlaceOrderTagged(ctx, req, tag)
}

// CancelOrder implements OrderClient.
func (o *orderClient) CancelOrder(ctx context.Context, orderID string) error {
	return o.kite.CancelOrder(ctx, orderID)
}

// GetPositions implements OrderClient.
func (o *orderClient) GetPositions(ctx context.Context) ([]contracts.Position, error) {
	return o.kite.GetPositions(ctx)
}

// GetOrder returns the latest broker state for one order.
func (o *orderClient) GetOrder(ctx context.Context, orderID string) (Order, error) {
	return o.kite.GetOrder(ctx, orderID)
}

// GetOrders returns the current session's broker order book.
func (o *orderClient) GetOrders(ctx context.Context) ([]Order, error) {
	return o.kite.GetOrders(ctx)
}

// GetLastPrices returns current LTP marks for configured instruments.
func (o *orderClient) GetLastPrices(ctx context.Context, symbols []string) (map[string]float64, error) {
	return o.kite.GetLastPrices(ctx, symbols)
}

// Compile-time proof that the connector still satisfies every surface the FX graph and
// the shared mocks depend on.
var (
	_ Streamer        = (*Kite)(nil)
	_ Client          = (*Kite)(nil)
	_ OrderClient     = (*orderClient)(nil)
	_ OrderReconciler = (*Kite)(nil)
	_ OrderReconciler = (*orderClient)(nil)
	_ QuoteClient     = (*Kite)(nil)
	_ QuoteClient     = (*orderClient)(nil)
)

// errNoToken is returned when a symbol is present in config but absent from the
// instrument dump, so the failure names the config as the culprit.
func errNoToken(symbols []string) error {
	return fmt.Errorf("instrument dump has no cash instrument for %s: %w",
		strings.Join(symbols, ", "), ErrInstrumentNotFound)
}
