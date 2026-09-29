// Package broker contains the Zerodha Kite Connect connector: the instrument
// master, the WebSocket tick streamer, and the REST order client.
//
// The frozen domain types live in github.com/mft/core/contracts. See
// docs/contracts.md §2 for the broker contract.
package broker

import (
	"context"
	"fmt"
	"strings"

	"github.com/mft/core/contracts"
)

// Tick is a raw market data tick delivered by the broker stream.
//
// It is an alias of contracts.Tick so that a consumer can move to the frozen
// type without a breaking change, and so that the instrument token and the
// JSON wire tags are available to every consumer of the stream.
type Tick = contracts.Tick

// Streamer connects to the broker WebSocket and delivers ticks.
//
// Stream blocks until ctx is cancelled, then returns ctx.Err(). A nil or
// empty symbols slice means "the instruments declared in broker config".
type Streamer interface {
	Stream(ctx context.Context, symbols []string, out chan<- Tick) error
}

// Client executes orders on the broker.
//
// Deprecated: this is the pre-contracts signature and predates
// contracts.OrderRequest, which carries an explicit order type. It is
// retained because core/fx.go binds it and core/testutil.MockClient
// implements it; use OrderClient instead. Removing it is a foundation
// change that must update core/fx.go, core/testutil and
// services/execution/engine.go together.
type Client interface {
	PlaceOrder(ctx context.Context, symbol string, side string, quantity int, price float64) (string, error)
}

// OrderClient is the order surface frozen in docs/contracts.md §2. *Kite
// implements it via NewOrderClient; the adapter exists only because Client
// above still uses the older PlaceOrder signature.
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

// NewOrderClient adapts a Kite connector to the docs/contracts.md §2 order
// surface, translating contracts.OrderRequest into the connector call.
func NewOrderClient(kite *Kite) OrderClient { return &orderClient{kite: kite} }

// PlaceOrder implements OrderClient. A request with an empty Type is
// inferred from the price: a non-zero price means LIMIT, otherwise MARKET,
// matching the contracts.OrderRequest documentation.
func (o *orderClient) PlaceOrder(ctx context.Context, req contracts.OrderRequest) (string, error) {
	typ := strings.ToUpper(strings.TrimSpace(req.Type))
	if typ == "" {
		typ = contracts.OrderTypeMarket
		if req.Price > 0 {
			typ = contracts.OrderTypeLimit
		}
	}
	return o.kite.placeOrder(ctx, req, typ)
}

// CancelOrder implements OrderClient.
func (o *orderClient) CancelOrder(ctx context.Context, orderID string) error {
	return o.kite.CancelOrder(ctx, orderID)
}

// GetPositions implements OrderClient.
func (o *orderClient) GetPositions(ctx context.Context) ([]contracts.Position, error) {
	return o.kite.GetPositions(ctx)
}

// Compile-time proof that the connector still satisfies every surface the
// FX graph and the shared mocks depend on.
var (
	_ Streamer    = (*Kite)(nil)
	_ Client      = (*Kite)(nil)
	_ OrderClient = (*orderClient)(nil)
)

// errNoToken is returned when a symbol is present in config but absent from
// the instrument dump, so the failure names the config as the culprit.
func errNoToken(symbols []string) error {
	return fmt.Errorf("instrument dump has no cash instrument for %s: %w",
		strings.Join(symbols, ", "), ErrInstrumentNotFound)
}
