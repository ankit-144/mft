package broker

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"

	"github.com/mft/core/contracts"
)

// PlaceOrder submits an order to Kite and returns the broker order id.
//
// A price of zero places a market order; a non-zero price places a limit
// order, which is how contracts.OrderRequest documents its Price field. The
// symbol is resolved through the instrument master first, so an unknown or
// misspelled symbol fails before any order reaches the exchange.
//
// Deprecated: use NewOrderClient for the contracts.md §2 shape, which carries
// an explicit order type.
func (k *Kite) PlaceOrder(ctx context.Context, symbol string, side string, quantity int, price float64) (string, error) {
	req := contracts.OrderRequest{Symbol: symbol, Side: side, Quantity: quantity, Price: price}
	orderType := contracts.OrderTypeMarket
	if price > 0 {
		orderType = contracts.OrderTypeLimit
	}
	return k.placeOrder(ctx, req, orderType)
}

// placeOrder is the shared implementation behind PlaceOrder and
// orderClient.PlaceOrder: it validates the request, resolves the instrument
// token, and posts the regular-order form Kite expects.
func (k *Kite) placeOrder(ctx context.Context, req contracts.OrderRequest, orderType string) (string, error) {
	if err := k.checkAuth(); err != nil {
		return "", err
	}

	symbol := strings.ToUpper(strings.TrimSpace(req.Symbol))
	if symbol == "" {
		return "", fmt.Errorf("kite order: empty symbol: %w", ErrInvalidOrder)
	}

	direction := strings.ToUpper(strings.TrimSpace(req.Side))
	if direction != contracts.SideBuy && direction != contracts.SideSell {
		return "", fmt.Errorf("kite order %s: side %q must be BUY or SELL: %w", symbol, req.Side, ErrInvalidOrder)
	}
	if req.Quantity <= 0 {
		return "", fmt.Errorf("kite order %s: quantity %d must be positive: %w", symbol, req.Quantity, ErrInvalidOrder)
	}
	if req.Price < 0 {
		return "", fmt.Errorf("kite order %s: negative price %v: %w", symbol, req.Price, ErrInvalidOrder)
	}

	switch orderType {
	case contracts.OrderTypeMarket:
		if req.Price > 0 {
			return "", fmt.Errorf("kite order %s: MARKET order must not carry a price: %w", symbol, ErrInvalidOrder)
		}
	case contracts.OrderTypeLimit:
		if req.Price <= 0 {
			return "", fmt.Errorf("kite order %s: LIMIT order requires a price: %w", symbol, ErrInvalidOrder)
		}
	default:
		return "", fmt.Errorf("kite order %s: type %q must be MARKET or LIMIT: %w", symbol, orderType, ErrInvalidOrder)
	}

	inst, err := k.Instrument(ctx, symbol)
	if err != nil {
		return "", err
	}
	if inst.LotSize > 1 && req.Quantity%inst.LotSize != 0 {
		return "", fmt.Errorf("kite order %s: quantity %d is not a multiple of lot size %d: %w",
			symbol, req.Quantity, inst.LotSize, ErrInvalidOrder)
	}

	k.mu.Lock()
	product := k.product
	k.mu.Unlock()

	form := url.Values{
		"tradingsymbol": {inst.Symbol},
		"exchange":      {inst.Exchange},
		"direction":     {direction},
		"product":       {product},
		"order_type":    {orderType},
		"quantity":      {strconv.Itoa(req.Quantity)},
		"variety":       {"regular"},
	}
	if orderType == contracts.OrderTypeLimit {
		form.Set("price", strconv.FormatFloat(req.Price, 'f', -1, 64))
	}

	body, err := k.do(ctx, httpPost, k.endpoints.ordersURL(), form, 1<<20)
	if err != nil {
		return "", fmt.Errorf("kite order %s: %w", symbol, err)
	}

	var resp struct {
		Status  string `json:"status"`
		OrderID string `json:"order_id"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", fmt.Errorf("kite order %s: decode response: %w: %w", symbol, ErrUnavailable, err)
	}
	if resp.OrderID == "" {
		return "", fmt.Errorf("kite order %s: response carried no order_id: %w", symbol, ErrUnavailable)
	}
	return resp.OrderID, nil
}

// CancelOrder withdraws a working order by its Kite order id.
func (k *Kite) CancelOrder(ctx context.Context, orderID string) error {
	id := strings.TrimSpace(orderID)
	if id == "" {
		return fmt.Errorf("kite cancel: empty order id: %w", ErrOrderNotFound)
	}
	if _, err := k.do(ctx, httpDelete, k.endpoints.orderURL(id), nil, 1<<20); err != nil {
		return fmt.Errorf("kite cancel %s: %w", id, err)
	}
	return nil
}

// kitePosition is one row of the Kite /positions response. The quantity field
// is a float on the wire because the same endpoint also serves commodity and
// currency legs.
type kitePosition struct {
	TradingSymbol   string  `json:"tradingsymbol"`
	InstrumentToken int64   `json:"instrument_token"`
	Exchange        string  `json:"exchange"`
	Quantity        float64 `json:"quantity"`
	AveragePrice    float64 `json:"average_price"`
	Product         string  `json:"product"`
}

// GetPositions returns the current net positions. Net quantities are reported
// as they come from the broker: a negative quantity is a short, and a zero
// quantity is a day position that has been squared off. Filtering them is the
// caller's decision, not the connector's.
func (k *Kite) GetPositions(ctx context.Context) ([]contracts.Position, error) {
	body, err := k.do(ctx, httpGet, k.endpoints.positionsURL(), nil, 1<<20)
	if err != nil {
		return nil, fmt.Errorf("kite positions: %w", err)
	}

	rows, err := decodePositions(body)
	if err != nil {
		return nil, err
	}

	out := make([]contracts.Position, 0, len(rows))
	for _, row := range rows {
		symbol := strings.ToUpper(strings.TrimSpace(row.TradingSymbol))
		if symbol == "" {
			continue
		}
		out = append(out, contracts.Position{
			Symbol:   symbol,
			Quantity: int(row.Quantity),
			AvgPrice: row.AveragePrice,
		})
	}
	return out, nil
}

// decodePositions accepts both shapes Kite has used for /positions: a bare
// JSON array, and an object wrapping the array under "data".
func decodePositions(body []byte) ([]kitePosition, error) {
	var rows []kitePosition
	if err := json.Unmarshal(body, &rows); err == nil {
		return rows, nil
	}

	var wrapped struct {
		Data []kitePosition `json:"data"`
	}
	if err := json.Unmarshal(body, &wrapped); err != nil {
		return nil, fmt.Errorf("kite positions: decode response: %w: %w", ErrUnavailable, err)
	}
	return wrapped.Data, nil
}
