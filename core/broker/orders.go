package broker

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/mft/core/contracts"
)

// PlaceOrder submits an order to Kite and returns the broker order id.
func (k *Kite) PlaceOrder(ctx context.Context, symbol string, side string, quantity int, price float64) (string, error) {
	req := contracts.OrderRequest{Symbol: symbol, Side: side, Quantity: quantity, Price: price}
	orderType := contracts.OrderTypeMarket
	if price > 0 {
		orderType = contracts.OrderTypeLimit
	}
	return k.placeOrder(ctx, req, orderType, "")
}

// PlaceOrderTagged submits an order with Kite's short client tag for later broker-side
// reconciliation.
func (k *Kite) PlaceOrderTagged(ctx context.Context, req contracts.OrderRequest, tag string) (string, error) {
	typ := strings.ToUpper(strings.TrimSpace(req.Type))
	if typ == "" {
		typ = contracts.OrderTypeMarket
		if req.Price > 0 {
			typ = contracts.OrderTypeLimit
		}
	}
	return k.placeOrder(ctx, req, typ, tag)
}

// placeOrder is the shared implementation behind PlaceOrder and orderClient.PlaceOrder:
// it validates the request, resolves the instrument token, and posts the regular-order
// form Kite expects.
func (k *Kite) placeOrder(ctx context.Context, req contracts.OrderRequest, orderType, tag string) (string, error) {
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
	if len(tag) > 20 || strings.TrimSpace(tag) != tag || strings.ContainsAny(tag, " \t\r\n") {
		return "", fmt.Errorf("kite order %s: invalid tag %q: %w", symbol, tag, ErrInvalidOrder)
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
		"tradingsymbol":    {inst.Symbol},
		"exchange":         {inst.Exchange},
		"transaction_type": {direction},
		"product":          {product},
		"order_type":       {orderType},
		"quantity":         {strconv.Itoa(req.Quantity)},
		"variety":          {"regular"},
	}
	if tag != "" {
		form.Set("tag", tag)
	}
	if orderType == contracts.OrderTypeLimit {
		form.Set("price", strconv.FormatFloat(req.Price, 'f', -1, 64))
	}

	body, err := k.do(ctx, httpPost, k.endpoints.ordersURL(), form, 1<<20)
	if err != nil {
		return "", fmt.Errorf("kite order %s: %w", symbol, err)
	}

	var resp struct {
		Status string `json:"status"`
		Data   struct {
			OrderID string `json:"order_id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", fmt.Errorf("kite order %s: decode response: %w: %w", symbol, ErrUnavailable, err)
	}
	if resp.Status != "success" || resp.Data.OrderID == "" {
		return "", fmt.Errorf("kite order %s: response carried no order_id: %w", symbol, ErrUnavailable)
	}
	return resp.Data.OrderID, nil
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

// kitePosition is one row of the Kite /positions response.
type kitePosition struct {
	TradingSymbol   string  `json:"tradingsymbol"`
	InstrumentToken int64   `json:"instrument_token"`
	Exchange        string  `json:"exchange"`
	Quantity        float64 `json:"quantity"`
	AveragePrice    float64 `json:"average_price"`
	Product         string  `json:"product"`
}

// OrderState is the normalized state of an order reported by Kite.
type OrderState string

const (
	OrderOpen OrderState = "OPEN"

	OrderPartial OrderState = "PARTIAL"

	OrderFilled OrderState = "FILLED"

	OrderCancelled OrderState = "CANCELLED"

	OrderRejected OrderState = "REJECTED"

	OrderUnknown OrderState = "UNKNOWN"
)

// Order is the broker's latest view of one submitted order.
type Order struct {
	ID               string
	Tag              string
	Status           OrderState
	Symbol           string
	Side             string
	Quantity         int
	Price            float64
	Type             string
	FilledQuantity   int
	AverageFillPrice float64
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

// OrderReconciler queries orders for callers that must resolve uncertain submission
// outcomes without changing the shared Client contract.
type OrderReconciler interface {
	GetOrder(ctx context.Context, orderID string) (Order, error)
	GetOrders(ctx context.Context) ([]Order, error)
}

// QuoteClient returns live marks for risk and portfolio calculations.
type QuoteClient interface {
	GetLastPrices(ctx context.Context, symbols []string) (map[string]float64, error)
}

// GetPositions returns the current net positions.
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

// decodePositions reads net holdings from Kite's portfolio envelope.
func decodePositions(body []byte) ([]kitePosition, error) {
	var wrapped struct {
		Status string `json:"status"`
		Data   struct {
			Net []kitePosition `json:"net"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &wrapped); err != nil {
		return nil, fmt.Errorf("kite positions: decode response: %w: %w", ErrUnavailable, err)
	}
	if wrapped.Status != "success" {
		return nil, fmt.Errorf("kite positions: response status %q: %w", wrapped.Status, ErrUnavailable)
	}
	return wrapped.Data.Net, nil
}

type kiteOrder struct {
	ID             string  `json:"order_id"`
	Tag            string  `json:"tag"`
	Status         string  `json:"status"`
	Symbol         string  `json:"tradingsymbol"`
	Side           string  `json:"transaction_type"`
	Quantity       int     `json:"quantity"`
	Price          float64 `json:"price"`
	Type           string  `json:"order_type"`
	FilledQuantity int     `json:"filled_quantity"`
	AveragePrice   float64 `json:"average_price"`
	OrderTime      string  `json:"order_timestamp"`
	UpdateTime     string  `json:"exchange_update_timestamp"`
}

func (k *Kite) GetOrders(ctx context.Context) ([]Order, error) {
	body, err := k.do(ctx, httpGet, k.endpoints.ordersListURL(), nil, 4<<20)
	if err != nil {
		return nil, fmt.Errorf("kite orders: %w", err)
	}
	rows, err := decodeOrderRows(body)
	if err != nil {
		return nil, err
	}
	out := make([]Order, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.contract())
	}
	return out, nil
}

func (k *Kite) GetOrder(ctx context.Context, orderID string) (Order, error) {
	id := strings.TrimSpace(orderID)
	if id == "" {
		return Order{}, fmt.Errorf("kite order history: empty order id: %w", ErrOrderNotFound)
	}
	body, err := k.do(ctx, httpGet, k.endpoints.orderHistoryURL(id), nil, 2<<20)
	if err != nil {
		return Order{}, fmt.Errorf("kite order history %s: %w", id, err)
	}
	rows, err := decodeOrderRows(body)
	if err != nil {
		return Order{}, err
	}
	if len(rows) == 0 {
		return Order{}, fmt.Errorf("kite order history %s: empty response: %w", id, ErrOrderNotFound)
	}
	return rows[len(rows)-1].contract(), nil
}

func decodeOrderRows(body []byte) ([]kiteOrder, error) {
	var response struct {
		Status string      `json:"status"`
		Data   []kiteOrder `json:"data"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("kite orders: decode response: %w: %w", ErrUnavailable, err)
	}
	if response.Status != "success" {
		return nil, fmt.Errorf("kite orders: response status %q: %w", response.Status, ErrUnavailable)
	}
	return response.Data, nil
}

func (row kiteOrder) contract() Order {
	return Order{
		ID: row.ID, Tag: row.Tag, Status: normalizeOrderState(row.Status),
		Symbol:   strings.ToUpper(strings.TrimSpace(row.Symbol)),
		Side:     strings.ToUpper(strings.TrimSpace(row.Side)),
		Quantity: row.Quantity, Price: row.Price, Type: strings.ToUpper(row.Type),
		FilledQuantity: row.FilledQuantity, AverageFillPrice: row.AveragePrice,
		CreatedAt: parseKiteTime(row.OrderTime), UpdatedAt: parseKiteTime(row.UpdateTime),
	}
}

func normalizeOrderState(status string) OrderState {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case "OPEN", "PENDING", "TRIGGER PENDING", "AMO REQ RECEIVED", "VALIDATION PENDING", "OPEN PENDING":
		return OrderOpen
	case "PARTIAL", "PARTIALLY FILLED":
		return OrderPartial
	case "COMPLETE", "COMPLETED", "FILLED":
		return OrderFilled
	case "CANCELLED", "CANCELED":
		return OrderCancelled
	case "REJECTED":
		return OrderRejected
	default:
		return OrderUnknown
	}
}

func parseKiteTime(value string) time.Time {
	value = strings.TrimSpace(value)
	if value == "" || value == "null" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02 15:04:05"} {
		if layout == time.RFC3339 {
			if stamp, err := time.Parse(layout, value); err == nil {
				return stamp.UTC()
			}
			continue
		}
		if stamp, err := time.ParseInLocation(layout, value, time.FixedZone("IST", 5*60*60+30*60)); err == nil {
			return stamp.UTC()
		}
	}
	return time.Time{}
}

func (k *Kite) GetLastPrices(ctx context.Context, symbols []string) (map[string]float64, error) {
	query := url.Values{}
	for _, symbol := range symbols {
		inst, err := k.Instrument(ctx, symbol)
		if err != nil {
			return nil, err
		}
		query.Add("i", inst.Exchange+":"+inst.Symbol)
	}
	if len(query) == 0 {
		return map[string]float64{}, nil
	}
	body, err := k.do(ctx, httpGet, k.endpoints.quoteURL()+"?"+query.Encode(), nil, 2<<20)
	if err != nil {
		return nil, fmt.Errorf("kite quotes: %w", err)
	}
	out, err := decodeLastPrices(body)
	if err != nil {
		return nil, err
	}
	for _, symbol := range symbols {
		if _, ok := out[strings.ToUpper(strings.TrimSpace(symbol))]; !ok {
			return nil, fmt.Errorf("kite quotes: missing mark for %q: %w", symbol, ErrUnavailable)
		}
	}
	return out, nil
}

func decodeLastPrices(body []byte) (map[string]float64, error) {
	var response struct {
		Status string `json:"status"`
		Data   map[string]struct {
			LastPrice float64 `json:"last_price"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("kite quotes: decode response: %w: %w", ErrUnavailable, err)
	}
	if response.Status != "success" {
		return nil, fmt.Errorf("kite quotes: response status %q: %w", response.Status, ErrUnavailable)
	}
	out := make(map[string]float64, len(response.Data))
	for key, row := range response.Data {
		_, symbol, ok := strings.Cut(key, ":")
		if !ok || row.LastPrice <= 0 {
			return nil, fmt.Errorf("kite quotes: invalid mark for %q: %w", key, ErrUnavailable)
		}
		out[strings.ToUpper(symbol)] = row.LastPrice
	}
	return out, nil
}
