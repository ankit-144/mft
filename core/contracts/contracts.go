// Package contracts defines shared market, signal, order and portfolio boundaries.
package contracts

import "time"

// Side values for an order or signal.
const (
	SideBuy  = "BUY"
	SideSell = "SELL"
)

// Order types.
const (
	OrderTypeMarket = "MARKET"
	OrderTypeLimit  = "LIMIT"
)

// Execution order states exposed by the execution API.
const (
	OrderStatusSubmitting  = "SUBMITTING"
	OrderStatusPaperFilled = "PAPER_FILLED"
	OrderStatusUnknown     = "UNKNOWN"
	OrderStatusOpen        = "OPEN"
	OrderStatusPartial     = "PARTIAL"
	OrderStatusFilled      = "FILLED"
	OrderStatusCancelled   = "CANCELLED"
	OrderStatusRejected    = "REJECTED"
)

// Tick is a raw market data tick from the broker.
type Tick struct {
	Symbol    string    `json:"symbol"`
	Token     int64     `json:"token"`
	Price     float64   `json:"price"`
	Volume    int64     `json:"volume"`
	Timestamp time.Time `json:"timestamp"`
}

// Candle is an aggregated one-minute OHLCV bar.
type Candle struct {
	Symbol    string    `json:"symbol"`
	Timestamp time.Time `json:"timestamp"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    int64     `json:"volume"`
}

// Signal is a model conviction for one symbol at one minute close.
type Signal struct {
	Symbol         string    `json:"symbol"`
	Side           string    `json:"side"`
	Quantity       int       `json:"quantity"`
	Price          float64   `json:"price"`
	Score          float64   `json:"score"`
	Model          string    `json:"model"`
	AsOf           time.Time `json:"as_of"`
	IdempotencyKey string    `json:"idempotency_key"`
}

// OrderRequest describes an order to place at the broker.
type OrderRequest struct {
	Symbol         string  `json:"symbol"`
	Side           string  `json:"side"`
	Quantity       int     `json:"quantity"`
	Price          float64 `json:"price"`
	Type           string  `json:"type"`
	IdempotencyKey string  `json:"idempotency_key,omitempty"`
}

// OrderResult is the latest known broker state for an execution request.
type OrderResult struct {
	OrderID        string  `json:"order_id,omitempty"`
	Status         string  `json:"status"`
	FilledQuantity int     `json:"filled_quantity"`
	AveragePrice   float64 `json:"average_price,omitempty"`
}

// Position is an open holding in one symbol.
type Position struct {
	Symbol   string  `json:"symbol"`
	Quantity int     `json:"quantity"`
	AvgPrice float64 `json:"avg_price"`
}

// Instrument describes a tradable symbol.
type Instrument struct {
	Token    int64  `json:"token"`
	Symbol   string `json:"symbol"`
	Exchange string `json:"exchange"`
	LotSize  int    `json:"lot_size"`
}

// Portfolio is the risk engine's view of current exposure.
type Portfolio struct {
	Cash              float64            `json:"cash"`
	Equity            float64            `json:"equity"`
	RealisedPnL       float64            `json:"realised_pnl"`
	RealisedPnLToday  float64            `json:"realised_pnl_today"`
	PeakEquity        float64            `json:"peak_equity"`
	OpenPositions     map[string]int     `json:"open_positions"`
	PositionValues    map[string]float64 `json:"position_values"`
	ReservedCash      float64            `json:"reserved_cash"`
	ReservedPositions map[string]int     `json:"reserved_positions,omitempty"`
	ReservedSells     map[string]int     `json:"reserved_sells,omitempty"`
	ReservedValues    map[string]float64 `json:"reserved_values,omitempty"`
}

// Risk rejection reason codes.
const (
	ReasonMaxPosition  = "RISK_MAX_POSITION"
	ReasonMaxPositions = "RISK_MAX_POSITIONS"
	ReasonMaxDrawdown  = "RISK_MAX_DRAWDOWN"
	ReasonDailyLoss    = "RISK_DAILY_LOSS"
	ReasonDebounced    = "RISK_DEBOUNCED"
	ReasonBadQuantity  = "RISK_BAD_QUANTITY"
	ReasonMarketClosed = "RISK_MARKET_CLOSED"
	ReasonDuplicate    = "RISK_DUPLICATE"
	ReasonStaleSignal  = "RISK_STALE_SIGNAL"
)

// Rejection is a typed risk-gate failure.
type Rejection struct {
	Code    string
	Message string
}

// Error implements the error interface.
func (r *Rejection) Error() string {
	if r.Message == "" {
		return r.Code
	}
	return r.Code + ": " + r.Message
}

// Reject builds a Rejection.
func Reject(code, message string) *Rejection {
	return &Rejection{Code: code, Message: message}
}
