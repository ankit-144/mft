// Package contracts holds the frozen domain types and cross-component
// interfaces shared by every MFT service.
//
// This package is owned by the foundation layer. Components import from it;
// they must not redefine these types. Changing anything here is a foundation
// change against master, not a component change.
//
// See docs/contracts.md for the full specification and rationale.
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

// Tick is a raw market data tick from the broker.
type Tick struct {
	Symbol    string    `json:"symbol"`
	Token     int64     `json:"token"`
	Price     float64   `json:"price"`
	Volume    int64     `json:"volume"`
	Timestamp time.Time `json:"timestamp"`
}

// Candle is an aggregated one-minute OHLCV bar. Timestamp is truncated to the
// minute, in UTC.
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
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"`
	Quantity int     `json:"quantity"`
	Price    float64 `json:"price"`
	Type     string  `json:"type"`
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
	Cash          float64        `json:"cash"`
	RealisedPnL   float64        `json:"realised_pnl"`
	PeakEquity    float64        `json:"peak_equity"`
	OpenPositions map[string]int `json:"open_positions"`
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
)

// Rejection is a typed risk-gate failure. Code is one of the Reason*
// constants and is safe to switch on by callers and metrics.
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
