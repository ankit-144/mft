package jobs

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"go.uber.org/zap"
)

// Kite history endpoint and limits.
const (
	// kiteAPIURL is Kite Connect's production base. It is a variable rather
	// than a constant so a test can point the client at an httptest server.
	kiteAPIURL = "https://api.kite.trade"

	// historyPath is the Kite Connect historical data route. The exchange and
	// trading symbol are path segments; the interval is a bare word.
	historyPath = "/data/historical/%s/%s/%s/%s/%s"

	// historyResponseMaxBytes bounds a candle response. Kite caps a single
	// response at a few thousand rows; the limit stops a corrupted or hostile
	// response from exhausting memory.
	historyResponseMaxBytes = 16 << 20

	// defaultHistoryTimeout is used when broker.request_timeout_seconds is
	// zero or negative.
	defaultHistoryTimeout = 10 * time.Second
)

// candle fields inside a Kite historical row. Kite returns a positional array
// rather than an object, in this fixed order.
const (
	histFieldTimestamp = 0
	histFieldOpen      = 1
	histFieldHigh      = 2
	histFieldLow       = 3
	histFieldClose     = 4
	histFieldVolume    = 5
	histFieldsRequired = 6
)

// historyEnvelope is the Kite historical response. On success Status is
// "success" and Data.Candles holds the rows; on failure Status is "error" and
// Errors carries the same envelope core/broker decodes, which is left to the
// shared error parser.
type historyEnvelope struct {
	Status string `json:"status"`
	Data   struct {
		Candles []json.RawMessage `json:"candles"`
	} `json:"data"`
	Errors []struct {
		ErrorCode string `json:"error_code"`
		Message   string `json:"message"`
	} `json:"errors"`
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// KiteHistory is the Kite-backed implementation of HistoricalClient.
//
// Kite's historical data is a REST call, and C1 scoped core/broker to the
// WebSocket stream and the order endpoints, so the client lives here rather
// than behind core/broker: this component owns services/jobs, and adding a
// fourth file to another component's package would collide with its branch.
//
// It holds credentials rather than a *broker.Kite, because Kite's credential
// fields are unexported by design and reading them out of the connector would
// be a worse coupling than reading the same frozen config block. Symbol
// resolution is delegated to the [InstrumentResolver] seam, so instrument
// tokens still come from C1's instrument master and there is one place that
// knows what a tradable symbol is.
type KiteHistory struct {
	apiKey      string
	accessToken string
	baseURL     string
	client      *http.Client
	resolver    InstrumentResolver
	log         *zap.Logger
}

// NewKiteHistory returns a historical client for the configured Kite account.
// The resolver is C1's *broker.Kite, which supplies exchange and instrument
// token; a nil resolver is replaced by one that resolves nothing, in which case
// every call fails with broker.ErrInstrumentNotFound.
func NewKiteHistory(cfg *config.Config, resolver InstrumentResolver, log *zap.Logger) *KiteHistory {
	if log == nil {
		log = zap.NewNop()
	}
	timeout := time.Duration(cfg.Broker.RequestTimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = defaultHistoryTimeout
	}
	return &KiteHistory{
		apiKey:      cfg.Broker.APIKey,
		accessToken: cfg.Broker.AccessToken,
		baseURL:     kiteAPIURL,
		client:      &http.Client{Timeout: timeout},
		resolver:    resolver,
		log:         log,
	}
}

// newKiteHistory builds a client pointed at baseURL. The endpoint is a
// parameter rather than a package variable so that a test can reach an
// httptest server without mutating shared state, and so that two such clients
// cannot interfere with each other.
func newKiteHistory(cfg *config.Config, resolver InstrumentResolver, baseURL string, log *zap.Logger) *KiteHistory {
	k := NewKiteHistory(cfg, resolver, log)
	k.baseURL = strings.TrimRight(baseURL, "/")
	return k
}

// base returns the configured base URL, falling back to production.
func (k *KiteHistory) base() string {
	if k.baseURL == "" {
		return kiteAPIURL
	}
	return k.baseURL
}

// log returns the client logger, never nil.
func (k *KiteHistory) logger() *zap.Logger {
	if k.log == nil {
		return zap.NewNop()
	}
	return k.log
}

// HistoricalCandles implements HistoricalClient against
// GET /data/historical/{exchange}/{symbol}/{interval}/{from}/{to}.
//
// The interval must be a bare Kite interval word ("minute", "day"); the
// one-minute interval this platform trades at is "minute". The range is
// [from, to], matching the segment boundaries the store writes.
func (k *KiteHistory) HistoricalCandles(ctx context.Context, symbol string, from, to time.Time, interval string) ([]contracts.Candle, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(k.accessToken) == "" {
		return nil, fmt.Errorf("kite history: access_token is not configured: %w", broker.ErrAuth)
	}
	if !to.After(from) {
		return nil, fmt.Errorf("kite history: %s: empty range %s..%s: %w",
			symbol, from.Format(time.DateOnly), to.Format(time.DateOnly), broker.ErrUnavailable)
	}
	interval = strings.TrimSpace(interval)
	if interval == "" {
		return nil, fmt.Errorf("kite history: %s: empty interval: %w", symbol, broker.ErrUnavailable)
	}

	inst, err := k.instrument(ctx, symbol)
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf(historyPath,
		url.PathEscape(inst.Exchange),
		url.PathEscape(inst.Symbol),
		url.PathEscape(interval),
		from.UTC().Format(time.DateOnly),
		to.UTC().Format(time.DateOnly),
	)
	query := url.Values{}
	query.Set("instrument_token", strconv.FormatInt(inst.Token, 10))
	query.Set("continuous", "0")
	query.Set("include_oi", "0")

	raw, err := k.get(ctx, endpoint+"?"+query.Encode())
	if err != nil {
		return nil, fmt.Errorf("kite history: %s: %w", symbol, err)
	}
	return k.decode(inst.Symbol, raw)
}

// instrument resolves a symbol to its exchange and token via the resolver.
func (k *KiteHistory) instrument(ctx context.Context, symbol string) (contracts.Instrument, error) {
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	if k.resolver == nil {
		return contracts.Instrument{}, fmt.Errorf("kite history: %s: no instrument resolver: %w", sym, broker.ErrInstrumentNotFound)
	}
	all, err := k.resolver.Instruments(ctx)
	if err != nil {
		return contracts.Instrument{}, fmt.Errorf("kite history: resolve %s: %w", sym, err)
	}
	for _, inst := range all {
		if strings.ToUpper(strings.TrimSpace(inst.Symbol)) == sym {
			return inst, nil
		}
	}
	return contracts.Instrument{}, fmt.Errorf("kite history: %s: %w", sym, broker.ErrInstrumentNotFound)
}

// get performs the authenticated request and maps a non-2xx status onto the
// same sentinels core/broker uses, so the fetcher can tell a 429 to back off
// from a 401 to stop.
func (k *KiteHistory) get(ctx context.Context, endpoint string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, k.base()+endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: build %s: %w", broker.ErrUnavailable, endpoint, err)
	}
	req.Header.Set("Authorization", "token "+k.apiKey+":"+k.accessToken)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "mft-jobs/1.0")

	resp, err := k.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %w", broker.ErrUnavailable, endpoint, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, historyResponseMaxBytes))
	if err != nil {
		return nil, fmt.Errorf("%w: read %s: %w", broker.ErrUnavailable, endpoint, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, historyStatusError(resp.StatusCode, raw)
	}
	return raw, nil
}

// historyStatusError maps a Kite non-2xx onto a sentinel. The mapping mirrors
// core/broker's classifyStatus so the two clients agree on what a 429 is; a
// JSON body is preferred when present because Kite's status line alone loses
// the "too many requests" wording.
func historyStatusError(status int, body []byte) error {
	var env historyEnvelope
	if err := json.Unmarshal(body, &env); err == nil {
		code, message := env.firstError()
		switch {
		case status == http.StatusTooManyRequests,
			strings.EqualFold(code, "429"),
			strings.Contains(strings.ToLower(message), "too many request"):
			return fmt.Errorf("kite history: http %d: %s: %w", status, message, broker.ErrRateLimit)
		case status == http.StatusUnauthorized, status == http.StatusForbidden:
			return fmt.Errorf("kite history: http %d: %s: %w", status, message, broker.ErrAuth)
		case status == http.StatusNotFound:
			// A 404 on this route means Kite does not know the exchange or
			// symbol, not that an order is missing. ErrOrderNotFound would be
			// a misleading sentinel here, and ErrInstrumentNotFound is what
			// lets the job give up on the symbol instead of retrying.
			return fmt.Errorf("kite history: http %d: %s: %w", status, message, broker.ErrInstrumentNotFound)
		case status >= 500:
			return fmt.Errorf("kite history: http %d: %s: %w", status, message, broker.ErrUnavailable)
		}
	}
	switch {
	case status == http.StatusTooManyRequests:
		return fmt.Errorf("kite history: http %d: %w", status, broker.ErrRateLimit)
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return fmt.Errorf("kite history: http %d: %w", status, broker.ErrAuth)
	case status == http.StatusNotFound:
		return fmt.Errorf("kite history: http %d: %w", status, broker.ErrInstrumentNotFound)
	case status >= 500:
		return fmt.Errorf("kite history: http %d: %w", status, broker.ErrUnavailable)
	default:
		return fmt.Errorf("kite history: http %d: %w", status, broker.ErrInvalidOrder)
	}
}

// firstError picks the most specific error Kite reported, preferring the
// multi-error envelope over the legacy single-error shape.
func (e *historyEnvelope) firstError() (code, message string) {
	if len(e.Errors) > 0 {
		return e.Errors[0].ErrorCode, e.Errors[0].Message
	}
	return e.Error.Code, e.Error.Message
}

// decode turns a Kite historical body into candles.
//
// A "success" envelope is authoritative. A body that says "error" but arrived
// with a 2xx status is mapped to a sentinel too, because Kite does that when
// a token expires mid-session.
func (k *KiteHistory) decode(symbol string, raw []byte) ([]contracts.Candle, error) {
	var env historyEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("kite history: %s: decode response: %w: %w", symbol, broker.ErrUnavailable, err)
	}
	if !strings.EqualFold(env.Status, "success") {
		code, message := env.firstError()
		return nil, historyStatusError(statusFromError(code, message), raw)
	}

	candles := make([]contracts.Candle, 0, len(env.Data.Candles))
	for i, row := range env.Data.Candles {
		c, err := decodeHistoryRow(symbol, row)
		if err != nil {
			// One malformed row must not discard a whole segment: Kite has
			// shipped rows with a missing volume field before, and the OHLC of
			// the surrounding bars is still worth backfilling.
			k.logger().Warn("skipping malformed historical row",
				zap.String("symbol", symbol),
				zap.Int("row", i),
				zap.Error(err),
			)
			continue
		}
		candles = append(candles, c)
	}
	return candles, nil
}

// statusFromError guesses the HTTP status of a body that arrived as a 2xx but
// carries an error envelope, so the sentinel mapping has something to switch
// on.
func statusFromError(code, message string) int {
	lower := strings.ToLower(code + " " + message)
	switch {
	case strings.Contains(lower, "too many request"), code == "429":
		return http.StatusTooManyRequests
	case strings.Contains(lower, "invalid_token"),
		strings.Contains(lower, "invalid_api_key"),
		strings.Contains(lower, "session expired"),
		strings.Contains(lower, "incorrect api_key"),
		strings.Contains(lower, "access token"):
		return http.StatusUnauthorized
	case strings.Contains(lower, "instrument"), strings.Contains(lower, "unknown symbol"):
		return http.StatusNotFound
	default:
		return http.StatusBadRequest
	}
}

// decodeHistoryRow parses one Kite candle array:
//
//	["2024-01-02T09:15:00+05:30", open, high, low, close, volume]
//
// Kite quotes these as JSON strings, not numbers, so every field is decoded
// leniently rather than assuming a numeric JSON type.
func decodeHistoryRow(symbol string, row json.RawMessage) (contracts.Candle, error) {
	var fields []json.RawMessage
	if err := json.Unmarshal(row, &fields); err != nil {
		return contracts.Candle{}, fmt.Errorf("row is not an array: %w", err)
	}
	if len(fields) < histFieldsRequired {
		return contracts.Candle{}, fmt.Errorf("row has %d fields, want at least %d", len(fields), histFieldsRequired)
	}

	var candle contracts.Candle
	candle.Symbol = symbol

	var stamp string
	if err := json.Unmarshal(fields[histFieldTimestamp], &stamp); err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: decode timestamp: %w", histFieldTimestamp, err)
	}
	ts, err := time.Parse(time.RFC3339, stamp)
	if err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: parse timestamp %q: %w", histFieldTimestamp, stamp, err)
	}
	candle.Timestamp = ts.UTC().Truncate(time.Minute)

	if candle.Open, err = histFloat(fields[histFieldOpen]); err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: open: %w", histFieldOpen, err)
	}
	if candle.High, err = histFloat(fields[histFieldHigh]); err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: high: %w", histFieldHigh, err)
	}
	if candle.Low, err = histFloat(fields[histFieldLow]); err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: low: %w", histFieldLow, err)
	}
	if candle.Close, err = histFloat(fields[histFieldClose]); err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: close: %w", histFieldClose, err)
	}
	if candle.Volume, err = histInt(fields[histFieldVolume]); err != nil {
		return contracts.Candle{}, fmt.Errorf("field %d: volume: %w", histFieldVolume, err)
	}
	return candle, nil
}

// histFloat decodes a Kite numeric field, which is either a bare JSON number
// or a quoted string.
func histFloat(field json.RawMessage) (float64, error) {
	raw := strings.Trim(strings.TrimSpace(string(field)), `"`)
	if raw == "" || raw == "null" {
		return 0, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("%q is not a finite number", raw)
	}
	return v, nil
}

// histInt decodes a Kite integer field, which may arrive as "12345.0".
func histInt(field json.RawMessage) (int64, error) {
	raw := strings.Trim(strings.TrimSpace(string(field)), `"`)
	if raw == "" || raw == "null" {
		return 0, nil
	}
	if v, err := strconv.ParseInt(raw, 10, 64); err == nil {
		return v, nil
	}
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, fmt.Errorf("%q is not an integer", raw)
	}
	return int64(f), nil
}

// Compile-time proof that the Kite client satisfies the historical seam.
var _ HistoricalClient = (*KiteHistory)(nil)
