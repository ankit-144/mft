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
	kiteAPIURL = "https://api.kite.trade"

	historyPath = "/instruments/historical/%d/%s"

	historyResponseMaxBytes = 16 << 20

	defaultHistoryTimeout = 10 * time.Second
)

// candle fields inside a Kite historical row.
const (
	histFieldTimestamp = 0
	histFieldOpen      = 1
	histFieldHigh      = 2
	histFieldLow       = 3
	histFieldClose     = 4
	histFieldVolume    = 5
	histFieldsRequired = 6
)

// historyEnvelope is the Kite historical response.
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
type KiteHistory struct {
	apiKey      string
	accessToken string
	baseURL     string
	client      *http.Client
	resolver    InstrumentResolver
	log         *zap.Logger
	pacer       *broker.RESTPacer
}

// NewKiteHistory returns a historical client for the configured Kite account.
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
		pacer:       broker.NewRESTPacer(),
	}
}

// newKiteHistory builds a client pointed at baseURL.
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

// HistoricalCandles implements HistoricalClient against Kite's token route.
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

	requestPath := historyRequestPath(inst, from, to, interval)
	raw, err := k.get(ctx, requestPath)
	if err != nil {
		return nil, fmt.Errorf("kite history: %s: %w", symbol, err)
	}
	candles, err := k.decode(inst.Symbol, raw)
	if err != nil {
		return nil, err
	}
	return historicalRange(candles, from, to), nil
}

func historyRequestPath(inst contracts.Instrument, from, to time.Time, interval string) string {
	endpoint := fmt.Sprintf(historyPath, inst.Token, url.PathEscape(interval))
	query := url.Values{}
	query.Set("from", from.In(indiaTimeZone).Format("2006-01-02 15:04:05"))
	query.Set("to", to.In(indiaTimeZone).Format("2006-01-02 15:04:05"))
	query.Set("continuous", "0")
	query.Set("oi", "0")
	return endpoint + "?" + query.Encode()
}

func historicalRange(candles []contracts.Candle, from, to time.Time) []contracts.Candle {
	out := candles[:0]
	for _, candle := range candles {
		if !candle.Timestamp.Before(from) && candle.Timestamp.Before(to) {
			out = append(out, candle)
		}
	}
	return out
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

// get performs the authenticated request and maps a non-2xx status onto the same
// sentinels core/broker uses, so the fetcher can tell a 429 to back off from a 401 to
// stop.
func (k *KiteHistory) get(ctx context.Context, endpoint string) ([]byte, error) {
	if err := k.pacer.Wait(ctx, broker.RESTHistorical); err != nil {
		return nil, err
	}
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

// historyStatusError maps a Kite non-2xx onto a sentinel.
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

// firstError picks the most specific error Kite reported, preferring the multi-error
// envelope over the legacy single-error shape.
func (e *historyEnvelope) firstError() (code, message string) {
	if len(e.Errors) > 0 {
		return e.Errors[0].ErrorCode, e.Errors[0].Message
	}
	return e.Error.Code, e.Error.Message
}

// decode turns a Kite historical body into candles.
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

// statusFromError guesses the HTTP status of a body that arrived as a 2xx but carries
// an error envelope, so the sentinel mapping has something to switch on.
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
	ts, err := parseKiteHistoryTime(stamp)
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

func parseKiteHistoryTime(value string) (time.Time, error) {
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05-0700"} {
		if stamp, err := time.Parse(layout, value); err == nil {
			return stamp.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unsupported Kite timestamp %q", value)
}

// histFloat decodes a Kite numeric field, which is either a bare JSON number or a
// quoted string.
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
