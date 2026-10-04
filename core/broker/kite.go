package broker

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"go.uber.org/zap"
)

// Production Kite Connect endpoints.
const (
	defaultHTTPBaseURL = "https://api.kite.trade"
	defaultWSBaseURL   = "wss://ws.kite.trade"
)

// Tunables that have no key in the shared config schema.
const (
	defaultRequestTimeout = 10 * time.Second
	defaultPongTimeout    = 90 * time.Second
	defaultWriteWait      = 5 * time.Second
	defaultReconnectBase  = time.Second
	defaultMaxBackoff     = 60 * time.Second
	defaultInstrumentTTL  = time.Hour
	defaultProduct        = "NRML"
)

// instrumentDumpMaxBytes bounds the CSV instrument download.
const instrumentDumpMaxBytes = 64 << 20

// REST verbs, aliased so the call sites read as prose.
const (
	httpGet    = http.MethodGet
	httpPost   = http.MethodPost
	httpDelete = http.MethodDelete
)

// validProducts is the set of Kite product codes the connector accepts.
var validProducts = map[string]bool{"MIS": true, "CNC": true, "NRML": true}

// endpoints holds the Kite Connect URLs.
type endpoints struct {
	httpBase string
	wsBase   string
}

func (e endpoints) instrumentsURL() string { return e.httpBase + "/instruments" }
func (e endpoints) ordersURL() string      { return e.httpBase + "/orders/regular" }
func (e endpoints) orderURL(id string) string {
	return e.httpBase + "/orders/regular/" + url.PathEscape(id)
}
func (e endpoints) positionsURL() string  { return e.httpBase + "/portfolio/positions" }
func (e endpoints) ordersListURL() string { return e.httpBase + "/orders" }
func (e endpoints) orderHistoryURL(id string) string {
	return e.httpBase + "/orders/" + url.PathEscape(id)
}
func (e endpoints) quoteURL() string { return e.httpBase + "/quote/ltp" }
func (e endpoints) streamURL(apiKey, token string) string {
	query := url.Values{"api_key": {apiKey}, "access_token": {token}}
	return e.wsBase + "/?" + query.Encode()
}

// Kite is the Zerodha Kite Connect connector.
type Kite struct {
	apiKey      string
	apiSecret   string
	accessToken string
	watchlist   []string
	product     string

	requestTimeout time.Duration
	maxBackoff     time.Duration
	reconnectBase  time.Duration
	pongTimeout    time.Duration
	writeWait      time.Duration
	instrumentTTL  time.Duration

	endpoints endpoints
	client    *http.Client
	dialer    *websocket.Dialer
	pacer     *RESTPacer

	mu         sync.Mutex
	instMu     sync.Mutex
	instrument []contracts.Instrument
	loadedAt   time.Time

	seq        atomic.Uint64
	reconnects atomic.Int64
	delivered  atomic.Int64

	loggerPtr atomic.Pointer[zap.Logger]
}

// NewKite returns a Kite connector with no credentials and production endpoints.
func NewKite() *Kite {
	return newKite(&config.BrokerConfig{})
}

// NewKiteFromConfig returns a Kite connector configured from the shared broker config
// block.
func NewKiteFromConfig(cfg config.BrokerConfig) (*Kite, error) {
	if cfg.RequestTimeoutSeconds < 0 {
		return nil, fmt.Errorf("broker: request_timeout_seconds: %d: %w", cfg.RequestTimeoutSeconds, ErrInvalidOrder)
	}
	if cfg.ReconnectMaxBackoffSecs < 0 {
		return nil, fmt.Errorf("broker: reconnect_max_backoff_seconds: %d: %w", cfg.ReconnectMaxBackoffSecs, ErrInvalidOrder)
	}
	return newKite(&cfg), nil
}

// newKite builds a connector from cfg, applying the defaults for every value the config
// schema leaves at zero.
func newKite(cfg *config.BrokerConfig) *Kite {
	k := &Kite{
		apiKey:         cfg.APIKey,
		apiSecret:      cfg.APISecret,
		accessToken:    cfg.AccessToken,
		watchlist:      append([]string(nil), cfg.Instruments...),
		product:        defaultProduct,
		requestTimeout: time.Duration(cfg.RequestTimeoutSeconds) * time.Second,
		maxBackoff:     time.Duration(cfg.ReconnectMaxBackoffSecs) * time.Second,
		endpoints:      endpoints{httpBase: defaultHTTPBaseURL, wsBase: defaultWSBaseURL},
		pacer:          NewRESTPacer(),
	}
	k.loggerPtr.Store(zap.NewNop())
	if k.requestTimeout <= 0 {
		k.requestTimeout = defaultRequestTimeout
	}
	if k.maxBackoff <= 0 {
		k.maxBackoff = defaultMaxBackoff
	}
	k.reconnectBase = defaultReconnectBase
	k.pongTimeout = defaultPongTimeout
	k.writeWait = defaultWriteWait
	k.instrumentTTL = defaultInstrumentTTL
	k.client = &http.Client{Timeout: k.requestTimeout}
	k.dialer = &websocket.Dialer{HandshakeTimeout: k.requestTimeout, Proxy: http.ProxyFromEnvironment}
	return k
}

// log returns the logger the connector was built with.
func (k *Kite) log() *zap.Logger { return k.loggerPtr.Load() }

// SetLogger attaches a logger for reconnect and frame diagnostics.
func (k *Kite) SetLogger(l *zap.Logger) {
	if l == nil {
		l = zap.NewNop()
	}
	k.loggerPtr.Store(l)
}

// SetProduct selects the Kite product code (MIS, CNC or NRML) used for new orders.
func (k *Kite) SetProduct(product string) error {
	p := strings.ToUpper(strings.TrimSpace(product))
	if !validProducts[p] {
		return fmt.Errorf("kite product %q: %w", product, ErrInvalidOrder)
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	k.product = p
	return nil
}

// Reconnects returns the number of reconnect attempts Stream has scheduled after the
// first connection.
func (k *Kite) Reconnects() int64 { return k.reconnects.Load() }

// Delivered returns the number of ticks handed to the consumer channel.
func (k *Kite) Delivered() int64 { return k.delivered.Load() }

// Sequence returns the sequence number stamped on the most recently delivered tick.
func (k *Kite) Sequence() uint64 { return k.seq.Load() }

// authHeader returns the Kite authorization header value.
func (k *Kite) authHeader() string {
	return "token " + k.apiKey + ":" + k.accessToken
}

// checkAuth fails fast when the connector has no access token.
func (k *Kite) checkAuth() error {
	if strings.TrimSpace(k.apiKey) == "" {
		return fmt.Errorf("kite: api_key is not configured: %w", ErrAuth)
	}
	if strings.TrimSpace(k.accessToken) == "" {
		return fmt.Errorf("kite: access_token is not configured: %w", ErrAuth)
	}
	return nil
}

// do performs an authenticated Kite REST call and returns the response body.
func (k *Kite) do(ctx context.Context, method, endpoint string, form url.Values, maxBytes int64) ([]byte, error) {
	if err := k.checkAuth(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := k.pacer.Wait(ctx, requestCategory(endpoint)); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var body io.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("kite: build %s %s: %w: %w", method, endpoint, ErrInvalidOrder, err)
	}
	req.Header.Set("Authorization", k.authHeader())
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "mft-broker/1.0")
	if form != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	resp, err := k.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("kite: %s %s: %w: %w", method, endpoint, ErrUnavailable, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes))
	if err != nil {
		return nil, fmt.Errorf("kite: read %s %s: %w: %w", method, endpoint, ErrUnavailable, err)
	}

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, fmt.Errorf("kite: %s %s: %w", method, endpoint, parseKiteError(resp.StatusCode, raw))
	}
	return raw, nil
}

func requestCategory(endpoint string) RESTCategory {
	path := endpoint
	if parsed, err := url.Parse(endpoint); err == nil && parsed.Path != "" {
		path = parsed.Path
	}
	switch {
	case strings.HasPrefix(path, "/quote/"):
		return RESTQuote
	case strings.HasPrefix(path, "/instruments/historical/"):
		return RESTHistorical
	case strings.HasPrefix(path, "/orders"):
		return RESTOrder
	default:
		return RESTOther
	}
}
