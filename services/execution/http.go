package execution

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/log"
	"github.com/mft/core/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Server timeouts.
const (
	readHeaderTimeout = 5 * time.Second
	readTimeout       = 15 * time.Second
	writeTimeout      = 15 * time.Second
	idleTimeout       = 60 * time.Second
	shutdownTimeout   = 10 * time.Second
)

// maxRequestBody is the largest request body the API accepts.
const maxRequestBody = 64 << 10

// brokerProbeTimeout bounds the health endpoint's broker call.
const brokerProbeTimeout = 3 * time.Second

// Transport error codes.
const (
	codeBadRequest       = "BAD_REQUEST"
	codePayloadTooBig    = "PAYLOAD_TOO_LARGE"
	codeNotFound         = "NOT_FOUND"
	codeMethodNotAllowed = "METHOD_NOT_ALLOWED"
	codeBrokerError      = "BROKER_UNAVAILABLE"
	codeInternal         = "INTERNAL_ERROR"
	codeUnauthorized     = "UNAUTHORIZED"
	codeOutcomeUnknown   = "ORDER_OUTCOME_UNKNOWN"
)

// knownRoutes is the route table docs/contracts.md §7 fixes, used only to tell "no
// such endpoint" apart from "right endpoint, wrong verb".
var knownRoutes = map[string]string{
	"/v1/signals":   http.MethodPost,
	"/v1/orders":    http.MethodPost,
	"/v1/portfolio": http.MethodGet,
	"/v1/health":    http.MethodGet,
}

// Executor is the engine surface this package depends on.
type Executor interface {
	// ExecuteSignal claims the idempotency key, runs the risk chain, and places the order
	// only if every check passes.
	ExecuteSignal(ctx context.Context, sig contracts.Signal) (string, error)
	// ExecuteOrder claims the request key and applies the same risk chain.
	ExecuteOrder(ctx context.Context, req contracts.OrderRequest) (string, error)
	// OrderStatus reports the latest broker state for an idempotency key.
	OrderStatus(key string) contracts.OrderResult
	// Portfolio returns the engine's current exposure view.
	Portfolio() contracts.Portfolio
}

// Executor is satisfied by the engine built in engine.go.
var _ Executor = (*Engine)(nil)

// signalResponse is the body of an accepted POST /v1/signals.
type signalResponse struct {
	OrderID string  `json:"order_id"`
	Status  string  `json:"status"`
	Score   float64 `json:"score"`
}

// orderResponse is the body of an accepted POST /v1/orders.
type orderResponse struct {
	OrderID string `json:"order_id"`
	Status  string `json:"status"`
}

// errorResponse is the body of every failure, per docs/contracts.md §7.
type errorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
	OrderID string `json:"order_id,omitempty"`
	Status  string `json:"status,omitempty"`
}

// transportError is a failure the API can describe exactly: a code a caller can branch
// on, and a message worth showing.
type transportError struct {
	code    string
	message string
	orderID string
	status  string
}

// Error implements the error interface.
func (e *transportError) Error() string { return e.code + ": " + e.message }

// api is the execution HTTP surface.
type api struct {
	engine Executor
	log    *zap.Logger
	health *metrics.Health

	requests    *prometheus.CounterVec
	duration    *prometheus.HistogramVec
	faults      *prometheus.CounterVec
	token       string
	requireAuth bool
}

// NewHandler builds the authenticated execution API handler and readiness probes.
func NewHandler(engine Executor, orders broker.OrderClient, reg prometheus.Registerer, logger *zap.Logger, cfg ...*config.ExecutionConfig) http.Handler {
	checks := metrics.NewChecks()
	paper := len(cfg) > 0 && cfg[0] != nil && cfg[0].PaperTrading
	if !paper {
		checks.Add(metrics.Checker{Name: "broker", Check: probeBroker(orders)})
	}
	if readiness, ok := engine.(interface{ Ready() (bool, string) }); ok {
		checks.Add(metrics.Checker{Name: "execution", Check: func(context.Context) error {
			if ready, reason := readiness.Ready(); !ready {
				return errors.New(reason)
			}
			return nil
		}})
	}
	a := &api{
		engine: engine,
		log:    logger,
		health: metrics.NewHealth("execution", checks),
		requests: metrics.CounterVec(reg, "mft_execution_http_requests_total",
			"HTTP requests served by the execution API, by route, method and status code.",
			"route", "method", "status"),
		duration: metrics.HistogramVec(reg, "mft_execution_http_request_duration_seconds",
			"Wall-clock time to serve one execution API request, by route.", "route"),
		faults: metrics.CounterVec(reg, "mft_execution_http_faults_total",
			"Execution API responses that are not an accepted order, by transport error code. "+
				"Risk rejections are counted separately by mft_execution_rejections_total.",
			"code"),
	}
	if len(cfg) > 0 && cfg[0] != nil {
		a.requireAuth = !cfg[0].PaperTrading || cfg[0].APIToken != ""
		a.token = cfg[0].APIToken
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/signals", a.route("signals", a.handleSignal))
	mux.HandleFunc("POST /v1/orders", a.route("orders", a.handleOrder))
	mux.HandleFunc("GET /v1/portfolio", a.route("portfolio", a.handlePortfolio))
	mux.HandleFunc("GET /v1/health", a.route("health", a.handleHealth))
	mux.HandleFunc("/", a.route("unknown", a.handleUnknown))

	return log.Middleware(logger)(a.authorize(mux))
}

// StartHTTPServer serves the execution API on execution.addr for the lifetime of the fx
// application, and is registered by execution.Module.
func StartHTTPServer(
	lc fx.Lifecycle,
	engine *Engine,
	orders broker.OrderClient,
	cfg *config.Config,
	reg *prometheus.Registry,
	logger *zap.Logger,
) {
	srv := &http.Server{
		Addr:              cfg.Execution.Addr,
		Handler:           NewHandler(engine, orders, reg, logger, &cfg.Execution),
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,

		ErrorLog: zap.NewStdLog(logger),
	}

	lc.Append(fx.Hook{
		OnStart: func(context.Context) error {
			ln, err := net.Listen("tcp", cfg.Execution.Addr)
			if err != nil {
				return fmt.Errorf("execution: listen on %s: %w", cfg.Execution.Addr, err)
			}
			logger.Info("execution HTTP server listening", zap.String("addr", ln.Addr().String()))
			go func() {
				if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
					logger.Error("execution HTTP server error", zap.Error(err))
				}
			}()
			return nil
		},
		OnStop: func(ctx context.Context) error {

			ctx, cancel := context.WithTimeout(ctx, shutdownTimeout)
			defer cancel()
			if err := srv.Shutdown(ctx); err != nil {
				return fmt.Errorf("execution: shut down http server: %w", err)
			}
			logger.Info("execution HTTP server stopped")
			return nil
		},
	})
}

// RegisterEngine ensures the engine is constructed at startup.
func RegisterEngine(*Engine) {}

// route wraps a handler with the two things every route needs: metrics labelled by the
// route, and a panic barrier scoped to that request.
func (a *api) route(name string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		defer func() {
			a.requests.WithLabelValues(name, r.Method, strconv.Itoa(rec.status)).Inc()
			a.duration.WithLabelValues(name).Observe(time.Since(start).Seconds())
		}()
		defer a.recoverPanic(rec, r)

		handler(rec, r)
	}
}

// recoverPanic contains a handler panic to the connection that caused it.
func (a *api) recoverPanic(rec *statusRecorder, r *http.Request) {
	cause := recover()
	if cause == nil {
		return
	}
	log.WithContext(r.Context(), a.log).Error("panic serving request",
		zap.String("method", r.Method),
		zap.String("path", r.URL.Path),
		zap.Any("panic", cause),
		zap.ByteString("stack", debug.Stack()))
	if rec.committed {

		return
	}
	a.faults.WithLabelValues(codeInternal).Inc()
	writeJSON(rec, http.StatusInternalServerError, errorResponse{
		Error:   codeInternal,
		Message: "internal server error",
	})
}

// handleSignal implements POST /v1/signals: inference submits a signal, the full risk
// gate runs, and the order is placed only if every check passes.
func (a *api) handleSignal(w http.ResponseWriter, r *http.Request) {
	var sig contracts.Signal
	if err := decodeJSON(w, r, &sig); err != nil {
		a.writeError(w, err)
		return
	}
	normaliseSignal(&sig)
	if err := validateSignal(sig); err != nil {
		a.writeError(w, err)
		return
	}

	orderID, err := a.engine.ExecuteSignal(r.Context(), sig)
	if err != nil {
		te := classify(err, orderID)
		te.status = a.engine.OrderStatus(sig.IdempotencyKey).Status
		a.writeError(w, te)
		return
	}

	state := a.engine.OrderStatus(sig.IdempotencyKey).Status
	writeJSON(w, responseCode(state), signalResponse{OrderID: orderID, Status: state, Score: sig.Score})
}

// handleOrder implements POST /v1/orders: direct placement, through the same risk gate
// as a signal.
func (a *api) handleOrder(w http.ResponseWriter, r *http.Request) {
	var req contracts.OrderRequest
	if err := decodeJSON(w, r, &req); err != nil {
		a.writeError(w, err)
		return
	}
	req.Symbol = strings.ToUpper(strings.TrimSpace(req.Symbol))
	req.Side = strings.ToUpper(strings.TrimSpace(req.Side))
	req.Type = strings.ToUpper(strings.TrimSpace(req.Type))

	orderType := orderTypeFor(req.Type, req.Price)
	if err := validateOrder(req, orderType); err != nil {
		a.writeError(w, err)
		return
	}

	if req.IdempotencyKey == "" {
		a.writeError(w, &transportError{code: codeBadRequest, message: "idempotency_key is required on POST /v1/orders"})
		return
	}
	orderID, err := a.engine.ExecuteOrder(r.Context(), req)
	if err != nil {
		te := classify(err, orderID)
		te.status = a.engine.OrderStatus(req.IdempotencyKey).Status
		a.writeError(w, te)
		return
	}

	state := a.engine.OrderStatus(req.IdempotencyKey).Status
	writeJSON(w, responseCode(state), orderResponse{OrderID: orderID, Status: state})
}

// handlePortfolio implements GET /v1/portfolio: the engine's exposure view, which is
// exactly the contracts.Portfolio the risk checks are measured against.
func (a *api) handlePortfolio(w http.ResponseWriter, _ *http.Request) {
	portfolio := a.engine.Portfolio()
	if portfolio.OpenPositions == nil {

		portfolio.OpenPositions = map[string]int{}
	}
	writeJSON(w, http.StatusOK, portfolio)
}

// handleHealth reports execution readiness and live broker connectivity.
func (a *api) handleHealth(w http.ResponseWriter, r *http.Request) {
	report, ready := a.health.Readyz(r.Context())
	if !ready {
		writeJSON(w, http.StatusServiceUnavailable, report)
		return
	}
	writeJSON(w, http.StatusOK, report)
}

// handleUnknown answers every unrouted request in the API's error shape, and
// distinguishes an unknown path from a known path reached with the wrong verb.
func (a *api) handleUnknown(w http.ResponseWriter, r *http.Request) {
	if method, ok := knownRoutes[r.URL.Path]; ok {
		w.Header().Set("Allow", method)
		a.writeError(w, &transportError{code: codeMethodNotAllowed, message: fmt.Sprintf(
			"%s is %s, not %s", r.URL.Path, method, r.Method)})
		return
	}
	a.writeError(w, &transportError{
		code:    codeNotFound,
		message: "no route for " + r.Method + " " + r.URL.Path,
	})
}

// orderTypeFor resolves the order type from a request, letting an explicit
// contracts.OrderRequest.Type override the price.
func orderTypeFor(declared string, price float64) string {
	switch t := strings.ToUpper(strings.TrimSpace(declared)); t {
	case contracts.OrderTypeMarket, contracts.OrderTypeLimit:
		return t
	}
	if price > 0 {
		return contracts.OrderTypeLimit
	}
	return contracts.OrderTypeMarket
}

func responseCode(state string) int {
	switch state {
	case contracts.OrderStatusSubmitting, contracts.OrderStatusOpen, contracts.OrderStatusPartial:
		return http.StatusAccepted
	case contracts.OrderStatusUnknown:
		return http.StatusServiceUnavailable
	default:
		return http.StatusOK
	}
}

// normaliseSignal applies the canonical wire forms before validation.
func normaliseSignal(sig *contracts.Signal) {
	sig.Symbol = strings.ToUpper(strings.TrimSpace(sig.Symbol))
	sig.Side = strings.ToUpper(strings.TrimSpace(sig.Side))
	sig.IdempotencyKey = strings.TrimSpace(sig.IdempotencyKey)
}

// validateSignal applies the wire-shape rules docs/contracts.md §7 states: a signal
// names a symbol, a side, a positive reference price, and — because inference
// retries, and a signal with no key cannot be made idempotent — an idempotency key.
func validateSignal(sig contracts.Signal) error {
	switch {
	case sig.Symbol == "":
		return &transportError{code: codeBadRequest, message: "symbol is required"}
	case sig.Side != contracts.SideBuy && sig.Side != contracts.SideSell:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"side %q must be %s or %s", sig.Side, contracts.SideBuy, contracts.SideSell)}
	case sig.Price <= 0:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"price %v must be positive", sig.Price)}
	case sig.IdempotencyKey == "":
		return &transportError{code: codeBadRequest,
			message: "idempotency_key is required on POST /v1/signals"}
	case sig.AsOf.IsZero():
		return &transportError{code: codeBadRequest, message: "as_of is required on POST /v1/signals"}
	}
	return nil
}

// validateOrder applies the wire-shape rules for POST /v1/orders.
func validateOrder(req contracts.OrderRequest, orderType string) error {
	switch {
	case req.Symbol == "":
		return &transportError{code: codeBadRequest, message: "symbol is required"}
	case req.Side != contracts.SideBuy && req.Side != contracts.SideSell:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"side %q must be %s or %s", req.Side, contracts.SideBuy, contracts.SideSell)}
	case req.Type != "" && req.Type != contracts.OrderTypeMarket && req.Type != contracts.OrderTypeLimit:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"type %q must be %s or %s", req.Type, contracts.OrderTypeMarket, contracts.OrderTypeLimit)}
	case req.Quantity < 1:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"quantity %d must be at least 1", req.Quantity)}
	case orderType == contracts.OrderTypeLimit && req.Price <= 0:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"price %v must be positive: the risk gate sizes an order on quantity*price, "+
				"so LIMIT orders require a price", req.Price)}
	case orderType == contracts.OrderTypeMarket && req.Price != 0:
		return &transportError{code: codeBadRequest, message: "a MARKET order must not carry a limit price"}
	}
	return nil
}

// classify maps an engine error onto the wire.
func classify(err error, orderID string) *transportError {
	var rejection *contracts.Rejection
	if errors.As(err, &rejection) {
		out := &transportError{code: rejection.Code, message: rejection.Message}
		if rejection.Code == contracts.ReasonDuplicate {
			out.orderID = orderID
		}
		return out
	}
	var indeterminate *IndeterminateError
	if errors.As(err, &indeterminate) {
		return &transportError{code: codeOutcomeUnknown, message: "broker may have accepted this order; retry with the same idempotency key after reconciliation", orderID: orderID}
	}
	return &transportError{code: codeBrokerError, message: err.Error()}
}

func (a *api) authorize(next http.Handler) http.Handler {
	if !a.requireAuth {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		provided := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if a.token == "" || subtle.ConstantTimeCompare([]byte(provided), []byte(a.token)) != 1 {
			a.writeError(w, &transportError{code: codeUnauthorized, message: "a valid bearer token is required"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// statusForCode maps an error code onto the status docs/contracts.md §7 specifies: 409
// for a replayed key, 400 for a risk rejection or a bad request, and a code that names
// the failure for everything else.
func statusForCode(code string) int {
	switch code {
	case contracts.ReasonDuplicate:
		return http.StatusConflict
	case codeNotFound:
		return http.StatusNotFound
	case codeMethodNotAllowed:
		return http.StatusMethodNotAllowed
	case codeUnauthorized:
		return http.StatusUnauthorized
	case codeOutcomeUnknown:
		return http.StatusServiceUnavailable
	case codePayloadTooBig:
		return http.StatusRequestEntityTooLarge
	case codeBrokerError:
		return http.StatusBadGateway
	case codeInternal:
		return http.StatusInternalServerError
	default:
		return http.StatusBadRequest
	}
}

// writeError renders a failure in the one error shape the API has.
func (a *api) writeError(w http.ResponseWriter, err error) {
	var te *transportError
	if !errors.As(err, &te) {
		te = &transportError{code: codeInternal, message: "internal server error"}
	}
	if te.message == "" {
		te.message = te.code
	}
	a.faults.WithLabelValues(te.code).Inc()
	writeJSON(w, statusForCode(te.code), errorResponse{
		Error:   te.code,
		Message: te.message,
		OrderID: te.orderID,
		Status:  te.status,
	})
}

// decodeJSON reads a bounded, strict JSON body into dst.
func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return decodeError(err)
	}

	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return &transportError{code: codeBadRequest,
			message: "body must contain exactly one JSON object"}
	}
	return nil
}

// decodeError separates the two ways a decode fails that a caller can act on
// differently: a body over the limit, which will fail again however it is retried, and
// a body that is not the JSON that was asked for.
func decodeError(err error) error {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return &transportError{code: codePayloadTooBig, message: fmt.Sprintf(
			"request body exceeds the %d byte limit", maxRequestBody)}
	}
	return &transportError{code: codeBadRequest, message: "malformed JSON body: " + err.Error()}
}

// brokerChecks returns the dependency probes behind GET /v1/health.
func brokerChecks(orders broker.OrderClient) *metrics.Checks {
	checks := metrics.NewChecks()
	checks.Add(metrics.Checker{Name: "broker", Check: probeBroker(orders)})
	return checks
}

// probeBroker reports whether the broker session is usable.
func probeBroker(orders broker.OrderClient) func(context.Context) error {
	return func(ctx context.Context) error {
		if orders == nil {
			return errors.New("no broker order client is bound")
		}
		ctx, cancel := context.WithTimeout(ctx, brokerProbeTimeout)
		defer cancel()
		if _, err := orders.GetPositions(ctx); err != nil {
			return fmt.Errorf("broker positions: %w", err)
		}
		return nil
	}
}

// writeJSON renders v as the whole response.
func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := json.Marshal(v)
	if err != nil {
		body = []byte(`{"error":"` + codeInternal + `","message":"could not encode response"}`)
		status = http.StatusInternalServerError
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(append(body, '\n'))
}

// statusRecorder remembers the status a handler chose, defaulting to 200 the way
// net/http does when a handler never calls WriteHeader, and records whether a response
// has been committed so the panic barrier knows if it can still rewrite one.
type statusRecorder struct {
	http.ResponseWriter
	status    int
	committed bool
}

// WriteHeader records the first status written.
func (s *statusRecorder) WriteHeader(status int) {
	if !s.committed {
		s.status = status
		s.committed = true
	}
	s.ResponseWriter.WriteHeader(status)
}

// Write records that a response was committed even if the handler never called
// WriteHeader, which net/http treats as an implicit 200.
func (s *statusRecorder) Write(b []byte) (int, error) {
	s.committed = true
	return s.ResponseWriter.Write(b)
}
