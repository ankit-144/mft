package execution

// This file is the execution service's HTTP transport: the four routes of
// docs/contracts.md §7, and nothing else. Every risk decision belongs to the
// engine; what lives here is decoding, wire-shape validation, status mapping,
// and error shaping. The one thing this file does own end to end is the
// promise that a replayed idempotency key places no second order and is told
// the order id the first attempt produced.

import (
	"context"
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

// Server timeouts. The pre-existing server set only ReadHeaderTimeout, which
// bounds the request line and the headers and nothing else: a client that
// trickles the body forever holds a connection, and a goroutine and a file
// descriptor with it, for as long as it likes. ReadTimeout closes that.
// WriteTimeout is deliberately longer than the connector's own
// broker.request_timeout_seconds (10s by default) so that a placement which
// is merely slow is not cut off mid-flight and left in an unknown state.
const (
	readHeaderTimeout = 5 * time.Second
	readTimeout       = 15 * time.Second
	writeTimeout      = 15 * time.Second
	idleTimeout       = 60 * time.Second
	shutdownTimeout   = 10 * time.Second
)

// maxRequestBody is the largest request body the API accepts. A signal is
// eight small fields; a body larger than this is a mistake or an attack, and
// both are refused before a byte reaches the risk gate.
const maxRequestBody = 64 << 10

// brokerProbeTimeout bounds the health endpoint's broker call. A liveness
// probe that can hang is a liveness probe that eventually gets the process
// killed; the connector's own timeout is the ceiling, this is the one that
// keeps /v1/health answering.
const brokerProbeTimeout = 3 * time.Second

// Order states reported on the wire, as docs/contracts.md §7 documents them.
const (
	statusFilled = "FILLED"
	statusOpen   = "OPEN"
)

// Transport error codes.
//
// These are deliberately disjoint from the contracts Reason* codes.
// mft_execution_rejections_total is keyed on a risk decision, and a request
// that never reached the risk gate — a typo, a truncated body, a broker outage
// — must not appear in it.
const (
	codeBadRequest       = "BAD_REQUEST"
	codePayloadTooBig    = "PAYLOAD_TOO_LARGE"
	codeNotFound         = "NOT_FOUND"
	codeMethodNotAllowed = "METHOD_NOT_ALLOWED"
	codeBrokerError      = "BROKER_UNAVAILABLE"
	codeInternal         = "INTERNAL_ERROR"
)

// knownRoutes is the route table docs/contracts.md §7 fixes, used only to tell
// "no such endpoint" apart from "right endpoint, wrong verb". net/http's mux
// answers the second case with 405 on its own, but only when nothing else
// matches, and the catch-all route below always does.
var knownRoutes = map[string]string{
	"/v1/signals":   http.MethodPost,
	"/v1/orders":    http.MethodPost,
	"/v1/portfolio": http.MethodGet,
	"/v1/health":    http.MethodGet,
}

// Executor is the engine surface this package depends on. *Engine is the
// production implementation; stating it as an interface keeps the transport
// honest about what it uses and lets the status mapping be tested without a
// risk chain.
type Executor interface {
	// ExecuteSignal claims the idempotency key, runs the risk chain, and
	// places the order only if every check passes.
	ExecuteSignal(ctx context.Context, sig contracts.Signal) (string, error)
	// Execute places an order given as loose fields, through the same chain.
	Execute(ctx context.Context, symbol, side string, quantity int, price float64) (string, error)
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
//
// The order id is carried only on a replayed idempotency key, which is the
// one failure where the caller needs it: it is how inference learns that the
// order it was retrying already exists and stops retrying.
type errorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
	OrderID string `json:"order_id,omitempty"`
}

// transportError is a failure the API can describe exactly: a code a caller
// can branch on, and a message worth showing. A risk Rejection is one of
// these already, so a single writer renders both.
type transportError struct {
	code    string
	message string
	orderID string
}

// Error implements the error interface.
func (e *transportError) Error() string { return e.code + ": " + e.message }

// api is the execution HTTP surface.
type api struct {
	engine Executor
	log    *zap.Logger
	health *metrics.Health

	requests *prometheus.CounterVec
	duration *prometheus.HistogramVec
	faults   *prometheus.CounterVec
}

// NewHandler builds the execution API handler: the four routes of
// docs/contracts.md §7, behind request-id propagation and request logging
// (log.Middleware), per-route metrics, and a per-request panic barrier.
func NewHandler(engine Executor, orders broker.OrderClient, reg prometheus.Registerer, logger *zap.Logger) http.Handler {
	a := &api{
		engine: engine,
		log:    logger,
		health: metrics.NewHealth("execution", brokerChecks(orders)),
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

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/signals", a.route("signals", a.handleSignal))
	mux.HandleFunc("POST /v1/orders", a.route("orders", a.handleOrder))
	mux.HandleFunc("GET /v1/portfolio", a.route("portfolio", a.handlePortfolio))
	mux.HandleFunc("GET /v1/health", a.route("health", a.handleHealth))
	mux.HandleFunc("/", a.route("unknown", a.handleUnknown))

	return log.Middleware(logger)(mux)
}

// StartHTTPServer serves the execution API on execution.addr for the lifetime
// of the fx application, and is registered by execution.Module.
//
// The listener is bound inside OnStart rather than inside the serving
// goroutine, so a port already in use fails the application instead of
// logging an error next to a service that is up and cannot be reached.
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
		Handler:           NewHandler(engine, orders, reg, logger),
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
		// Route net/http's own diagnostics — oversized headers, handshake
		// failures — into the same log stream as the requests.
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
			// Shutdown stops accepting, drains in-flight requests, then closes
			// idle connections. fx already bounds ctx; bounding it again makes
			// the wait explicit wherever this hook is driven from.
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

// route wraps a handler with the two things every route needs: metrics
// labelled by the route, and a panic barrier scoped to that request.
func (a *api) route(name string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		// Deferred calls run last-in-first-out, so the panic barrier is
		// registered second and therefore runs first: by the time the metrics
		// are recorded the 500 it wrote is already on the status recorder.
		defer func() {
			a.requests.WithLabelValues(name, r.Method, strconv.Itoa(rec.status)).Inc()
			a.duration.WithLabelValues(name).Observe(time.Since(start).Seconds())
		}()
		defer a.recoverPanic(rec, r)

		handler(rec, r)
	}
}

// recoverPanic contains a handler panic to the connection that caused it.
//
// net/http drops the connection when a handler panics, so without this one
// bug in one route tears down every request in flight; with it the caller
// gets a 500 in the same error shape as every other failure and the process
// keeps trading. The panic value is logged, never returned: it is internal
// detail and this body is read by inference.
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
		// The response is already on the wire and cannot be rewritten. The
		// truncated body and the dropped connection are the honest outcome.
		return
	}
	a.faults.WithLabelValues(codeInternal).Inc()
	writeJSON(rec, http.StatusInternalServerError, errorResponse{
		Error:   codeInternal,
		Message: "internal server error",
	})
}

// handleSignal implements POST /v1/signals: inference submits a signal, the
// full risk gate runs, and the order is placed only if every check passes.
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
		a.writeError(w, classify(err, orderID))
		return
	}

	status, state := outcomeFor(orderTypeFor("", sig.Price))
	writeJSON(w, status, signalResponse{OrderID: orderID, Status: state, Score: sig.Score})
}

// handleOrder implements POST /v1/orders: direct placement, through the same
// risk gate as a signal.
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

	orderID, err := a.engine.Execute(r.Context(), req.Symbol, req.Side, req.Quantity, req.Price)
	if err != nil {
		a.writeError(w, classify(err, orderID))
		return
	}

	status, state := outcomeFor(orderType)
	writeJSON(w, status, orderResponse{OrderID: orderID, Status: state})
}

// handlePortfolio implements GET /v1/portfolio: the engine's exposure view,
// which is exactly the contracts.Portfolio the risk checks are measured
// against.
func (a *api) handlePortfolio(w http.ResponseWriter, _ *http.Request) {
	portfolio := a.engine.Portfolio()
	if portfolio.OpenPositions == nil {
		// A flat book is {} on the wire, not null, so a caller can index the
		// map without special-casing an empty account.
		portfolio.OpenPositions = map[string]int{}
	}
	writeJSON(w, http.StatusOK, portfolio)
}

// handleHealth implements GET /v1/health: liveness plus broker connectivity.
//
// It answers 503 when the broker session is unusable. On this service a
// process that cannot reach Kite is not trading, however healthy it looks,
// and a liveness check that reported otherwise would be actively misleading.
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
//
// The price rule is the broker's own — broker.Kite.placeOrder places MARKET at
// a zero price and LIMIT above it — so the API reports the order type the
// exchange will see rather than a second, disagreeing opinion about it.
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

// outcomeFor maps an order type onto the status docs/contracts.md §7 promises.
//
// A market order has no resting state: it is filled as it is submitted or
// refused, so the order id the broker returns is a fill. A limit order rests
// at the exchange until something fills or cancels it, so the same order id
// describes a working order and nothing is claimed about the fill.
//
// This is a statement about what was asked for, not a report of what happened.
// ExecuteSignal returns an order id and no fill state, so the transport cannot
// observe a fill even in principle, and claiming more than it knows would be a
// lie an inference loop could act on.
//
// The consequence, stated plainly because it is a gap and not a design: this
// service places limit orders only. The risk chain sizes a position as
// quantity*price, so a zero-price market order is unmeasurable and both
// endpoints refuse it (see validateOrder), which leaves 202 Accepted / FILLED
// unreachable end to end. The mapping is implemented and specified so that the
// day the engine can size and place a market order, the API answers 202
// without another change here. Until then, a caller that asks for one gets a
// 400 naming the reason rather than a promise the engine cannot keep.
func outcomeFor(orderType string) (int, string) {
	if orderType == contracts.OrderTypeMarket {
		return http.StatusAccepted, statusFilled
	}
	return http.StatusOK, statusOpen
}

// normaliseSignal applies the canonical wire forms before validation. Kite
// matches instruments and sides case-insensitively while the risk gate
// compares them exactly, so the transport is the one layer that can bridge the
// two without either of them knowing about the other.
func normaliseSignal(sig *contracts.Signal) {
	sig.Symbol = strings.ToUpper(strings.TrimSpace(sig.Symbol))
	sig.Side = strings.ToUpper(strings.TrimSpace(sig.Side))
	sig.IdempotencyKey = strings.TrimSpace(sig.IdempotencyKey)
}

// validateSignal applies the wire-shape rules docs/contracts.md §7 states: a
// signal names a symbol, a side, a positive reference price, and — because
// inference retries, and a signal with no key cannot be made idempotent — an
// idempotency key.
//
// These are the same rules risk.ValidateSignal enforces, repeated at the edge
// on purpose. A malformed request must be answered 400 with a transport error
// code and must never reach the risk gate, because there is no reason code for
// a request that was never a decision, and inventing one would put an
// undeclared value into mft_execution_rejections_total. No risk policy is
// duplicated here: not one of these is a budget, a limit or a session check.
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
	}
	return nil
}

// validateOrder applies the wire-shape rules for POST /v1/orders.
//
// The price rule is the risk gate's own and it is load-bearing, not cosmetic:
// the chain sizes a position as quantity*price, so a zero-price order is not a
// small order, it is an unmeasurable one. risk.ValidateSignal refuses it, and
// so does this. Refusing it here is what lets classify treat every non-rejection
// from the engine as a placement failure rather than a guess.
//
// The declared type is checked before it is resolved, because resolving an
// unrecognised one to the price would silently place an order the caller did
// not ask for.
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
	case req.Price <= 0:
		return &transportError{code: codeBadRequest, message: fmt.Sprintf(
			"price %v must be positive: the risk gate sizes an order on quantity*price, "+
				"so this service places limit orders only", req.Price)}
	case orderType == contracts.OrderTypeMarket:
		return &transportError{code: codeBadRequest, message: "a MARKET order must not carry a price"}
	}
	return nil
}

// classify maps an engine error onto the wire.
//
// A *contracts.Rejection is a decision the risk gate already made and is
// surfaced verbatim: mft_execution_rejections_total is keyed on that code, and
// an inference loop branching on RISK_DEBOUNCED has to see the same string the
// metric was labelled with. The order id is attached on a replay, which is the
// one rejection where it carries information.
//
// Anything else is a placement failure. That is not a guess: the request has
// already passed every shape check by the time it reaches the engine, so the
// only work left in the engine is the chain and the broker call, and the chain
// reports as a Rejection. Answering 400 would tell inference to retry
// something that is not its fault, so it is 502.
func classify(err error, orderID string) error {
	var rejection *contracts.Rejection
	if errors.As(err, &rejection) {
		out := &transportError{code: rejection.Code, message: rejection.Message}
		if rejection.Code == contracts.ReasonDuplicate {
			out.orderID = orderID
		}
		return out
	}
	return &transportError{code: codeBrokerError, message: err.Error()}
}

// statusForCode maps an error code onto the status docs/contracts.md §7
// specifies: 409 for a replayed key, 400 for a risk rejection or a bad
// request, and a code that names the failure for everything else.
func statusForCode(code string) int {
	switch code {
	case contracts.ReasonDuplicate:
		return http.StatusConflict
	case codeNotFound:
		return http.StatusNotFound
	case codeMethodNotAllowed:
		return http.StatusMethodNotAllowed
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

// writeError renders a failure in the one error shape the API has. Any error
// that is not already a transportError is treated as an internal fault, so a
// missed classification degrades to a 500 rather than leaking a code that was
// never declared.
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
	})
}

// decodeJSON reads a bounded, strict JSON body into dst.
//
// Strict means unknown fields and trailing content are refused. That is not
// fussiness: a mistyped idempotency_key decodes to an empty string, the
// required-field check below rejects it as BAD_REQUEST, and a caller that
// retried the same typo would be refused for as long as it kept sending the
// request the server did not understand.
func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return decodeError(err)
	}
	// A second value means the caller sent something that is not the request
	// this endpoint expects, whatever the first value said.
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return &transportError{code: codeBadRequest,
			message: "body must contain exactly one JSON object"}
	}
	return nil
}

// decodeError separates the two ways a decode fails that a caller can act on
// differently: a body over the limit, which will fail again however it is
// retried, and a body that is not the JSON that was asked for.
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
//
// It asks for positions rather than pinging: /positions is the cheapest
// authenticated read Kite offers, and it fails on a missing or expired access
// token — the failure this endpoint exists to surface — before it can fail on
// an order.
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

// writeJSON renders v as the whole response. Marshalling before writing the
// header means an encoding failure is still a clean 500 rather than a 200 with
// a truncated body.
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

// statusRecorder remembers the status a handler chose, defaulting to 200 the
// way net/http does when a handler never calls WriteHeader, and records
// whether a response has been committed so the panic barrier knows if it can
// still rewrite one.
type statusRecorder struct {
	http.ResponseWriter
	status    int
	committed bool
}

// WriteHeader records the first status written. A second call is passed
// through to net/http, which ignores it, and must not overwrite the record.
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
