// Package execution implements Service 3 of the MFT platform: the execution
// and risk engine. It validates trading signals against risk parameters and
// routes orders to the broker.
package execution

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/fluxkv"
	"github.com/mft/services/execution/risk"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module is the FX module for the execution service.
var Module = fx.Module("execution",
	fx.Provide(NewEngine),
	fx.Invoke(StartHTTPServer, RegisterEngine, StartCacheSweeper),
)

// IdempotencyTTL is the fallback for how long a processed idempotency key is
// remembered, used when the config does not set execution.idempotency_ttl_seconds.
//
// Inference retries a signal, so the key has to outlive the retry loop rather
// than the debounce window: debounce is 5 minutes by default, which is the
// minimum gap between two *different* decisions about the same symbol, not a
// bound on how long the same request may be retried. A trading day is 375
// minutes, so a day-long key cannot expire inside a session.
const IdempotencyTTL = 24 * time.Hour

// idempotencyTTL resolves the key lifetime from config, falling back to
// IdempotencyTTL when the key is absent or non-positive.
func idempotencyTTL(cfg *config.Config) time.Duration {
	if secs := cfg.Execution.IdempotencyTTLSeconds; secs > 0 {
		return time.Duration(secs) * time.Second
	}
	return IdempotencyTTL
}

// SweepInterval is how often the fluxKV TTL store is swept of expired
// entries. Reads already ignore expired entries, so this is memory
// housekeeping, not correctness; see the fluxkv package doc.
const SweepInterval = time.Minute

// Engine is the execution and risk gatekeeper.
//
// Signal processing is serialised by mu. The reason is exactly-once order
// placement: the idempotency claim, the broker call, and the portfolio update
// have to be one critical section or a concurrent replay of the same key can
// pass the idempotency check and place a second order. At one signal per
// minute per symbol the throughput cost is nil.
type Engine struct {
	client broker.Client
	cache  *fluxkv.KV
	cfg    *config.ExecutionConfig
	log    *zap.Logger

	checks      contracts.Checks
	book        *risk.Book
	clock       risk.Clock
	debounceTTL time.Duration
	idemTTL     time.Duration

	mu sync.Mutex

	ordersPlaced prometheus.Counter
	rejected     prometheus.Counter
	rejections   *prometheus.CounterVec
}

// NewEngine builds the execution engine with Prometheus metrics.
//
// The risk policy comes from the frozen execution config. One policy input
// has no config key and is therefore left at its documented default: per-symbol
// lot sizes (absent => lot size 1, correct for NSE cash equity delivery).
// The holiday calendar is built from execution.market_holidays.
func NewEngine(client broker.Client, cache *fluxkv.KV, cfg *config.Config, reg *prometheus.Registry, log *zap.Logger) *Engine {
	return newEngine(client, cache, cfg, reg, log, risk.FromConfig(cfg.Execution), risk.SystemClock)
}

// newEngine is the injectable constructor. Tests drive the clock and the
// policy through it instead of reaching into the struct.
func newEngine(
	client broker.Client,
	cache *fluxkv.KV,
	cfg *config.Config,
	reg *prometheus.Registry,
	log *zap.Logger,
	policy risk.Policy,
	clock risk.Clock,
) *Engine {
	factory := promauto.With(reg)
	return &Engine{
		client:      client,
		cache:       cache,
		cfg:         &cfg.Execution,
		log:         log,
		checks:      policy.New(cache),
		book:        risk.NewBook(cfg.Execution.Capital),
		debounceTTL: policy.DebounceTTL,
		idemTTL:     idempotencyTTL(cfg),
		clock:       clock,
		ordersPlaced: factory.NewCounter(prometheus.CounterOpts{
			Name: "mft_execution_orders_placed_total",
			Help: "Total number of orders placed.",
		}),
		rejected: factory.NewCounter(prometheus.CounterOpts{
			Name: "mft_execution_orders_rejected_total",
			Help: "Total number of order requests that produced no fill, whether rejected by risk or by the broker.",
		}),
		rejections: factory.NewCounterVec(prometheus.CounterOpts{
			Name: "mft_execution_rejections_total",
			Help: "Total number of signals rejected by the risk gate, by reason code.",
		}, []string{"reason"}),
	}
}

// ExecuteSignal processes a trade signal: it claims the idempotency key, runs
// the full risk chain, and places the order only if every check passes.
//
// On a risk rejection it returns the *contracts.Rejection and does not touch
// the broker. A replayed idempotency key returns the order id placed the
// first time along with a RISK_DUPLICATE rejection, so the caller can answer
// 409 with the original order id (docs/contracts.md §7).
func (e *Engine) ExecuteSignal(ctx context.Context, sig contracts.Signal) (string, error) {
	if err := risk.ValidateSignal(sig); err != nil {
		e.rejected.Inc()
		return "", err
	}
	return e.execute(ctx, sig, e.idemTTL)
}

// Execute places an order given as loose fields. It runs the same risk chain
// as ExecuteSignal.
//
// The order carries no idempotency key, so one is derived from the order's
// own content and lives only as long as the debounce window. That is
// deliberately weaker than ExecuteSignal's day-long key: it reproduces exactly
// the pre-existing behaviour of this endpoint — an identical order inside the
// debounce window is refused — without making a legitimate second identical
// order later in the day impossible. Callers that can carry a key should use
// ExecuteSignal.
func (e *Engine) Execute(ctx context.Context, symbol, side string, quantity int, price float64) (string, error) {
	sig := contracts.Signal{
		Symbol:   symbol,
		Side:     side,
		Quantity: quantity,
		Price:    price,
		AsOf:     e.clock(),
		// The key is derived from the order's own content, so the caller is
		// not asked for one this endpoint has no field for.
		IdempotencyKey: fmt.Sprintf("ORDER:%s:%s:%d:%.4f", symbol, side, quantity, price),
	}
	if err := risk.ValidateSignal(sig); err != nil {
		e.rejected.Inc()
		return "", err
	}
	return e.execute(ctx, sig, e.debounceTTL)
}

// Portfolio returns the engine's current exposure view. The positions map is a
// copy; the caller cannot mutate engine state through it.
func (e *Engine) Portfolio() contracts.Portfolio {
	return e.book.Snapshot()
}

// execute is the serialised body shared by both entry points.
func (e *Engine) execute(ctx context.Context, sig contracts.Signal, idemTTL time.Duration) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Claim first. A replayed key never reaches a check, let alone the
	// broker, and gets the order id the first attempt produced.
	if idemKey := idempotencyKey(sig.IdempotencyKey); idemKey != "" {
		if v, ok := e.cache.Get(idemKey); ok {
			orderID, _ := v.(string)
			e.rejected.Inc()
			e.rejections.WithLabelValues(contracts.ReasonDuplicate).Inc()
			return orderID, contracts.Reject(contracts.ReasonDuplicate,
				fmt.Sprintf("idempotency key %q already processed as order %s", sig.IdempotencyKey, orderID))
		}
	}

	if err := e.checks.Check(ctx, sig, e.book.Snapshot()); err != nil {
		e.rejected.Inc()
		var rej *contracts.Rejection
		if errors.As(err, &rej) {
			e.rejections.WithLabelValues(rej.Code).Inc()
		}
		return "", err
	}

	orderID, err := e.client.PlaceOrder(ctx, sig.Symbol, sig.Side, sig.Quantity, sig.Price)
	if err != nil {
		// Nothing is claimed and nothing is debounced: a failed placement
		// must be retryable, and a retry must be able to place the order.
		e.rejected.Inc()
		return "", fmt.Errorf("place order %s %s x%d: %w", sig.Symbol, sig.Side, sig.Quantity, err)
	}

	if idemKey := idempotencyKey(sig.IdempotencyKey); idemKey != "" {
		e.cache.Set(idemKey, orderID, idemTTL)
	}
	e.cache.Set(risk.DebounceKey(sig.Symbol, sig.Side), orderID, e.debounceTTL)

	// The broker filled the order; if the book cannot represent it, that is a
	// bug worth shouting about, not a reason to pretend the fill did not
	// happen. Log loudly and keep the fill.
	if err := e.book.Apply(sig); err != nil {
		e.log.Error("portfolio update failed after fill",
			zap.String("symbol", sig.Symbol),
			zap.String("side", sig.Side),
			zap.Int("quantity", sig.Quantity),
			zap.String("order_id", orderID),
			zap.Error(err))
	}

	e.ordersPlaced.Inc()
	e.log.Info("order placed",
		zap.String("order_id", orderID),
		zap.String("symbol", sig.Symbol),
		zap.String("side", sig.Side),
		zap.Int("quantity", sig.Quantity),
		zap.Float64("price", sig.Price),
		zap.String("idempotency_key", sig.IdempotencyKey))
	return orderID, nil
}

// idempotencyKey namespaces a caller-supplied key so it cannot collide with a
// debounce key.
func idempotencyKey(key string) string {
	if key == "" {
		return ""
	}
	return "IDEM:" + key
}

// StartCacheSweeper registers the periodic fluxKV sweep. Expired entries are
// already invisible to Get, so this only reclaims memory; it exists so that
// one-shot keys (debounce windows, idempotency keys) do not accumulate for the
// lifetime of the process.
func StartCacheSweeper(lc fx.Lifecycle, cache *fluxkv.KV, log *zap.Logger) {
	lc.Append(fx.Hook{
		OnStart: func(context.Context) error {
			go func() {
				ticker := time.NewTicker(SweepInterval)
				defer ticker.Stop()
				for range ticker.C {
					if n := cache.Sweep(time.Now()); n > 0 {
						log.Debug("swept expired fluxkv entries", zap.Int("count", n))
					}
				}
			}()
			return nil
		},
		OnStop: func(context.Context) error { return nil },
	})
}
