// Package execution implements Service 3 of the MFT platform: the execution
// and risk engine. It validates trading signals against risk parameters and
// routes orders to the broker.
package execution

import (
	"context"
	"fmt"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/fluxkv"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module is the FX module for the execution service.
var Module = fx.Module("execution",
	fx.Provide(NewEngine),
	fx.Invoke(StartHTTPServer, RegisterEngine),
)

// Engine is the execution and risk gatekeeper.
type Engine struct {
	client broker.Client
	cache  *fluxkv.KV
	cfg    *config.ExecutionConfig
	log    *zap.Logger

	ordersPlaced prometheus.Counter
	rejected     prometheus.Counter
}

// NewEngine builds the execution engine with Prometheus metrics.
func NewEngine(client broker.Client, cache *fluxkv.KV, cfg *config.Config, reg *prometheus.Registry, log *zap.Logger) *Engine {
	factory := promauto.With(reg)
	return &Engine{
		client: client,
		cache:  cache,
		cfg:    &cfg.Execution,
		log:    log,
		ordersPlaced: factory.NewCounter(prometheus.CounterOpts{
			Name: "execution_orders_placed_total",
			Help: "Total number of orders placed.",
		}),
		rejected: factory.NewCounter(prometheus.CounterOpts{
			Name: "execution_orders_rejected_total",
			Help: "Total number of orders rejected by risk checks.",
		}),
	}
}

// Execute processes a trade signal. It enforces signal debouncing via a TTL
// key in fluxKV and delegates order placement to the broker client.
func (e *Engine) Execute(ctx context.Context, symbol, side string, quantity int, price float64) (string, error) {
	debounceKey := fmt.Sprintf("EXEC:%s:%s", symbol, side)
	if _, ok := e.cache.Get(debounceKey); ok {
		e.rejected.Inc()
		return "", fmt.Errorf("signal debounced for %s", debounceKey)
	}

	orderID, err := e.client.PlaceOrder(ctx, symbol, side, quantity, price)
	if err != nil {
		e.rejected.Inc()
		return "", err
	}

	e.cache.Set(debounceKey, true, time.Duration(e.cfg.DebounceTTLSeconds)*time.Second)
	e.ordersPlaced.Inc()
	return orderID, nil
}
