// Package metrics provides shared Prometheus metrics registration and an
// HTTP endpoint for scraping.
package metrics

import (
	"context"
	"net/http"
	"time"

	"github.com/mft/core/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Registry returns the shared Prometheus registry pre-loaded with Go and
// process collectors.
func Registry() *prometheus.Registry {
	reg := prometheus.NewRegistry()
	reg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)
	return reg
}

// Handler returns an http.Handler that serves Prometheus metrics from reg.
func Handler(reg *prometheus.Registry) http.Handler {
	return promhttp.HandlerFor(reg, promhttp.HandlerOpts{})
}

// Server registers the metrics HTTP server lifecycle hook. It starts the
// server on app start and shuts it down gracefully on app stop.
func Server(lc fx.Lifecycle, cfg *config.Config, handler http.Handler, log *zap.Logger) {
	srv := &http.Server{
		Addr:              cfg.Metrics.Addr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
	}

	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			go func() {
				log.Info("metrics server listening", zap.String("addr", cfg.Metrics.Addr))
				if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
					log.Error("metrics server error", zap.Error(err))
				}
			}()
			return nil
		},
		OnStop: func(ctx context.Context) error {
			return srv.Shutdown(context.Background())
		},
	})
}
