// Package metrics provides the shared Prometheus foundation for every MFT
// service: the naming rule, constructors, build information, latency
// histograms, and the scrape and health endpoints.
//
// Every metric this platform exposes carries the "mft_" prefix. That single
// rule is what lets one Prometheus instance, one Grafana dashboard and one
// alert set cover all four services without two of them shadowing each other.
// The constructors here enforce it at registration time, so a violation panics
// at startup rather than scraping under a foreign name.
//
// The helpers are opt-in. A component that prefers promauto.With(reg) keeps
// working; only the name it registers is subject to review.
package metrics

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/mft/core/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// shutdownTimeout bounds the graceful stop of the metrics server, so a wedged
// scrape connection cannot hold up application shutdown.
const shutdownTimeout = 5 * time.Second

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
// OpenMetrics is enabled so that exemplars — the correlation id attached by
// ObserveDuration — survive the scrape.
func Handler(reg *prometheus.Registry) http.Handler {
	return promhttp.HandlerFor(reg, promhttp.HandlerOpts{EnableOpenMetrics: true})
}

// Mux returns the routes the metrics server exposes: the scrape handler at
// cfg.Metrics.Path and the health endpoints at /healthz and /readyz. The
// health handler is the process-wide DefaultChecks set.
//
// A scrape path of "/" is honoured but leaves no room for the health
// endpoints; the default is /metrics and nothing in this platform needs that.
func Mux(cfg *config.Config, scrape http.Handler) *http.ServeMux {
	mux := http.NewServeMux()
	path := metricsPath(cfg)
	mux.Handle(path, scrape)
	if path != "/" {
		mux.Handle("/", NewHealth(cfg.App.Name, DefaultChecks()))
	}
	return mux
}

// metricsPath returns the configured scrape path, defaulting to /metrics.
func metricsPath(cfg *config.Config) string {
	if cfg == nil || cfg.Metrics.Path == "" {
		return "/metrics"
	}
	return cfg.Metrics.Path
}

// Server registers the metrics HTTP server lifecycle hook. It starts the
// server on app start and shuts it down gracefully on app stop.
//
// The listener is opened during OnStart, not in a goroutine, so a port already
// in use fails startup instead of leaving a process that looks healthy but
// cannot be scraped.
func Server(lc fx.Lifecycle, cfg *config.Config, handler http.Handler, log *zap.Logger) {
	srv := &http.Server{
		Addr:              cfg.Metrics.Addr,
		Handler:           Mux(cfg, handler),
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			ln, err := net.Listen("tcp", srv.Addr)
			if err != nil {
				return fmt.Errorf("listen on metrics addr %s: %w", srv.Addr, err)
			}
			log.Info("metrics server listening",
				zap.String("addr", srv.Addr),
				zap.String("path", metricsPath(cfg)),
				zap.Duration("uptime", time.Since(processStart)),
			)
			go func() {
				if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
					log.Error("metrics server error", zap.Error(err))
				}
			}()
			return nil
		},
		OnStop: func(ctx context.Context) error {
			ctx, cancel := context.WithTimeout(ctx, shutdownTimeout)
			defer cancel()
			// Shutdown waits for in-flight scrapes to finish, so the last
			// scrape of a stopping process still sees a consistent snapshot.
			if err := srv.Shutdown(ctx); err != nil {
				log.Error("metrics server shutdown", zap.Error(err))
				return fmt.Errorf("shutdown metrics server: %w", err)
			}
			log.Info("metrics server stopped")
			return nil
		},
	})
}
