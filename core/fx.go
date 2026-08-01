// Package core provides the shared FX module with common dependencies for
// all MFT services: config, logger, metrics registry, fluxKV cache, and
// storage.
package core

import (
	"context"
	"os"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/log"
	"github.com/mft/core/metrics"
	"github.com/mft/core/storage"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module bundles the core dependencies into an FX module.
var Module = fx.Module("core",
	fx.Provide(
		ConfigPath,
		config.Load,
		log.New,
		metrics.Registry,
		metrics.Handler,
		fluxkv.New,
		NewStorageWriter,
		broker.NewKite,
		func(k *broker.Kite) broker.Streamer { return k },
		func(k *broker.Kite) broker.Client { return k },
	),
	fx.Invoke(metrics.Server),
)

// ConfigPath resolves the config file path, honoring the MFT_CONFIG
// environment variable and defaulting to configs/config.yaml.
func ConfigPath() string {
	if p := os.Getenv("MFT_CONFIG"); p != "" {
		return p
	}
	return "configs/config.yaml"
}

// NewStorageWriter constructs the Parquet storage writer from config.
func NewStorageWriter(cfg *config.Config) (*storage.Writer, error) {
	return storage.NewWriter(cfg.Storage.DataDir+"/ticks", 10000), nil
}

// ShutdownStorage registers the storage writer for graceful shutdown.
var ShutdownStorage = fx.Invoke(
	func(lc fx.Lifecycle, w *storage.Writer, log *zap.Logger) {
		lc.Append(fx.Hook{
			OnStop: func(ctx context.Context) error {
				log.Info("stopping storage writer")
				return nil
			},
		})
	},
)

// Registry exposes the shared registry for service-level metric registration.
func Registry(r *prometheus.Registry) *prometheus.Registry {
	return r
}
