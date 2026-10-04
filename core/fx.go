// Package core provides the shared FX module with common dependencies for all MFT
// services: config, logger, metrics registry, fluxKV cache, and storage.
package core

import (
	"os"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/log"
	"github.com/mft/core/metrics"
	"github.com/mft/core/storage"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
)

// Module bundles the core dependencies into an FX module.
var Module = fx.Module("core",
	fx.Provide(
		ConfigPath,
		LoadConfig,
		log.New,
		metrics.Registry,
		metrics.Handler,
		fluxkv.New,
		NewStorageWriter,
		NewKite,
		func(k *broker.Kite) broker.Streamer { return k },
		func(k *broker.Kite) broker.Client { return k },
		broker.NewOrderClient,
	),
	fx.Invoke(RegisterBuildInfo, metrics.Server),
)

// LoadConfig applies the launcher's service identity to the shared configuration.
func LoadConfig(path string) (*config.Config, error) {
	cfg, err := config.Load(path)
	if err != nil {
		return nil, err
	}
	if name := os.Getenv("MFT_SERVICE_NAME"); name != "" {
		cfg.App.Name = name
	}
	return cfg, nil
}

// RegisterBuildInfo exposes the current binary's identity in its metrics registry.
func RegisterBuildInfo(reg *prometheus.Registry, cfg *config.Config) error {
	return metrics.RegisterBuildInfo(reg, metrics.BuildInfo{Service: cfg.App.Name, Env: cfg.App.Env})
}

// NewKite builds the broker connector from config.
func NewKite(cfg *config.Config) (*broker.Kite, error) {
	return broker.NewKiteFromConfig(cfg.Broker)
}

// ConfigPath resolves the config file path, honoring the MFT_CONFIG environment
// variable and defaulting to configs/config.yaml.
func ConfigPath() string {
	if p := os.Getenv("MFT_CONFIG"); p != "" {
		return p
	}
	return "configs/config.yaml"
}

// NewStorageWriter constructs the Parquet tick writer from config.
func NewStorageWriter(cfg *config.Config) (*storage.Writer, error) {
	return storage.NewWriterFromConfig(cfg.Storage)
}
