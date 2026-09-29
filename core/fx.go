// Package core provides the shared FX module with common dependencies for
// all MFT services: config, logger, metrics registry, fluxKV cache, and
// storage.
package core

import (
	"os"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/log"
	"github.com/mft/core/metrics"
	"github.com/mft/core/storage"
	"go.uber.org/fx"
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
		NewKite,
		func(k *broker.Kite) broker.Streamer { return k },
		func(k *broker.Kite) broker.Client { return k },
		broker.NewOrderClient,
	),
	fx.Invoke(metrics.Server),
)

// NewKite builds the broker connector from config. Binding NewKite here
// instead of the bare broker.NewKite matters: the zero-config constructor
// leaves the connector without credentials or a watchlist, so a service
// started through fx would fail to authenticate on every call.
func NewKite(cfg *config.Config) (*broker.Kite, error) {
	return broker.NewKiteFromConfig(cfg.Broker)
}

// ConfigPath resolves the config file path, honoring the MFT_CONFIG
// environment variable and defaulting to configs/config.yaml.
func ConfigPath() string {
	if p := os.Getenv("MFT_CONFIG"); p != "" {
		return p
	}
	return "configs/config.yaml"
}

// NewStorageWriter constructs the Parquet tick writer from config.
//
// The writer's lifecycle is owned by whoever consumes it: ingestion starts and
// stops both writers itself so it can drain the tick channel first, so this
// constructor must not Start anything.
func NewStorageWriter(cfg *config.Config) (*storage.Writer, error) {
	return storage.NewWriterFromConfig(cfg.Storage)
}
