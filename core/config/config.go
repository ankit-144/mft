// Package config loads and validates application configuration from a YAML file.
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration shared across all MFT services.
type Config struct {
	App       AppConfig       `yaml:"app"`
	Metrics   MetricsConfig   `yaml:"metrics"`
	Broker    BrokerConfig    `yaml:"broker"`
	Storage   StorageConfig   `yaml:"storage"`
	Execution ExecutionConfig `yaml:"execution"`
	Jobs      JobsConfig      `yaml:"jobs"`
}

// AppConfig holds general application settings.
type AppConfig struct {
	Name      string `yaml:"name"`
	Env       string `yaml:"env"`
	LogLevel  string `yaml:"log_level"`
}

// MetricsConfig configures the Prometheus metrics endpoint.
type MetricsConfig struct {
	Addr string `yaml:"addr"`
}

// BrokerConfig holds Zerodha Kite Connect credentials and watched instruments.
type BrokerConfig struct {
	APIKey      string   `yaml:"api_key"`
	APISecret   string   `yaml:"api_secret"`
	AccessToken string   `yaml:"access_token"`
	Instruments []string `yaml:"instruments"`
}

// StorageConfig configures local data directories.
type StorageConfig struct {
	DataDir string `yaml:"data_dir"`
}

// ExecutionConfig configures the execution & risk engine.
type ExecutionConfig struct {
	Addr             string  `yaml:"addr"`
	DebounceTTLSeconds int   `yaml:"debounce_ttl_seconds"`
	MaxDrawdownPct   float64 `yaml:"max_drawdown_pct"`
	Capital          float64 `yaml:"capital"`
}

// JobsConfig configures the background jobs scheduler.
type JobsConfig struct {
	Schedule          string `yaml:"schedule"`
	RateLimitPerSecond int   `yaml:"rate_limit_per_second"`
	HistoricalDir     string `yaml:"historical_dir"`
}

// Load reads and parses the YAML configuration file at path.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// Validate performs basic validation of required configuration fields.
func (c *Config) Validate() error {
	if c.App.Name == "" {
		c.App.Name = "mft"
	}
	if c.Metrics.Addr == "" {
		c.Metrics.Addr = ":9090"
	}
	if c.Storage.DataDir == "" {
		c.Storage.DataDir = "data"
	}
	if c.Jobs.Schedule == "" {
		c.Jobs.Schedule = "0 2 * * 6"
	}
	if c.Jobs.RateLimitPerSecond == 0 {
		c.Jobs.RateLimitPerSecond = 3
	}
	return nil
}
