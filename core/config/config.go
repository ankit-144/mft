// Package config loads and validates application configuration from a YAML file.
//
// The schema declared here is FROZEN. Every key any MFT component needs is
// already present, so that ten independently-developed components can add
// features without editing this file. See docs/contracts.md §8.
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
	Analytics AnalyticsConfig `yaml:"analytics"`
	Execution ExecutionConfig `yaml:"execution"`
	Inference InferenceConfig `yaml:"inference"`
	Jobs      JobsConfig      `yaml:"jobs"`
}

// AppConfig holds general application settings.
type AppConfig struct {
	Name     string `yaml:"name"`
	Env      string `yaml:"env"`
	LogLevel string `yaml:"log_level"`
	Timezone string `yaml:"timezone"`
}

// MetricsConfig configures the Prometheus metrics endpoint.
type MetricsConfig struct {
	Addr string `yaml:"addr"`
	Path string `yaml:"path"`
}

// BrokerConfig holds Zerodha Kite Connect credentials and watched instruments.
type BrokerConfig struct {
	APIKey                  string   `yaml:"api_key"`
	APISecret               string   `yaml:"api_secret"`
	AccessToken             string   `yaml:"access_token"`
	Instruments             []string `yaml:"instruments"`
	ReconnectMaxBackoffSecs int      `yaml:"reconnect_max_backoff_seconds"`
	RequestTimeoutSeconds   int      `yaml:"request_timeout_seconds"`
}

// StorageConfig configures the Parquet cold store.
type StorageConfig struct {
	DataDir           string `yaml:"data_dir"`
	FlushIntervalSecs int    `yaml:"flush_interval_seconds"`
	FlushMaxRows      int    `yaml:"flush_max_rows"`
	PartitionBy       string `yaml:"partition_by"`
}

// AnalyticsConfig configures the embedded DuckDB query engine.
type AnalyticsConfig struct {
	DuckDBPath string `yaml:"duckdb_path"`
	ReadOnly   bool   `yaml:"read_only"`
}

// ExecutionConfig configures the execution & risk engine.
type ExecutionConfig struct {
	Addr               string  `yaml:"addr"`
	Capital            float64 `yaml:"capital"`
	DebounceTTLSeconds int     `yaml:"debounce_ttl_seconds"`
	MaxPositionPct     float64 `yaml:"max_position_pct"`
	MaxOpenPositions   int     `yaml:"max_open_positions"`
	MaxDrawdownPct     float64 `yaml:"max_drawdown_pct"`
	DailyLossLimit     float64 `yaml:"daily_loss_limit"`
	MaxOrderQuantity   int     `yaml:"max_order_quantity"`
}

// InferenceConfig configures the TabFM inference loop.
type InferenceConfig struct {
	Addr           string   `yaml:"addr"`
	Model          string   `yaml:"model"`
	ExecutionURL   string   `yaml:"execution_url"`
	ContextRows    int      `yaml:"context_rows"`
	HorizonBars    int      `yaml:"horizon_bars"`
	ScoreThreshold float64  `yaml:"score_threshold"`
	OrderQuantity  int      `yaml:"order_quantity"`
	Instruments    []string `yaml:"instruments"`
	DryRun         bool     `yaml:"dry_run"`
}

// JobsConfig configures the background jobs scheduler.
type JobsConfig struct {
	Schedule             string `yaml:"schedule"`
	RateLimitPerSecond   int    `yaml:"rate_limit_per_second"`
	HistoricalDir        string `yaml:"historical_dir"`
	BackfillLookbackDays int    `yaml:"backfill_lookback_days"`
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

// Validate fills in defaults and rejects configurations that are internally
// inconsistent.
func (c *Config) Validate() error {
	if c.App.Name == "" {
		c.App.Name = "mft"
	}
	if c.App.Timezone == "" {
		c.App.Timezone = "Asia/Kolkata"
	}
	if c.Metrics.Addr == "" {
		c.Metrics.Addr = ":9090"
	}
	if c.Metrics.Path == "" {
		c.Metrics.Path = "/metrics"
	}
	if c.Broker.ReconnectMaxBackoffSecs == 0 {
		c.Broker.ReconnectMaxBackoffSecs = 60
	}
	if c.Broker.RequestTimeoutSeconds == 0 {
		c.Broker.RequestTimeoutSeconds = 10
	}
	if c.Storage.DataDir == "" {
		c.Storage.DataDir = "data"
	}
	if c.Storage.FlushIntervalSecs == 0 {
		c.Storage.FlushIntervalSecs = 300
	}
	if c.Storage.FlushMaxRows == 0 {
		c.Storage.FlushMaxRows = 10000
	}
	if c.Storage.PartitionBy == "" {
		c.Storage.PartitionBy = "date"
	}
	if c.Analytics.DuckDBPath == "" {
		c.Analytics.DuckDBPath = c.Storage.DataDir + "/mft.duckdb"
	}
	if c.Execution.Addr == "" {
		c.Execution.Addr = ":8080"
	}
	if c.Execution.DebounceTTLSeconds == 0 {
		c.Execution.DebounceTTLSeconds = 300
	}
	if c.Execution.MaxPositionPct == 0 {
		c.Execution.MaxPositionPct = 10.0
	}
	if c.Execution.MaxOpenPositions == 0 {
		c.Execution.MaxOpenPositions = 10
	}
	if c.Execution.MaxDrawdownPct == 0 {
		c.Execution.MaxDrawdownPct = 5.0
	}
	if c.Execution.MaxOrderQuantity == 0 {
		c.Execution.MaxOrderQuantity = 500
	}
	if c.Inference.Addr == "" {
		c.Inference.Addr = ":8000"
	}
	if c.Inference.Model == "" {
		c.Inference.Model = "tabfm"
	}
	if c.Inference.ExecutionURL == "" {
		c.Inference.ExecutionURL = "http://localhost:8080"
	}
	if c.Inference.ContextRows == 0 {
		c.Inference.ContextRows = 100
	}
	if c.Inference.HorizonBars == 0 {
		c.Inference.HorizonBars = 1
	}
	if c.Inference.OrderQuantity == 0 {
		c.Inference.OrderQuantity = 10
	}
	if c.Jobs.Schedule == "" {
		c.Jobs.Schedule = "0 2 * * 6"
	}
	if c.Jobs.RateLimitPerSecond == 0 {
		c.Jobs.RateLimitPerSecond = 3
	}
	if c.Jobs.HistoricalDir == "" {
		c.Jobs.HistoricalDir = c.Storage.DataDir + "/historical"
	}
	if c.Jobs.BackfillLookbackDays == 0 {
		c.Jobs.BackfillLookbackDays = 730
	}

	return c.crossValidate()
}

// crossValidate rejects combinations that would be unsafe at runtime.
func (c *Config) crossValidate() error {
	if c.Inference.ScoreThreshold < 0 || c.Inference.ScoreThreshold > 1 {
		return fmt.Errorf("inference.score_threshold: %v is outside [0, 1]", c.Inference.ScoreThreshold)
	}
	if c.Inference.Model != "tabfm" && c.Inference.Model != "heuristic" {
		return fmt.Errorf("inference.model: %q must be one of tabfm, heuristic", c.Inference.Model)
	}
	if c.Execution.MaxPositionPct <= 0 || c.Execution.MaxPositionPct > 100 {
		return fmt.Errorf("execution.max_position_pct: %v is outside (0, 100]", c.Execution.MaxPositionPct)
	}
	if c.Execution.Capital < 0 {
		return fmt.Errorf("execution.capital: %v must not be negative", c.Execution.Capital)
	}
	return nil
}
