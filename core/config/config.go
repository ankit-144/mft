// Package config loads and validates application configuration from a YAML file.
package config

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"time"

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
	Product                 string   `yaml:"product"`
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
	PaperTrading             bool     `yaml:"paper_trading"`
	JournalPath              string   `yaml:"journal_path"`
	APIToken                 string   `yaml:"api_token"`
	ReconcileIntervalSeconds int      `yaml:"reconcile_interval_seconds"`
	MaxPendingOrders         int      `yaml:"max_pending_orders"`
	MaxSignalAgeSeconds      int      `yaml:"max_signal_age_seconds"`
	Addr                     string   `yaml:"addr"`
	Capital                  float64  `yaml:"capital"`
	DebounceTTLSeconds       int      `yaml:"debounce_ttl_seconds"`
	IdempotencyTTLSeconds    int      `yaml:"idempotency_ttl_seconds"`
	MaxPositionPct           float64  `yaml:"max_position_pct"`
	MaxOpenPositions         int      `yaml:"max_open_positions"`
	MaxDrawdownPct           float64  `yaml:"max_drawdown_pct"`
	DailyLossLimit           float64  `yaml:"daily_loss_limit"`
	MaxOrderQuantity         int      `yaml:"max_order_quantity"`
	MarketHolidays           []string `yaml:"market_holidays"`
}

// InferenceConfig configures the TabFM inference loop.
type InferenceConfig struct {
	MaxConcurrency       int      `yaml:"max_concurrency"`
	CursorPath           string   `yaml:"cursor_path"`
	MaxContextAgeSeconds int      `yaml:"max_context_age_seconds"`
	Addr                 string   `yaml:"addr"`
	Model                string   `yaml:"model"`
	ExecutionURL         string   `yaml:"execution_url"`
	ContextRows          int      `yaml:"context_rows"`
	HorizonBars          int      `yaml:"horizon_bars"`
	ScoreThreshold       float64  `yaml:"score_threshold"`
	OrderQuantity        int      `yaml:"order_quantity"`
	Instruments          []string `yaml:"instruments"`
	DryRun               bool     `yaml:"dry_run"`
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

	cfg := Config{Execution: ExecutionConfig{PaperTrading: true}, Inference: InferenceConfig{DryRun: true}}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	for variable, target := range map[string]*string{
		"KITE_API_KEY":      &cfg.Broker.APIKey,
		"KITE_API_SECRET":   &cfg.Broker.APISecret,
		"KITE_ACCESS_TOKEN": &cfg.Broker.AccessToken,
	} {
		if value := os.Getenv(variable); value != "" {
			*target = value
		}
	}

	if addr := os.Getenv("MFT_METRICS_ADDR"); addr != "" {
		cfg.Metrics.Addr = addr
	}
	if addr := os.Getenv("MFT_EXECUTION_ADDR"); addr != "" {
		cfg.Execution.Addr = addr
	}
	if token := os.Getenv("MFT_EXECUTION_API_TOKEN"); token != "" {
		cfg.Execution.APIToken = token
	}
	if model := os.Getenv("MFT_INFERENCE_MODEL"); model != "" {
		cfg.Inference.Model = model
	}
	if url := os.Getenv("MFT_EXECUTION_URL"); url != "" {
		cfg.Inference.ExecutionURL = url
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("resolve config directory: %w", err)
	}
	if filepath.Base(base) == "configs" {
		base = filepath.Dir(base)
	}
	for _, p := range []*string{&cfg.Storage.DataDir, &cfg.Analytics.DuckDBPath, &cfg.Jobs.HistoricalDir, &cfg.Execution.JournalPath, &cfg.Inference.CursorPath} {
		if !filepath.IsAbs(*p) {
			*p = filepath.Join(base, *p)
		}
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
	if c.Broker.Product == "" {
		c.Broker.Product = "MIS"
	}
	if c.Storage.DataDir == "" {
		c.Storage.DataDir = "data"
	}
	if c.Storage.FlushIntervalSecs == 0 {
		c.Storage.FlushIntervalSecs = 60
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
		c.Execution.Addr = "127.0.0.1:8080"
	}
	if c.Execution.JournalPath == "" {
		c.Execution.JournalPath = filepath.Join(c.Storage.DataDir, "execution", "state.json")
	}
	if c.Execution.ReconcileIntervalSeconds == 0 {
		c.Execution.ReconcileIntervalSeconds = 5
	}
	if c.Execution.MaxPendingOrders == 0 {
		c.Execution.MaxPendingOrders = 64
	}
	if c.Execution.MaxSignalAgeSeconds == 0 {
		c.Execution.MaxSignalAgeSeconds = 120
	}
	if c.Execution.DebounceTTLSeconds == 0 {
		c.Execution.DebounceTTLSeconds = 300
	}
	if c.Execution.IdempotencyTTLSeconds == 0 {
		c.Execution.IdempotencyTTLSeconds = 86400
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
		c.Inference.Model = "heuristic"
	}
	if c.Inference.MaxConcurrency == 0 {
		c.Inference.MaxConcurrency = 4
	}
	if c.Inference.MaxContextAgeSeconds == 0 {
		c.Inference.MaxContextAgeSeconds = 120
	}
	if c.Inference.CursorPath == "" {
		c.Inference.CursorPath = filepath.Join(c.Storage.DataDir, "inference", "cursors.json")
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
	for name, value := range map[string]float64{"capital": c.Execution.Capital, "max_position_pct": c.Execution.MaxPositionPct, "max_drawdown_pct": c.Execution.MaxDrawdownPct, "daily_loss_limit": c.Execution.DailyLossLimit, "score_threshold": c.Inference.ScoreThreshold} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s must be finite", name)
		}
	}
	if c.Inference.ScoreThreshold < 0 || c.Inference.ScoreThreshold > 1 {
		return fmt.Errorf("inference.score_threshold: %v is outside [0, 1]", c.Inference.ScoreThreshold)
	}
	if !regexp.MustCompile(`^[a-z][a-z0-9_]*$`).MatchString(c.Inference.Model) {
		return fmt.Errorf("inference.model: %q must be a registry identifier", c.Inference.Model)
	}
	if c.Execution.MaxPositionPct <= 0 || c.Execution.MaxPositionPct > 100 {
		return fmt.Errorf("execution.max_position_pct: %v is outside (0, 100]", c.Execution.MaxPositionPct)
	}
	if c.Execution.Capital < 0 {
		return fmt.Errorf("execution.capital: %v must not be negative", c.Execution.Capital)
	}
	if c.Execution.DailyLossLimit < 0 || c.Execution.MaxDrawdownPct <= 0 || c.Execution.MaxDrawdownPct > 100 || c.Execution.MaxOpenPositions < 1 || c.Execution.MaxOrderQuantity < 1 {
		return fmt.Errorf("execution risk limits are invalid")
	}
	if !c.Execution.PaperTrading && (c.Execution.Capital <= 0 || c.Execution.APIToken == "") {
		return fmt.Errorf("live execution requires positive capital and execution.api_token")
	}
	if c.Execution.MaxPendingOrders < 1 || c.Execution.MaxSignalAgeSeconds < 1 || c.Execution.ReconcileIntervalSeconds < 1 {
		return fmt.Errorf("execution pending, age and reconciliation limits must be positive")
	}
	if c.Inference.MaxConcurrency < 1 || c.Inference.MaxConcurrency > 64 || c.Inference.MaxContextAgeSeconds < 1 {
		return fmt.Errorf("inference concurrency must be in [1, 64] and context age positive")
	}
	if c.Inference.ContextRows < 1 || c.Inference.HorizonBars < 1 || c.Inference.OrderQuantity < 1 || c.Inference.HorizonBars >= c.Inference.ContextRows {
		return fmt.Errorf("inference requires positive context, horizon and quantity, with horizon below context rows")
	}
	if c.Storage.FlushIntervalSecs < 1 || c.Storage.FlushMaxRows < 1 || c.Jobs.RateLimitPerSecond < 1 {
		return fmt.Errorf("storage flush and jobs rate limits must be positive")
	}
	if c.Broker.Product != "MIS" && c.Broker.Product != "NRML" && c.Broker.Product != "CNC" {
		return fmt.Errorf("broker.product: %q must be one of MIS, NRML, CNC", c.Broker.Product)
	}
	for _, day := range c.Execution.MarketHolidays {
		if _, err := time.Parse("2006-01-02", day); err != nil {
			return fmt.Errorf("execution.market_holidays: %q is not a YYYY-MM-DD date", day)
		}
	}
	return nil
}
