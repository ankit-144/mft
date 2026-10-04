package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFillsDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := []byte("app:\n  name: test\n")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.App.Name != "test" {
		t.Errorf("app.name = %q, want %q", cfg.App.Name, "test")
	}
	if cfg.Metrics.Addr != ":9090" {
		t.Errorf("metrics.addr = %q, want default :9090", cfg.Metrics.Addr)
	}
	if cfg.Jobs.Schedule != "0 2 * * 6" {
		t.Errorf("jobs.schedule = %q, want default", cfg.Jobs.Schedule)
	}
	if cfg.Jobs.RateLimitPerSecond != 3 {
		t.Errorf("jobs.rate_limit_per_second = %d, want 3", cfg.Jobs.RateLimitPerSecond)
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "nope.yaml")); err == nil {
		t.Fatal("expected error for missing config file")
	}
}

func TestBrokerEnvironmentOverridesLocalConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("broker:\n  api_key: yaml-key\n  api_secret: yaml-secret\n  access_token: yaml-token\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("KITE_API_KEY", "fixture-key")
	t.Setenv("KITE_API_SECRET", "fixture-secret")
	t.Setenv("KITE_ACCESS_TOKEN", "fixture-token")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Broker.APIKey != "fixture-key" || cfg.Broker.APISecret != "fixture-secret" || cfg.Broker.AccessToken != "fixture-token" {
		t.Fatal("broker environment overrides were not applied")
	}
}

func TestLoadInvalidYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("app:\n  name: [unclosed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for invalid YAML")
	}
}

func TestSafeDefaultsAndStablePaths(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "configs", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Execution.PaperTrading || !cfg.Inference.DryRun {
		t.Fatal("unsafe execution defaults")
	}
	if cfg.Execution.Addr != "127.0.0.1:8080" || cfg.Inference.Model != "heuristic" {
		t.Fatal("unsafe service defaults")
	}
	if cfg.Storage.DataDir != filepath.Join(dir, "data") {
		t.Fatalf("data dir = %s", cfg.Storage.DataDir)
	}
	if cfg.Execution.JournalPath != filepath.Join(dir, "data", "execution", "state.json") {
		t.Fatal(cfg.Execution.JournalPath)
	}
}

func TestLiveOptInRequiresTokenAndCapital(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("execution:\n  paper_trading: false\n  capital: 1000\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("live execution accepted without token")
	}
	t.Setenv("MFT_EXECUTION_API_TOKEN", "test-token")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Execution.PaperTrading || cfg.Execution.APIToken != "test-token" {
		t.Fatal("explicit opt-in not preserved")
	}
}
