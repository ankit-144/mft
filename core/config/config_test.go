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
