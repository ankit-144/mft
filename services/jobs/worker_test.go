package jobs

import (
	"context"
	"testing"

	"github.com/mft/core/config"
	"github.com/mft/core/testutil"
)

func TestRunBackfill(t *testing.T) {
	cfg := &config.Config{}
	cfg.Jobs.RateLimitPerSecond = 3

	reg := testutil.NewRegistry()
	w := NewBackfillWorker(cfg, testutil.NewLogger(), reg)
	if err := w.RunBackfill(context.Background()); err != nil {
		t.Fatalf("RunBackfill() error = %v", err)
	}

	got := testutil.MetricValue(t, "jobs_backfills_run_total", reg, nil)
	if got != 1 {
		t.Fatalf("expected backfills_run_total = 1, got %d", got)
	}
}

func TestNewBackfillWorkerUsesConfig(t *testing.T) {
	cfg := &config.Config{}
	cfg.Jobs.RateLimitPerSecond = 7

	w := NewBackfillWorker(cfg, testutil.NewLogger(), testutil.NewRegistry())
	if w.rateLimitPerSecond != 7 {
		t.Fatalf("rateLimitPerSecond = %d, want 7", w.rateLimitPerSecond)
	}
}
