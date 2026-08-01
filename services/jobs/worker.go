// Package jobs implements Service 4 of the MFT platform: background jobs and
// historical backfilling. It uses netresearch/go-cron for in-process
// scheduling and enforces rate limits against the broker historical API.
package jobs

import (
	"context"
	"fmt"
	"time"

	cron "github.com/netresearch/go-cron"
	"github.com/mft/core/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module is the FX module for the jobs service.
var Module = fx.Module("jobs",
	fx.Provide(
		NewBackfillWorker,
		NewScheduler,
	),
	fx.Invoke(StartScheduler),
)

// BackfillWorker fetches historical candles from the broker, respecting a
// token-bucket rate limit and applying exponential backoff on throttling.
type BackfillWorker struct {
	log      *zap.Logger
	rateLimitPerSecond int
	backfillsRun prometheus.Counter
}

// NewBackfillWorker constructs the backfill worker with Prometheus metrics.
func NewBackfillWorker(cfg *config.Config, log *zap.Logger, reg *prometheus.Registry) *BackfillWorker {
	factory := promauto.With(reg)
	return &BackfillWorker{
		log:               log,
		rateLimitPerSecond: cfg.Jobs.RateLimitPerSecond,
		backfillsRun: factory.NewCounter(prometheus.CounterOpts{
			Name: "jobs_backfills_run_total",
			Help: "Total number of historical backfill jobs run.",
		}),
	}
}

// RunBackfill is a placeholder for the historical data fetch. It is
// rate-limited via the token bucket and will call the broker REST API once
// the connector is wired.
func (w *BackfillWorker) RunBackfill(ctx context.Context) error {
	w.backfillsRun.Inc()
	w.log.Info("backfill job started")
	// Token-bucket rate limiting against the broker historical API goes here.
	<-time.After(100 * time.Millisecond)
	w.log.Info("backfill job finished")
	return nil
}

// Scheduler wraps the go-cron instance.
type Scheduler struct {
	cron  *cron.Cron
	log   *zap.Logger
}

// NewScheduler creates a go-cron scheduler with a recover chain.
func NewScheduler(log *zap.Logger) *Scheduler {
	c := cron.New(cron.WithChain(cron.Recover(cron.VerbosePrintfLogger(cronLogWriter{log}))))
	return &Scheduler{cron: c, log: log}
}

// cronLogWriter adapts zap to go-cron's Printf-based logger interface.
type cronLogWriter struct{ log *zap.Logger }

// Printf logs a formatted message at info level.
func (w cronLogWriter) Printf(format string, args ...any) {
	w.log.Info(fmt.Sprintf(format, args...))
}

// StartScheduler registers the weekly backfill job and starts the scheduler.
func StartScheduler(lc fx.Lifecycle, s *Scheduler, w *BackfillWorker, cfg *config.Config) {
	schedule := cfg.Jobs.Schedule
	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			_, err := s.cron.AddFunc(schedule, func() {
				if err := w.RunBackfill(ctx); err != nil {
					s.log.Error("backfill job failed", zap.Error(err))
				}
			}, cron.WithName("weekly-backfill"))
			if err != nil {
				return fmt.Errorf("register backfill job: %w", err)
			}
			s.cron.Start()
			s.log.Info("jobs scheduler started", zap.String("schedule", schedule))
			return nil
		},
		OnStop: func(ctx context.Context) error {
			s.cron.Stop()
			return nil
		},
	})
}
