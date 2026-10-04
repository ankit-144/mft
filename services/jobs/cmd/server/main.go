// Command server runs the MFT background jobs service.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"go.uber.org/fx"

	"github.com/mft/core"
	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/log"
	"github.com/mft/core/metrics"
	"github.com/mft/services/jobs"
)

func main() {
	runNow := flag.Bool("run-now", false,
		"run one backfill pass now and exit instead of waiting for the cron schedule")
	flag.Parse()

	if *runNow {
		if err := runOnce(); err != nil {
			fmt.Fprintf(os.Stderr, "backfill failed: %v\n", err)
			os.Exit(1)
		}
		return
	}

	fx.New(
		core.Module,
		jobs.Module,
	).Run()
}

// runOnce builds the same graph the scheduler uses, runs a single backfill, and exits.
func runOnce() error {
	cfg, err := config.Load(core.ConfigPath())
	if err != nil {
		return err
	}
	logger, err := log.New(cfg)
	if err != nil {
		return err
	}
	defer func() { _ = logger.Sync() }()

	kite, err := broker.NewKiteFromConfig(cfg.Broker)
	if err != nil {
		return fmt.Errorf("build broker connector: %w", err)
	}
	kite.SetLogger(logger)

	worker := jobs.NewBackfillWorker(
		cfg,
		jobs.NewKiteHistory(cfg, kite, logger),
		jobs.NewParquetStore(cfg.Jobs.HistoricalDir),
		jobs.NewLimiterFromConfig(cfg),
		logger,
		metrics.Registry(),
	)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	return worker.RunBackfill(ctx)
}
