// Command server runs the MFT background jobs service.
//
// With no flags it starts the go-cron scheduler and waits. With -run-now it
// runs exactly one backfill pass in the foreground and exits, which is what
// makes backfill on-demand rather than weekly-only: an operator does not have
// to wait for Saturday 02:00 to come round, and a CI job does not need a
// timetable attached to get a fresh historical tree.
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

// runOnce builds the same graph the scheduler uses, runs a single backfill,
// and exits. It is built by hand rather than through fx so that the one-shot
// path does not bind the metrics port: an operator backfilling on a laptop
// should not fail because a second copy of the service is already running.
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

	// The connector is used only for instrument resolution: which exchange a
	// symbol trades on, and its instrument token. It is not the order path,
	// and this service never places an order.
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
