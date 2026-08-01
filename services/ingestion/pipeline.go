// Package ingestion implements Service 1 of the MFT platform: the tick
// ingestion pipeline. It reads broker ticks, updates the fluxKV hot-path
// cache, and buffers raw ticks for Parquet cold storage.
package ingestion

import (
	"context"

	"github.com/mft/core/broker"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/storage"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module is the FX module for the ingestion service.
var Module = fx.Module("ingestion",
	fx.Provide(NewPipeline),
	fx.Invoke((*Pipeline).Run),
)

// Pipeline wires the broker stream into the fluxKV cache and storage writer.
type Pipeline struct {
	streamer broker.Streamer
	cache    *fluxkv.KV
	storage  *storage.Writer
	log      *zap.Logger

	ticksProcessed prometheus.Counter
}

// NewPipeline builds the ingestion pipeline with Prometheus metrics.
func NewPipeline(streamer broker.Streamer, cache *fluxkv.KV, storage *storage.Writer, reg *prometheus.Registry, log *zap.Logger) *Pipeline {
	factory := promauto.With(reg)
	return &Pipeline{
		streamer: streamer,
		cache:    cache,
		storage:  storage,
		log:      log,
		ticksProcessed: factory.NewCounter(prometheus.CounterOpts{
			Name: "ingestion_ticks_processed_total",
			Help: "Total number of ticks processed by the ingestion pipeline.",
		}),
	}
}

// Run starts the pipeline lifecycle: a tick channel consumed by a processor
// goroutine, with the broker stream feeding it.
func (p *Pipeline) Run(lc fx.Lifecycle) {
	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			if err := p.storage.Start(ctx); err != nil {
				return err
			}

			ticks := make(chan broker.Tick, 1024)

			go func() {
				processor := p.process(ticks)
				_ = processor
			}()

			go func() {
				if err := p.streamer.Stream(ctx, nil, ticks); err != nil {
					p.log.Error("broker stream ended", zap.Error(err))
				}
			}()

			p.log.Info("ingestion pipeline started")
			return nil
		},
		OnStop: func(ctx context.Context) error {
			p.log.Info("ingestion pipeline stopped")
			return nil
		},
	})
}

func (p *Pipeline) process(ticks <-chan broker.Tick) func() {
	return func() {
		for t := range ticks {
			p.cache.UpdateCandle(t.Symbol, t.Timestamp, t.Price, t.Volume)
			p.storage.Append(storage.Tick{
				InstrumentToken: 0,
				Timestamp:       t.Timestamp.UnixMilli(),
				LastPrice:       t.Price,
				Volume:          t.Volume,
			})
			p.ticksProcessed.Inc()
		}
	}
}
