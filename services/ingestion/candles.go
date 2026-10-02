package ingestion

import (
	"context"
	"fmt"

	"github.com/mft/core/config"
	"github.com/mft/core/storage"
)

// CandleStore is the ingestion module's candle writer, bound under its own
// type so the service can carry both a tick and a candle writer.
//
// # Why a local type
//
// core/fx.go provides a single *storage.Writer for the whole platform, and it
// is the foundation's file: a change there conflicts with every parallel
// branch. fx distinguishes constructors by result type, so declaring a
// *storage.CandleWriter here — the type core/fx.go does not provide — binds
// the second writer without touching it. core/fx.go would be the tidier home
// for it; this is the change that avoids the merge conflict.
type CandleStore struct {
	writer *storage.CandleWriter
}

// NewCandleStore builds the candle writer from the frozen storage config.
// Rows land under <data_dir>/candles, hive-partitioned by symbol and UTC
// date, which is the tree the inference service reads through DuckDB.
func NewCandleStore(cfg *config.Config) (CandleStore, error) {
	if cfg == nil {
		return CandleStore{}, fmt.Errorf("ingestion: candle store needs a config")
	}
	writer, err := storage.NewCandleWriterFromConfig(cfg.Storage)
	if err != nil {
		return CandleStore{}, fmt.Errorf("ingestion: candle writer: %w", err)
	}
	return CandleStore{writer: writer}, nil
}

// NewCandleStoreFromWriter wraps an existing candle writer. Tests use it to
// point the pipeline at a temporary directory without a config file.
func NewCandleStoreFromWriter(writer *storage.CandleWriter) (CandleStore, error) {
	if writer == nil {
		return CandleStore{}, fmt.Errorf("ingestion: candle store needs a writer")
	}
	return CandleStore{writer: writer}, nil
}

// Append buffers one candle.
func (c CandleStore) Append(candle storage.Candle) error { return c.writer.Append(candle) }

// Start launches the candle writer's background flusher.
func (c CandleStore) Start(ctx context.Context) error { return c.writer.Start(ctx) }

// Flush writes every buffered candle to disk.
func (c CandleStore) Flush(ctx context.Context) error { return c.writer.Flush(ctx) }

// Close stops the background flusher and writes the remaining candles.
func (c CandleStore) Close() error { return c.writer.Close() }

// Err returns the last error raised by a background flush, or nil.
func (c CandleStore) Err() error { return c.writer.Err() }

// Pending returns the number of candles buffered but not yet written.
func (c CandleStore) Pending() int { return c.writer.Pending() }
