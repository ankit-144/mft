package ingestion

import (
	"context"
	"fmt"

	"github.com/mft/core/config"
	"github.com/mft/core/storage"
)

// CandleStore is the ingestion module's candle writer, bound under its own type so the
// service can carry both a tick and a candle writer.
type CandleStore struct {
	writer           *storage.CandleWriter
	appendWithStatus func(storage.Candle) (bool, error)
}

// NewCandleStore builds the candle writer from the shared storage config.
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

// NewCandleStoreFromWriter wraps an existing candle writer.
func NewCandleStoreFromWriter(writer *storage.CandleWriter) (CandleStore, error) {
	if writer == nil {
		return CandleStore{}, fmt.Errorf("ingestion: candle store needs a writer")
	}
	return CandleStore{writer: writer}, nil
}

// Append buffers one candle.
func (c CandleStore) Append(candle storage.Candle) error { return c.writer.Append(candle) }

// AppendWithStatus distinguishes rejection from an accepted candle whose writer
// retained rows after a flush failure.
func (c CandleStore) AppendWithStatus(candle storage.Candle) (bool, error) {
	if c.appendWithStatus != nil {
		return c.appendWithStatus(candle)
	}
	return c.writer.AppendWithStatus(candle)
}

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
