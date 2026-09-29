// Package storage persists market data to hive-partitioned Apache Parquet
// files. It is the durable substrate of the platform: ticks and one-minute
// candles are written here by ingestion, and read back by the jobs service,
// the Go analytics helpers, and — through the very same files — DuckDB inside
// the Python inference service.
//
// # The immutability invariant
//
// A Parquet file, once closed, is never opened again for writing. Not
// appended to, not rewritten, not truncated. Each flush produces a brand new
// file, and readers only ever open files that are already complete.
//
// This is not a stylistic preference. It is the invariant that lets the
// inference service read history while ingestion is still writing to the same
// tree (docs/contracts.md §3, "DuckDB rules"). A reader never has to take a
// lock, because there is no mutable state to lock.
//
// The package enforces it three ways:
//
//  1. Files are written to a temporary name (".inflight-*.parquet") and then
//     atomically renamed to "part-<timestamp>.parquet". The final name appears
//     only once the file is complete and fsynced, so a reader globbing
//     "part-*.parquet" can never observe a half-written file.
//  2. Final file names are chosen to not exist (os.Lstat), so a completed file
//     is never opened for writing, let alone overwritten.
//  3. Rows still buffered after a failed flush are retained, never discarded.
//     See writer.flushLocked.
//
// # Layout
//
//	data/ticks/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<timestamp>.parquet
//	data/candles/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<timestamp>.parquet
//
// The partition key is taken from the row itself (Tick.Symbol, Candle.Symbol,
// and the UTC date of the row's timestamp) — never from configuration or from
// a list of instruments — so a symbol that was not configured at startup still
// lands in the right place.
//
// # Errors
//
// Every filesystem and encoding error is wrapped with %w and returned. A
// buffer that fails to flush keeps its rows and is retried; nothing is
// silently dropped. Errors raised by the background flusher are recorded and
// surfaced by Err.
//
// # A note on names
//
// docs/contracts.md §3 declares `Writer[T any]` as the generic writer
// interface. The tick writer named `Writer` already exists on the base branch
// and is bound by core/fx.go and services/ingestion, both owned by other
// components, so renaming it would break their builds. The interface therefore
// ships as RowWriter[T] with the identical method set; the concrete writers
// (Writer for ticks, CandleWriter for candles) satisfy it, and the compile-time
// assertions at the bottom of this file are what hold that guarantee.
package storage

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
)

const (
	// DefaultFlushInterval is the flush cadence used by NewWriter, which
	// predates the config wiring and takes no interval.
	DefaultFlushInterval = 5 * time.Minute

	// DefaultFlushMaxRows is the buffer size used when none is configured.
	DefaultFlushMaxRows = 10000

	// DateLayout is the UTC date component of a hive partition directory.
	DateLayout = "2006-01-02"

	// PartitionByDate is the only partitioning scheme the frozen config
	// schema admits (storage.partition_by).
	PartitionByDate = "date"
)

// Dataset is the name of a Parquet dataset under the storage data directory.
type Dataset string

// The datasets the platform writes.
const (
	// DatasetTicks holds raw broker ticks.
	DatasetTicks Dataset = "ticks"
	// DatasetCandles holds aggregated one-minute bars.
	DatasetCandles Dataset = "candles"
)

// ErrClosed is returned by Append and Flush once Close has been called.
var ErrClosed = errors.New("storage: writer is closed")

// RowWriter is the writer contract from docs/contracts.md §3. Both Writer and
// CandleWriter satisfy it for their own row type.
type RowWriter[T any] interface {
	// Append adds one row to the current buffer.
	Append(row T) error
	// Flush writes every buffered row to disk and returns.
	Flush(ctx context.Context) error
	// Flush stops the background flusher and writes any remaining rows.
	Close() error
}

// Tick is the Parquet row for a raw broker tick. Timestamps are Unix
// milliseconds in UTC, per docs/contracts.md §1.
type Tick struct {
	Symbol          string  `parquet:"symbol"`
	InstrumentToken int64   `parquet:"instrument_token"`
	Timestamp       int64   `parquet:"timestamp"`
	LastPrice       float64 `parquet:"last_price"`
	Volume          int64   `parquet:"volume"`
}

// NewTick converts a domain tick into its Parquet row form.
func NewTick(t contracts.Tick) Tick {
	return Tick{
		Symbol:          t.Symbol,
		InstrumentToken: t.Token,
		Timestamp:       t.Timestamp.UTC().UnixMilli(),
		LastPrice:       t.Price,
		Volume:          t.Volume,
	}
}

// Contract converts the Parquet row back into its domain form.
func (t Tick) Contract() contracts.Tick {
	return contracts.Tick{
		Symbol:    t.Symbol,
		Token:     t.InstrumentToken,
		Price:     t.LastPrice,
		Volume:    t.Volume,
		Timestamp: time.UnixMilli(t.Timestamp).UTC(),
	}
}

// Candle is the Parquet row for an aggregated one-minute bar. Timestamps are
// Unix milliseconds in UTC, per docs/contracts.md §1.
type Candle struct {
	Symbol    string  `parquet:"symbol"`
	Timestamp int64   `parquet:"timestamp"`
	Open      float64 `parquet:"open"`
	High      float64 `parquet:"high"`
	Low       float64 `parquet:"low"`
	Close     float64 `parquet:"close"`
	Volume    int64   `parquet:"volume"`
}

// NewCandle converts a domain candle into its Parquet row form.
func NewCandle(c contracts.Candle) Candle {
	return Candle{
		Symbol:    c.Symbol,
		Timestamp: c.Timestamp.UTC().UnixMilli(),
		Open:      c.Open,
		High:      c.High,
		Low:       c.Low,
		Close:     c.Close,
		Volume:    c.Volume,
	}
}

// Contract converts the Parquet row back into its domain form.
func (c Candle) Contract() contracts.Candle {
	return contracts.Candle{
		Symbol:    c.Symbol,
		Timestamp: time.UnixMilli(c.Timestamp).UTC(),
		Open:      c.Open,
		High:      c.High,
		Low:       c.Low,
		Close:     c.Close,
		Volume:    c.Volume,
	}
}

// Writer persists raw broker ticks to the ticks dataset.
//
// Write and flush are serialised by a single mutex held across the whole
// encode-and-rename, so two flushes can never interleave and a closed file is
// never reopened. Append therefore blocks for the duration of a threshold
// flush; that is the deliberate price of the invariant above.
type Writer struct {
	w *writer[Tick]
}

// NewWriter returns a tick writer that buffers under dir and flushes every
// flushEvery rows, using the default interval.
//
// It keeps the pre-config two-argument shape so that core/fx.go and
// services/ingestion compile unchanged. New code should use NewWriterFromConfig,
// which honours storage.flush_interval_seconds and storage.flush_max_rows.
func NewWriter(dir string, flushEvery int) *Writer {
	opts := Options{
		Root:          dir,
		MaxRows:       flushEvery,
		FlushInterval: DefaultFlushInterval,
	}
	return &Writer{w: newWriter(opts, DatasetTicks, tickPartition)}
}

// NewWriterFromConfig builds a tick writer from the frozen storage config
// (docs/contracts.md §8). Rows land under <data_dir>/ticks, partitioned by
// symbol and UTC date.
func NewWriterFromConfig(cfg config.StorageConfig) (*Writer, error) {
	opts, err := optionsFromConfig(cfg, DatasetTicks)
	if err != nil {
		return nil, err
	}
	return &Writer{w: newWriter(opts, DatasetTicks, tickPartition)}, nil
}

// Append buffers a tick. When the buffer reaches flush_max_rows the batch is
// written before Append returns, so any filesystem error is reported to the
// caller rather than lost in a background goroutine.
func (w *Writer) Append(t Tick) error { return w.w.append(t) }

// Start launches the background flusher, which writes buffered rows every
// storage.flush_interval_seconds and once more when ctx is cancelled. Start is
// not required for correctness: without it, rows are still written by Append
// once the buffer fills, and by Close. Starting twice returns an error.
func (w *Writer) Start(ctx context.Context) error { return w.w.start(ctx) }

// Flush writes every buffered row to disk. It is safe to call concurrently
// with Append and with the background flusher; flushes never interleave.
func (w *Writer) Flush(ctx context.Context) error { return w.w.flush(ctx) }

// Close stops the background flusher and writes the remaining rows, then
// releases the writer. It is idempotent and returns the error from the final
// flush; a background failure that a later flush recovered from is reported by
// Err rather than here, because the rows did land.
func (w *Writer) Close() error { return w.w.close() }

// Err returns the last error raised by a background flush, or nil if the most
// recent flush succeeded. It is the only way to observe a failure that
// happened on the flusher goroutine rather than on the caller's stack.
func (w *Writer) Err() error { return w.w.err() }

// Pending returns the number of rows buffered but not yet written.
func (w *Writer) Pending() int { return w.w.pending() }

// CandleWriter persists aggregated one-minute candles to the candles dataset.
//
// It has the same guarantees and the same trade-offs as Writer.
type CandleWriter struct {
	w *writer[Candle]
}

// NewCandleWriter returns a candle writer writing to dir.
func NewCandleWriter(dir string, flushEvery int) *CandleWriter {
	opts := Options{
		Root:          dir,
		MaxRows:       flushEvery,
		FlushInterval: DefaultFlushInterval,
	}
	return &CandleWriter{w: newWriter(opts, DatasetCandles, candlePartition)}
}

// NewCandleWriterFromConfig builds a candle writer from the frozen storage
// config (docs/contracts.md §8). Rows land under <data_dir>/candles,
// partitioned by symbol and UTC date.
func NewCandleWriterFromConfig(cfg config.StorageConfig) (*CandleWriter, error) {
	opts, err := optionsFromConfig(cfg, DatasetCandles)
	if err != nil {
		return nil, err
	}
	return &CandleWriter{w: newWriter(opts, DatasetCandles, candlePartition)}, nil
}

// Append buffers a candle, flushing the batch first if the buffer is full.
func (w *CandleWriter) Append(c Candle) error { return w.w.append(c) }

// Start launches the background flusher. Starting twice returns an error.
func (w *CandleWriter) Start(ctx context.Context) error { return w.w.start(ctx) }

// Flush writes every buffered candle to disk.
func (w *CandleWriter) Flush(ctx context.Context) error { return w.w.flush(ctx) }

// Close stops the background flusher and writes the remaining candles. It is
// idempotent and returns the error from the final flush.
func (w *CandleWriter) Close() error { return w.w.close() }

// Err returns the last error raised by a background flush, or nil.
func (w *CandleWriter) Err() error { return w.w.err() }

// Pending returns the number of candles buffered but not yet written.
func (w *CandleWriter) Pending() int { return w.w.pending() }

var (
	_ RowWriter[Tick]   = (*Writer)(nil)
	_ RowWriter[Candle] = (*CandleWriter)(nil)
	_ Reader            = (*CandleReader)(nil)
)

// tickPartition derives the hive partition of a tick row from the row itself.
func tickPartition(t Tick) Partition {
	return NewPartition(t.Symbol, time.UnixMilli(t.Timestamp))
}

// candlePartition derives the hive partition of a candle row from the row
// itself.
func candlePartition(c Candle) Partition {
	return NewPartition(c.Symbol, time.UnixMilli(c.Timestamp))
}

// optionsFromConfig maps the frozen storage config onto writer options.
func optionsFromConfig(cfg config.StorageConfig, dataset Dataset) (Options, error) {
	root, err := dataDirFor(cfg, dataset)
	if err != nil {
		return Options{}, err
	}
	if cfg.FlushIntervalSecs <= 0 {
		return Options{}, fmt.Errorf("storage: flush_interval_seconds must be positive, got %d", cfg.FlushIntervalSecs)
	}
	if cfg.FlushMaxRows <= 0 {
		return Options{}, fmt.Errorf("storage: flush_max_rows must be positive, got %d", cfg.FlushMaxRows)
	}
	if p := cfg.PartitionBy; p != "" && p != PartitionByDate {
		return Options{}, fmt.Errorf("storage: partition_by %q is not supported, only %q", p, PartitionByDate)
	}
	return Options{
		Root:          root,
		MaxRows:       cfg.FlushMaxRows,
		FlushInterval: time.Duration(cfg.FlushIntervalSecs) * time.Second,
	}, nil
}
