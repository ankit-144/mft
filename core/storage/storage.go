// Package storage writes immutable Parquet partitions and provides typed readers.
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
	DefaultFlushInterval = 5 * time.Minute

	DefaultFlushMaxRows = 10000

	DateLayout = "2006-01-02"

	PartitionByDate = "date"
)

// Dataset is the name of a Parquet dataset under the storage data directory.
type Dataset string

// The datasets the platform writes.
const (
	DatasetTicks Dataset = "ticks"

	DatasetCandles Dataset = "candles"
)

// ErrClosed is returned by Append and Flush once Close has been called.
var ErrClosed = errors.New("storage: writer is closed")

// ErrCapacity reports that a full in-memory writer buffer could not be drained, so
// the attempted row was rejected and remains the caller's responsibility.
var ErrCapacity = errors.New("storage: writer buffer capacity reached")

// RowWriter is the writer contract from docs/contracts.md §3.
type RowWriter[T any] interface {
	// Append adds one row to the current buffer.
	Append(row T) error
	// Flush writes every buffered row to disk and returns.
	Flush(ctx context.Context) error
	// Flush stops the background flusher and writes any remaining rows.
	Close() error
}

// Tick is the Parquet row for a raw broker tick.
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

// Candle is the Parquet row for an aggregated one-minute bar.
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
type Writer struct {
	w *writer[Tick]
}

// NewWriter returns a tick writer that buffers under dir and flushes every flushEvery
// rows, using the default interval.
func NewWriter(dir string, flushEvery int) *Writer {
	opts := Options{
		Root:          dir,
		MaxRows:       flushEvery,
		FlushInterval: DefaultFlushInterval,
	}
	return &Writer{w: newWriter(opts, DatasetTicks, tickPartition)}
}

// NewWriterFromConfig builds a tick writer from the shared storage config
// (docs/contracts.md §8).
func NewWriterFromConfig(cfg config.StorageConfig) (*Writer, error) {
	opts, err := optionsFromConfig(cfg, DatasetTicks)
	if err != nil {
		return nil, err
	}
	return &Writer{w: newWriter(opts, DatasetTicks, tickPartition)}, nil
}

// Append buffers a tick.
func (w *Writer) Append(t Tick) error { return w.w.append(t) }

// AppendWithStatus reports whether the tick entered the writer buffer. A false result
// means the caller still owns the row; errors.Is(err, ErrCapacity) is retryable.
func (w *Writer) AppendWithStatus(t Tick) (bool, error) {
	return w.w.appendWithStatus(t)
}

// Start launches the background flusher, which writes buffered rows every
// storage.flush_interval_seconds and once more when ctx is cancelled.
func (w *Writer) Start(ctx context.Context) error { return w.w.start(ctx) }

// Flush writes every buffered row to disk.
func (w *Writer) Flush(ctx context.Context) error { return w.w.flush(ctx) }

// Close stops the background flusher and writes the remaining rows, then releases the
// writer.
func (w *Writer) Close() error { return w.w.close() }

// Err returns the last error raised by a background flush, or nil if the most recent
// flush succeeded.
func (w *Writer) Err() error { return w.w.err() }

// Pending returns the number of rows buffered but not yet written.
func (w *Writer) Pending() int { return w.w.pending() }

// CandleWriter persists aggregated one-minute candles to the candles dataset.
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

// NewCandleWriterFromConfig builds a candle writer from the shared storage config
// (docs/contracts.md §8).
func NewCandleWriterFromConfig(cfg config.StorageConfig) (*CandleWriter, error) {
	opts, err := optionsFromConfig(cfg, DatasetCandles)
	if err != nil {
		return nil, err
	}
	return &CandleWriter{w: newWriter(opts, DatasetCandles, candlePartition)}, nil
}

// Append buffers a candle, flushing the batch first if the buffer is full.
func (w *CandleWriter) Append(c Candle) error { return w.w.append(c) }

// AppendWithStatus reports whether the candle entered the writer buffer even when its
// synchronous flush returned an error.
func (w *CandleWriter) AppendWithStatus(c Candle) (bool, error) {
	return w.w.appendWithStatus(c)
}

// Start launches the background flusher.
func (w *CandleWriter) Start(ctx context.Context) error { return w.w.start(ctx) }

// Flush writes every buffered candle to disk.
func (w *CandleWriter) Flush(ctx context.Context) error { return w.w.flush(ctx) }

// Close stops the background flusher and writes the remaining candles.
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

// candlePartition derives the hive partition of a candle row from the row itself.
func candlePartition(c Candle) Partition {
	return NewPartition(c.Symbol, time.UnixMilli(c.Timestamp))
}

// optionsFromConfig maps the shared storage config onto writer options.
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
