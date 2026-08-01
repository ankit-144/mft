// Package storage persists tick and candle data to Apache Parquet files.
package storage

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
)

// Tick is the cold-storage schema for raw broker ticks.
type Tick struct {
	InstrumentToken int64   `parquet:"instrument_token"`
	Timestamp       int64   `parquet:"timestamp"`
	LastPrice       float64 `parquet:"last_price"`
	Volume          int64   `parquet:"volume"`
}

// Writer flushes ticks to Parquet files keyed by symbol and date.
type Writer struct {
	dir        string
	rows       []Tick
	buffer     chan Tick
	flushEvery int
}

// NewWriter creates a Parquet writer that flushes to dir in batches of
// flushEvery rows. The background flusher is started when Start is called.
func NewWriter(dir string, flushEvery int) *Writer {
	return &Writer{
		dir:        dir,
		rows:       make([]Tick, 0, flushEvery),
		buffer:     make(chan Tick, 1024),
		flushEvery: flushEvery,
	}
}

// Append queues a tick for the flusher.
func (w *Writer) Append(t Tick) {
	w.buffer <- t
}

// Start launches the background flusher goroutine, which writes batches of
// ticks to disk periodically or when the buffer threshold is reached.
func (w *Writer) Start(ctx context.Context) error {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				w.flush()
				return
			case t := <-w.buffer:
				w.rows = append(w.rows, t)
				if len(w.rows) >= w.flushEvery {
					w.flush()
				}
			case <-ticker.C:
				w.flush()
			}
		}
	}()
	return nil
}

func (w *Writer) flush() {
	if len(w.rows) == 0 {
		return
	}
	rows := w.rows
	w.rows = make([]Tick, 0, w.flushEvery)

	if err := os.MkdirAll(w.dir, 0o755); err != nil {
		return
	}

	path := w.pathFor(time.Now())
	file, err := os.Create(path)
	if err != nil {
		return
	}
	defer file.Close()

	writer := parquet.NewGenericWriter[Tick](file)
	if _, err := writer.Write(rows); err != nil {
		return
	}
	writer.Close()
}

// pathFor returns the Parquet file path for a flush timestamp.
func (w *Writer) pathFor(ts time.Time) string {
	return filepath.Join(w.dir, fmt.Sprintf("ticks_%s.parquet", ts.Format("20060102-150405")))
}
