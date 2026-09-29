package storage

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/parquet-go/parquet-go"
)

const (
	// fileStampLayout is the timestamp embedded in "part-<timestamp>.parquet".
	// Nanosecond precision keeps distinct flushes in distinct files.
	fileStampLayout = "20060102T150405.000000000Z"

	// inflightPrefix marks a partially written file. Readers glob
	// "part-*.parquet" and therefore never see these.
	inflightPrefix = ".inflight-"

	// maxPartNameAttempts bounds the search for a free file name in a
	// partition directory.
	maxPartNameAttempts = 1024
)

// writer is the hive-partitioned Parquet core shared by Writer and
// CandleWriter.
//
// All mutable state is behind mu, and mu is held for the whole of a flush —
// from creating the directory to renaming the finished file. Two flushes
// therefore cannot interleave, Append cannot race a flush, and no file that
// has been closed is ever reopened.
type writer[T any] struct {
	root     string
	dataset  Dataset
	maxRows  int
	interval time.Duration
	part     func(T) Partition

	// encode serialises one partition's rows into an open file. It is a
	// field so a test can inject a failing encoder; production always uses
	// parquetEncode, and every error it returns is wrapped and surfaced.
	encode func(f *os.File, rows []T) error

	// mu guards rows, closed and lastErr, and serialises flushes.
	mu      sync.Mutex
	rows    []T
	closed  bool
	lastErr error

	startMu sync.Mutex
	started bool
	stop    context.CancelFunc
	wg      sync.WaitGroup
}

func newWriter[T any](opts Options, dataset Dataset, part func(T) Partition) *writer[T] {
	maxRows := opts.MaxRows
	if maxRows <= 0 {
		maxRows = DefaultFlushMaxRows
	}
	interval := opts.FlushInterval
	if interval <= 0 {
		interval = DefaultFlushInterval
	}
	return &writer[T]{
		root:     opts.Root,
		dataset:  dataset,
		maxRows:  maxRows,
		interval: interval,
		part:     part,
		encode:   parquetEncode[T],
		rows:     make([]T, 0, maxRows),
	}
}

// parquetEncode writes rows to f as a complete parquet file. Both the row
// write and the footer write are checked: a parquet file whose footer failed
// to write is unreadable, so a silent Close would poison the dataset.
func parquetEncode[T any](f *os.File, rows []T) error {
	pw := parquet.NewGenericWriter[T](f)
	if _, err := pw.Write(rows); err != nil {
		return fmt.Errorf("parquet write of %d rows: %w", len(rows), err)
	}
	if err := pw.Close(); err != nil {
		return fmt.Errorf("parquet writer close: %w", err)
	}
	return nil
}

// append buffers a row and, once the buffer is full, writes the batch before
// returning so that the caller sees the error.
func (w *writer[T]) append(row T) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrClosed
	}
	w.rows = append(w.rows, row)
	if len(w.rows) >= w.maxRows {
		return w.flushLocked(context.Background())
	}
	return nil
}

// flush writes every buffered row. A cancelled context aborts before any file
// is created; rows stay buffered.
func (w *writer[T]) flush(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrClosed
	}
	return w.flushLocked(ctx)
}

// start launches the background flusher. Starting twice is an error; starting
// after Close returns ErrClosed.
func (w *writer[T]) start(ctx context.Context) error {
	w.startMu.Lock()
	defer w.startMu.Unlock()

	// closed is guarded by mu, not startMu: reading it here without mu would
	// race a concurrent Close.
	w.mu.Lock()
	closed := w.closed
	w.mu.Unlock()
	if closed {
		return ErrClosed
	}
	if w.started {
		return fmt.Errorf("storage: %s writer already started", w.dataset)
	}
	w.started = true

	loopCtx, cancel := context.WithCancel(ctx)
	w.stop = cancel
	w.wg.Add(1)
	go w.loop(loopCtx)
	return nil
}

// close stops the background flusher and writes the remaining rows. It is
// idempotent.
func (w *writer[T]) close() error {
	w.mu.Lock()
	w.closed = true
	w.mu.Unlock()

	w.startMu.Lock()
	if w.stop != nil {
		w.stop()
	}
	w.startMu.Unlock()
	w.wg.Wait()

	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.flushLocked(context.Background()); err != nil {
		return err
	}
	return w.lastErr
}

// err returns the last flush error observed on any path.
func (w *writer[T]) err() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lastErr
}

// pending returns the number of buffered, not yet written rows.
func (w *writer[T]) pending() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.rows)
}

// loop is the background flusher. Errors are recorded rather than returned;
// Err is the only way to observe them.
func (w *writer[T]) loop(ctx context.Context) {
	defer w.wg.Done()

	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			// Shutdown must not be cancelled half-way through a batch, so
			// the final flush runs on a detached context.
			w.mu.Lock()
			err := w.flushLocked(context.WithoutCancel(ctx))
			w.mu.Unlock()
			w.record(err)
			return
		case <-ticker.C:
			w.mu.Lock()
			err := w.flushLocked(ctx)
			w.mu.Unlock()
			w.record(err)
		}
	}
}

// record stores an error raised on the flusher goroutine.
func (w *writer[T]) record(err error) {
	if err == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.lastErr = err
}

// flushLocked writes the buffer out, one immutable file per partition. The
// caller must hold w.mu.
//
// On failure the rows that were not written stay in the buffer and are retried
// by the next flush: a filesystem error never costs data silently.
func (w *writer[T]) flushLocked(ctx context.Context) error {
	if len(w.rows) == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("storage: flush %s: %w", w.dataset, err)
	}

	order, groups := w.groupByPartition()
	for i, p := range order {
		if err := ctx.Err(); err != nil {
			err = fmt.Errorf("storage: flush %s: %w", w.dataset, err)
			w.retainUnwritten(order[:i])
			w.lastErr = err
			return err
		}
		if err := w.writePartition(p, groups[p]); err != nil {
			w.retainUnwritten(order[:i])
			w.lastErr = err
			return err
		}
	}

	w.rows = w.rows[:0]
	w.lastErr = nil
	return nil
}

// groupByPartition splits the buffer into per-partition batches, preserving
// first-seen order so that flushes are deterministic.
func (w *writer[T]) groupByPartition() ([]Partition, map[Partition][]T) {
	groups := make(map[Partition][]T, 4)
	order := make([]Partition, 0, 4)
	for _, row := range w.rows {
		p := w.part(row)
		if _, seen := groups[p]; !seen {
			order = append(order, p)
		}
		groups[p] = append(groups[p], row)
	}
	return order, groups
}

// retainUnwritten drops the rows belonging to partitions that were already
// written and keeps the rest, in their original order, so that the next flush
// retries exactly what is missing — no loss, no duplicate.
func (w *writer[T]) retainUnwritten(written []Partition) {
	if len(written) == 0 {
		return
	}
	done := make(map[Partition]bool, len(written))
	for _, p := range written {
		done[p] = true
	}
	rest := make([]T, 0, len(w.rows))
	for _, row := range w.rows {
		if !done[w.part(row)] {
			rest = append(rest, row)
		}
	}
	w.rows = rest
}

// writePartition encodes one partition's rows into a new immutable file.
//
// The file is written to ".inflight-*" and renamed to "part-*" only after the
// parquet writer has been closed and the bytes are on disk, so the final name
// never refers to an incomplete file.
func (w *writer[T]) writePartition(p Partition, rows []T) error {
	dir := p.Dir(w.root)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("storage: create partition dir %s: %w", dir, err)
	}

	file, err := os.CreateTemp(dir, inflightPrefix+"*.parquet")
	if err != nil {
		return fmt.Errorf("storage: create temp parquet in %s: %w", dir, err)
	}
	temp := file.Name()

	cleanup := func(cause error) error {
		if cerr := file.Close(); cerr != nil {
			cause = errors.Join(cause, fmt.Errorf("storage: close %s: %w", temp, cerr))
		}
		if rerr := os.Remove(temp); rerr != nil && !errors.Is(rerr, fs.ErrNotExist) {
			cause = errors.Join(cause, fmt.Errorf("storage: remove %s: %w", temp, rerr))
		}
		return cause
	}

	if err := w.encode(file, rows); err != nil {
		return cleanup(fmt.Errorf("storage: %s %d rows for %s: %w", w.dataset, len(rows), p, err))
	}
	if err := file.Sync(); err != nil {
		return cleanup(fmt.Errorf("storage: sync %s for %s %s: %w", temp, w.dataset, p, err))
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("storage: close %s for %s %s: %w", temp, w.dataset, p, err)
	}

	final, err := freePartPath(dir, time.Now())
	if err != nil {
		if rerr := os.Remove(temp); rerr != nil && !errors.Is(rerr, fs.ErrNotExist) {
			return errors.Join(err, fmt.Errorf("storage: remove %s: %w", temp, rerr))
		}
		return err
	}
	if err := os.Rename(temp, final); err != nil {
		if rerr := os.Remove(temp); rerr != nil && !errors.Is(rerr, fs.ErrNotExist) {
			return errors.Join(
				fmt.Errorf("storage: publish %s: %w", final, err),
				fmt.Errorf("storage: remove %s: %w", temp, rerr),
			)
		}
		return fmt.Errorf("storage: publish %s: %w", final, err)
	}
	return syncDir(dir, p)
}

// freePartPath returns a "part-<timestamp>.parquet" path in dir that does not
// exist. Refusing to reuse a name is what guarantees a closed file is never
// written again.
func freePartPath(dir string, ts time.Time) (string, error) {
	for attempt := 0; attempt < maxPartNameAttempts; attempt++ {
		candidate := filepath.Join(dir, fmt.Sprintf("part-%s.parquet",
			ts.UTC().Add(time.Duration(attempt)).Format(fileStampLayout)))
		_, err := os.Lstat(candidate)
		switch {
		case errors.Is(err, fs.ErrNotExist):
			return candidate, nil
		case err != nil:
			return "", fmt.Errorf("storage: stat %s: %w", candidate, err)
		}
	}
	return "", fmt.Errorf("storage: no free part file name in %s after %d attempts", dir, maxPartNameAttempts)
}

// syncDir fsyncs a directory so the rename that published a part file
// survives a crash.
func syncDir(dir string, p Partition) error {
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("storage: open partition dir %s: %w", dir, err)
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("storage: sync partition dir %s for %s: %w", dir, p, err)
	}
	return nil
}
