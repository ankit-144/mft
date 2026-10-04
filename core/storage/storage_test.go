package storage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/parquet-go/parquet-go"
)

const testSymbol = "RELIANCE"

// day is a fixed UTC session day; partition assertions must not depend on when
// the suite runs.
var testDay = time.Date(2026, 9, 29, 0, 0, 0, 0, time.UTC)

// candleAt builds one minute bar n minutes into testDay.
func candleAt(n int) contracts.Candle {
	ts := testDay.Add(time.Duration(n) * time.Minute)
	price := 100 + float64(n)
	return contracts.Candle{
		Symbol:    testSymbol,
		Timestamp: ts,
		Open:      price,
		High:      price + 1,
		Low:       price - 1,
		Close:     price + 0.5,
		Volume:    int64(100 + n),
	}
}

// partFiles returns the published part files under a hive partition.
func partFiles(t *testing.T, root string, p Partition) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(p.Dir(root), partGlob))
	if err != nil {
		t.Fatalf("glob %s: %v", p, err)
	}
	return matches
}

// TestHiveLayoutMatchesContract asserts the exact path from docs/contracts.md
// §3: data/candles/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<ts>.parquet.
func TestHiveLayoutMatchesContract(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)

	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := filepath.Join(root, "symbol="+testSymbol, "date=2026-09-29")
	got := partFiles(t, root, NewPartition(testSymbol, testDay))
	if len(got) != 1 {
		t.Fatalf("expected 1 part file under %s, got %v", want, got)
	}

	base := filepath.Base(got[0])
	if !strings.HasPrefix(base, "part-") || !strings.HasSuffix(base, ".parquet") {
		t.Errorf("part file name %q does not match part-<timestamp>.parquet", base)
	}
	if filepath.Dir(got[0]) != want {
		t.Errorf("partition dir = %s, want %s", filepath.Dir(got[0]), want)
	}
}

// TestPartitionComesFromRow asserts the partition key is taken from the row,
// not from a configured instrument list: a symbol nobody configured still
// lands under its own directory.
func TestPartitionComesFromRow(t *testing.T) {
	root := filepath.Join(t.TempDir(), "ticks")
	w := NewWriter(root, 50)

	rows := []Tick{
		NewTick(contracts.Tick{Symbol: "TCS", Price: 10, Timestamp: testDay, Volume: 1}),
		NewTick(contracts.Tick{Symbol: "RELIANCE", Price: 20, Timestamp: testDay, Volume: 2}),
		NewTick(contracts.Tick{Symbol: "TCS", Price: 11, Timestamp: testDay, Volume: 3}),
		NewTick(contracts.Tick{Symbol: "TCS", Price: 12, Timestamp: testDay.AddDate(0, 0, 1), Volume: 4}),
	}
	for _, r := range rows {
		if err := w.Append(r); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	cases := []struct {
		partition Partition
		wantRows  int
	}{
		{NewPartition("TCS", testDay), 2},
		{NewPartition("RELIANCE", testDay), 1},
		{NewPartition("TCS", testDay.AddDate(0, 0, 1)), 1},
	}
	for _, tc := range cases {
		files := partFiles(t, root, tc.partition)
		if len(files) == 0 {
			t.Errorf("no part file for %s", tc.partition)
			continue
		}
		got := readTicks(t, files)
		if len(got) != tc.wantRows {
			t.Errorf("%s: read %d rows, want %d", tc.partition, len(got), tc.wantRows)
		}
	}
}

// TestCandleRoundTrip writes N candles and reads them back in order.
func TestCandleRoundTrip(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)

	const n = 25
	want := make([]contracts.Candle, 0, n)
	for i := 0; i < n; i++ {
		c := candleAt(i)
		want = append(want, c)
		if err := w.Append(NewCandle(c)); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 0)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("read %d candles, want %d", len(got), len(want))
	}
	for i := range want {
		if !got[i].Timestamp.Equal(want[i].Timestamp) {
			t.Errorf("candle %d: timestamp = %s, want %s (ordering broken)",
				i, got[i].Timestamp, want[i].Timestamp)
		}
		if got[i] != want[i] {
			t.Errorf("candle %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestCandleRoundTripMultipleFlushes writes more rows than flush_max_rows so
// the buffer boundary forces several flushes, and asserts the union of the
// immutable files reads back complete and ordered.
func TestCandleRoundTripMultipleFlushes(t *testing.T) {
	dataDir := t.TempDir()
	// data_dir is the parent; the candles dataset lives under it.
	root := filepath.Join(dataDir, string(DatasetCandles))

	const maxRows = 7
	const n = 40 // 5 full buffers plus a remainder

	cfg := config.StorageConfig{
		DataDir:           dataDir,
		FlushIntervalSecs: 300,
		FlushMaxRows:      maxRows,
		PartitionBy:       PartitionByDate,
	}
	w, err := NewCandleWriterFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewCandleWriterFromConfig: %v", err)
	}
	if got := w.Pending(); got != 0 {
		t.Fatalf("Pending() = %d before append, want 0", got)
	}

	// Written across four consecutive days so the flushes land in four
	// distinct partitions.
	want := make([]contracts.Candle, 0, n)
	for i := 0; i < n; i++ {
		c := candleAt(i)
		c.Timestamp = c.Timestamp.AddDate(0, 0, i/10)
		want = append(want, c)
		if err := w.Append(NewCandle(c)); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	dayFiles := partFiles(t, root, NewPartition(testSymbol, testDay))
	if len(dayFiles) < 2 {
		t.Fatalf("expected several part files on day 1 after %d rows with flush_max_rows=%d, got %d",
			n, maxRows, len(dayFiles))
	}
	// Every flush publishes a new immutable file, so the rows on disk must
	// span more than one full buffer — a single reused file would mean
	// mutation. Days 2..4 hold 10 rows each and 10 > flush_max_rows, so the
	// sum below can only reach 40 if each boundary really did spill.
	rows := 0
	for day := 0; day < 4; day++ {
		ts := testDay.AddDate(0, 0, day)
		for _, f := range partFiles(t, root, NewPartition(testSymbol, ts)) {
			got, err := readCandleFile(f)
			if err != nil {
				t.Fatalf("read %s: %v", f, err)
			}
			rows += len(got)
		}
	}
	if rows != n {
		t.Errorf("read %d rows across the dataset, want %d", rows, n)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 0)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}
	if len(got) != n {
		t.Fatalf("read %d candles, want %d", len(got), n)
	}
	for i := range want {
		if !got[i].Timestamp.Equal(want[i].Timestamp) {
			t.Fatalf("candle %d: %s is out of order, want %s", i, got[i].Timestamp, want[i].Timestamp)
		}
		if got[i] != want[i] {
			t.Fatalf("candle %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestClosedFilesAreNeverMutated asserts a second flush neither rewrites nor
// truncates an already published file. It hashes each part file after the first
// flush and re-checks after the second.
func TestClosedFilesAreNeverMutated(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)

	for i := 0; i < 3; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush 1: %v", err)
	}

	partition := NewPartition(testSymbol, testDay)
	first := partFiles(t, root, partition)
	if len(first) != 1 {
		t.Fatalf("expected 1 part file, got %d", len(first))
	}
	before := snapshot(t, first)

	for i := 3; i < 6; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush 2: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	after := snapshot(t, first)
	for name, sum := range before {
		if after[name] != sum {
			t.Errorf("closed file %s was mutated: hash %s != %s", name, after[name], sum)
		}
	}
	if len(partFiles(t, root, partition)) != 2 {
		t.Errorf("expected the second flush to publish a new file, not overwrite the first")
	}
}

// TestFlushSurfacesUnwritableDirectory is the regression test for the silent
// data-loss path: an unwritable data dir must return a wrapped error, not nil.
func TestFlushSurfacesUnwritableDirectory(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "candles")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	w := NewCandleWriter(blocker, 100)
	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append: %v", err)
	}

	flushErr := w.Flush(context.Background())
	if flushErr == nil {
		t.Fatal("Flush returned nil, want an error: os.MkdirAll failure was swallowed")
	}
	if !strings.Contains(flushErr.Error(), "candles") {
		t.Errorf("error %q does not name the path it failed on", flushErr)
	}
	var pathErr *fs.PathError
	if !errors.As(flushErr, &pathErr) {
		t.Errorf("error %q does not wrap the underlying *fs.PathError", flushErr)
	}

	// The rows must still be buffered, not dropped on the floor.
	if got := w.Pending(); got != 1 {
		t.Errorf("Pending() = %d after a failed flush, want 1 (row was lost)", got)
	}
	if w.Err() == nil {
		t.Error("Err() = nil after a failed flush, want the recorded error")
	}
}

// TestThresholdFlushReportsAppendError asserts the flush triggered by a full
// buffer surfaces through Append, not only through Err.
func TestThresholdFlushReportsAppendError(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "candles")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	w := NewCandleWriter(blocker, 2)
	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append 1: %v", err)
	}
	err := w.Append(NewCandle(candleAt(1)))
	if err == nil {
		t.Fatal("Append that crosses flush_max_rows returned nil, want the flush error")
	}
	if !strings.Contains(err.Error(), "encode") && !strings.Contains(err.Error(), "partition") {
		t.Logf("unexpected error text: %v", err)
	}
}

func TestAppendWithStatusDistinguishesRetainedFlushFailure(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "candles")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	w := NewCandleWriter(blocker, 2)
	if accepted, err := w.AppendWithStatus(NewCandle(candleAt(0))); !accepted || err != nil {
		t.Fatalf("first append = %v, %v; want accepted without flush", accepted, err)
	}
	accepted, err := w.AppendWithStatus(NewCandle(candleAt(1)))
	if !accepted || err == nil {
		t.Fatalf("threshold append = %v, %v; want accepted row and flush error", accepted, err)
	}
	if got := w.Pending(); got != 2 {
		t.Fatalf("pending = %d, want both accepted rows retained", got)
	}
}

func TestTickWriterCapacityBoundsRetainedRowsAndRecoversOnce(t *testing.T) {
	root := filepath.Join(t.TempDir(), "ticks")
	w := NewWriter(root, 2)
	flushErr := errors.New("injected flush failure")
	w.w.encode = func(*os.File, []Tick) error { return flushErr }

	rows := []Tick{{Symbol: "RELIANCE", Timestamp: 1}, {Symbol: "RELIANCE", Timestamp: 2}, {Symbol: "RELIANCE", Timestamp: 3}}
	for i := 0; i < 2; i++ {
		accepted, err := w.AppendWithStatus(rows[i])
		if i == 0 && (!accepted || err != nil) {
			t.Fatalf("append %d = %v, %v; want accepted without flush", i, accepted, err)
		}
		if i == 1 && (!accepted || !errors.Is(err, flushErr)) {
			t.Fatalf("threshold append = %v, %v; want accepted with flush error", accepted, err)
		}
	}
	for i := 0; i < 20; i++ {
		accepted, err := w.AppendWithStatus(rows[2])
		if accepted || !errors.Is(err, ErrCapacity) || !errors.Is(err, flushErr) {
			t.Fatalf("capacity append %d = %v, %v; want rejected capacity wrapping flush failure", i, accepted, err)
		}
		if got := w.Pending(); got != 2 {
			t.Fatalf("pending rows after failed capacity attempt = %d, want 2", got)
		}
	}

	w.w.encode = parquetEncode[Tick]
	accepted, err := w.AppendWithStatus(rows[2])
	if !accepted || err != nil {
		t.Fatalf("append after recovery = %v, %v; want accepted", accepted, err)
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("flush after recovery: %v", err)
	}
	files, err := filepath.Glob(filepath.Join(root, "symbol=RELIANCE", "date=1970-01-01", "part-*.parquet"))
	if err != nil {
		t.Fatalf("glob tick files: %v", err)
	}
	got := readTicks(t, files)
	if len(got) != len(rows) {
		t.Fatalf("persisted rows = %v, want %d rows exactly once", got, len(rows))
	}
	for i := range rows {
		if got[i] != rows[i] {
			t.Fatalf("persisted row %d = %+v, want %+v", i, got[i], rows[i])
		}
	}
}

func TestPublishedRowsAreNotReplayedAfterDirectorySyncFailure(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 1)
	w.w.syncPartitionDir = func(string, Partition) error { return errors.New("simulated directory sync failure") }
	accepted, err := w.AppendWithStatus(NewCandle(candleAt(0)))
	if !accepted || err == nil {
		t.Fatalf("append = %v, %v; want published row with sync error", accepted, err)
	}
	if got := w.Pending(); got != 0 {
		t.Fatalf("pending = %d, want 0 for an already published row", got)
	}
	if flushErr := w.Flush(context.Background()); flushErr == nil {
		t.Fatal("Flush lost the unresolved directory sync error")
	}
	files := partFiles(t, root, NewPartition(testSymbol, testDay))
	if len(files) != 1 {
		t.Fatalf("published files = %v, want one file and no replay", files)
	}
}

// TestFailedFlushRetainsRowsAcrossRetry asserts no row is lost or duplicated
// when a flush fails partway and then succeeds.
func TestFailedFlushRetainsRowsAcrossRetry(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)

	// A file where the RELIANCE date directory needs to be makes that one
	// partition fail while TCS succeeds.
	blocked := filepath.Join(root, "symbol="+testSymbol)
	if err := os.MkdirAll(filepath.Dir(blocked), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	if err := os.WriteFile(blocked, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	// Append TCS first: the flush must succeed for that partition and fail on
	// RELIANCE, so the retry path is exercised with a partial write.
	tcs := candleAt(0)
	tcs.Symbol = "TCS"
	if err := w.Append(NewCandle(tcs)); err != nil {
		t.Fatalf("Append TCS: %v", err)
	}
	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append RELIANCE: %v", err)
	}

	if err := w.Flush(context.Background()); err == nil {
		t.Fatal("Flush = nil, want an error for the blocked partition")
	}
	if got := w.Pending(); got != 1 {
		t.Errorf("Pending() = %d, want 1: only the unwritten RELIANCE row should be retained", got)
	}
	tcsFiles := partFiles(t, root, NewPartition("TCS", testDay))
	if len(tcsFiles) != 1 {
		t.Errorf("TCS partition: %d files, want 1 published before the failure", len(tcsFiles))
	}

	// Unblock and retry: the retained row must land exactly once.
	if err := os.Remove(blocked); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush after unblock: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 0)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("read %d RELIANCE candles, want exactly 1 (retry duplicated or dropped the row)", len(got))
	}
	// The already-published TCS row must not be re-written by the retry.
	tcsGot, err := r.Candles(context.Background(), "TCS", 0)
	if err != nil {
		t.Fatalf("Candles(TCS): %v", err)
	}
	if len(tcsGot) != 1 {
		t.Errorf("read %d TCS candles, want 1 (retry rewrote a closed file)", len(tcsGot))
	}
}

// TestCandlesLimitReturnsTail asserts limit keeps the most recent bars, which
// is what the inference loop pulls at every minute close.
func TestCandlesLimitReturnsTail(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)
	for i := 0; i < 20; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 5)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("read %d candles, want 5", len(got))
	}
	if !got[0].Timestamp.Equal(candleAt(15).Timestamp) {
		t.Errorf("tail starts at %s, want %s", got[0].Timestamp, candleAt(15).Timestamp)
	}
	if !got[4].Timestamp.Equal(candleAt(19).Timestamp) {
		t.Errorf("tail ends at %s, want %s", got[4].Timestamp, candleAt(19).Timestamp)
	}
}

func TestCandleReaderTailDoesNotDecodeOlderFilesAfterLimit(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)
	for _, minute := range []int{0, 1440} {
		if err := w.Append(NewCandle(candleAt(minute))); err != nil {
			t.Fatal(err)
		}
		if err := w.Flush(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	files := partFiles(t, root, NewPartition(testSymbol, testDay))
	if len(files) != 1 {
		t.Fatalf("old partition files = %v, want 1", files)
	}
	if err := os.WriteFile(files[0], []byte("corrupt old file"), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 1)
	if err != nil || len(got) != 1 || !got[0].Timestamp.Equal(candleAt(1440).Timestamp) {
		t.Fatalf("tail = %+v, %v; want latest candle without reading older file", got, err)
	}
}

func TestCandleTailUsesTimestampsInsteadOfFileNames(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 1)
	newer := candleAt(2)
	newer.Close = 202
	older := candleAt(1)
	older.Close = 101
	for _, candle := range []contracts.Candle{newer, older} {
		if err := w.Append(NewCandle(candle)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	r, _ := NewCandleReader(root)
	got, err := r.Candles(context.Background(), testSymbol, 1)
	if err != nil || len(got) != 1 || !got[0].Timestamp.Equal(newer.Timestamp) {
		t.Fatalf("tail = %+v, %v; want timestamp %s despite the later filename", got, err, newer.Timestamp)
	}
}

func TestCandleTailReadsWholePartitionAndUsesLastDuplicateRow(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)
	rows := []contracts.Candle{candleAt(2), candleAt(0), candleAt(2), candleAt(1)}
	rows[2].Close = 999
	for _, candle := range rows {
		if err := w.Append(NewCandle(candle)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	r, _ := NewCandleReader(root)
	got, err := r.Candles(context.Background(), testSymbol, 2)
	if err != nil || len(got) != 2 {
		t.Fatalf("tail = %+v, %v; want 2 rows", got, err)
	}
	if !got[0].Timestamp.Equal(candleAt(1).Timestamp) || !got[1].Timestamp.Equal(candleAt(2).Timestamp) || got[1].Close != 999 {
		t.Fatalf("tail = %+v; want latest timestamps and last duplicate row", got)
	}
}

func BenchmarkCandleReaderTail(b *testing.B) {
	root := filepath.Join(b.TempDir(), "candles")
	w := NewCandleWriter(root, 1000)
	for i := 0; i < 12000; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			b.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		b.Fatal(err)
	}
	r, err := NewCandleReader(root)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Candles(context.Background(), testSymbol, 100); err != nil {
			b.Fatal(err)
		}
	}
}

// TestCandleRange asserts the backtesting range query is half-open, ordered,
// and capped.
func TestCandleRange(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)
	for i := 0; i < 10; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	ctx := context.Background()

	start, end := candleAt(3).Timestamp, candleAt(7).Timestamp
	got, err := r.CandleRange(ctx, testSymbol, start, end, 0)
	if err != nil {
		t.Fatalf("CandleRange: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("range [%s,%s) returned %d candles, want 4", start, end, len(got))
	}
	if !got[0].Timestamp.Equal(start) {
		t.Errorf("range starts at %s, want %s (inclusive)", got[0].Timestamp, start)
	}
	if !got[len(got)-1].Timestamp.Before(end) {
		t.Errorf("range ends at %s, want before %s (exclusive)", got[len(got)-1].Timestamp, end)
	}

	capped, err := r.CandleRange(ctx, testSymbol, start, end, 2)
	if err != nil {
		t.Fatalf("CandleRange capped: %v", err)
	}
	if len(capped) != 2 {
		t.Errorf("capped range returned %d candles, want 2", len(capped))
	}

	if _, err := r.CandleRange(ctx, testSymbol, end, start, 0); err == nil {
		t.Error("CandleRange with start after end returned nil error")
	}
}

// TestReaderMissingSymbolIsEmpty asserts an unknown symbol is an empty result,
// not an error: inference pulls a symbol that may not have traded yet.
func TestReaderMissingSymbolIsEmpty(t *testing.T) {
	r, err := NewCandleReader(filepath.Join(t.TempDir(), "candles"))
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), "NOBODY", 10)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d candles for an unknown symbol, want 0", len(got))
	}
}

// TestConfigValidation asserts bad storage settings are rejected instead of
// silently defaulted.
//
// An empty data_dir is deliberately *not* in this table: core/config.Validate
// defaults it to "data", and the writer mirrors that, so a config that reached
// it unvalidated still writes to the documented location.
func TestConfigValidation(t *testing.T) {
	cases := []struct {
		name string
		cfg  config.StorageConfig
	}{
		{"zero interval", config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: 0, FlushMaxRows: 10}},
		{"negative interval", config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: -1, FlushMaxRows: 10}},
		{"zero max rows", config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: 60, FlushMaxRows: 0}},
		{"negative max rows", config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: 60, FlushMaxRows: -1}},
		{"unsupported partition", config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: 60, FlushMaxRows: 10, PartitionBy: "hour"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewCandleWriterFromConfig(tc.cfg); err == nil {
				t.Error("NewCandleWriterFromConfig = nil error, want a validation failure")
			}
		})
	}
}

// TestEmptyDataDirMirrorsConfigDefault asserts an un-defaulted config still
// resolves to the documented data/ location rather than a relative "./candles".
func TestEmptyDataDirMirrorsConfigDefault(t *testing.T) {
	w, err := NewCandleWriterFromConfig(config.StorageConfig{FlushIntervalSecs: 60, FlushMaxRows: 10})
	if err != nil {
		t.Fatalf("NewCandleWriterFromConfig: %v", err)
	}
	if got, want := w.w.root, filepath.Join("data", string(DatasetCandles)); got != want {
		t.Errorf("root = %q, want %q", got, want)
	}
}

// TestConfigWiring asserts data_dir and the dataset name decide the root.
func TestConfigWiring(t *testing.T) {
	dataDir := t.TempDir()
	cfg := config.StorageConfig{
		DataDir:           dataDir,
		FlushIntervalSecs: 30,
		FlushMaxRows:      10,
		PartitionBy:       PartitionByDate,
	}

	candles, err := NewCandleWriterFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewCandleWriterFromConfig: %v", err)
	}
	ticks, err := NewWriterFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewWriterFromConfig: %v", err)
	}
	reader, err := NewCandleReaderFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewCandleReaderFromConfig: %v", err)
	}

	wantCandles := filepath.Join(dataDir, "candles")
	wantTicks := filepath.Join(dataDir, "ticks")
	if candles.w.root != wantCandles {
		t.Errorf("candle root = %s, want %s", candles.w.root, wantCandles)
	}
	if ticks.w.root != wantTicks {
		t.Errorf("tick root = %s, want %s", ticks.w.root, wantTicks)
	}
	if reader.Root() != wantCandles {
		t.Errorf("reader root = %s, want %s", reader.Root(), wantCandles)
	}
}

// TestBackgroundFlusherWritesOnContextCancel asserts the goroutine started by
// Start writes the buffer when its context ends, and that Close is idempotent.
func TestBackgroundFlusherWritesOnContextCancel(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 1000)

	ctx, cancel := context.WithCancel(context.Background())
	if err := w.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append: %v", err)
	}

	cancel()

	// Close waits for the flusher goroutine, so the file must exist after it.
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	if got := partFiles(t, root, NewPartition(testSymbol, testDay)); len(got) != 1 {
		t.Fatalf("expected 1 part file after shutdown flush, got %d", len(got))
	}
	if err := w.Append(NewCandle(candleAt(1))); !errors.Is(err, ErrClosed) {
		t.Errorf("Append after Close = %v, want ErrClosed", err)
	}
	if err := w.Flush(context.Background()); !errors.Is(err, ErrClosed) {
		t.Errorf("Flush after Close = %v, want ErrClosed", err)
	}
}

// TestFlushHonoursCancelledContext asserts a cancelled context aborts the
// flush instead of writing anyway, and the rows stay buffered.
func TestFlushHonoursCancelledContext(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)
	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := w.Flush(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("Flush(cancelled) = %v, want context.Canceled", err)
	}
	if got := w.Pending(); got != 1 {
		t.Errorf("Pending() = %d after a cancelled flush, want 1", got)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := partFiles(t, root, NewPartition(testSymbol, testDay)); len(got) != 1 {
		t.Errorf("Close did not write the retained row: %d part files", len(got))
	}
}

// TestConcurrentAppendAndFlush runs appends, explicit flushes and Close
// against one writer. The mutex must make this safe and must never interleave
// two flushes — running with -race is what actually proves it.
func TestConcurrentAppendAndFlush(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 16)

	const writers = 4
	const perWriter = 40

	var wg sync.WaitGroup
	for g := 0; g < writers; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				c := candleAt(i)
				c.Symbol = fmt.Sprintf("SYM%d", g)
				if err := w.Append(NewCandle(c)); err != nil {
					t.Errorf("Append: %v", err)
					return
				}
			}
		}(g)
	}

	flushers := 3
	for f := 0; f < flushers; f++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				if err := w.Flush(context.Background()); err != nil {
					t.Errorf("Flush: %v", err)
					return
				}
			}
		}()
	}

	wg.Wait()
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	for g := 0; g < writers; g++ {
		symbol := fmt.Sprintf("SYM%d", g)
		got, err := r.Candles(context.Background(), symbol, 0)
		if err != nil {
			t.Fatalf("Candles(%s): %v", symbol, err)
		}
		if len(got) != perWriter {
			t.Errorf("%s: read %d candles, want %d", symbol, len(got), perWriter)
		}
		for i := 1; i < len(got); i++ {
			if !got[i].Timestamp.After(got[i-1].Timestamp) {
				t.Errorf("%s: candles out of order at %d", symbol, i)
				break
			}
		}
	}
}

// TestEncodeFailureIsSurfacedAndTempRemoved asserts a failure inside the
// parquet encoder — the path where writer.Write or writer.Close returns an
// error — is wrapped, propagated to the caller, leaves no in-flight file, and
// keeps the rows buffered. Without this, a corrupt batch would vanish with a
// nil error from flush().
func TestEncodeFailureIsSurfacedAndTempRemoved(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)

	sentinel := errors.New("disk gremlin")
	w.w.encode = func(*os.File, []Candle) error {
		return fmt.Errorf("parquet writer close: %w", sentinel)
	}

	if err := w.Append(NewCandle(candleAt(0))); err != nil {
		t.Fatalf("Append: %v", err)
	}

	flushErr := w.Flush(context.Background())
	if flushErr == nil {
		t.Fatal("Flush = nil, want the encoder error surfaced")
	}
	if !errors.Is(flushErr, sentinel) {
		t.Errorf("Flush error %v does not wrap the encoder error", flushErr)
	}
	if !errors.Is(w.Err(), sentinel) {
		t.Errorf("Err() = %v, want the encoder error recorded", w.Err())
	}
	if got := w.Pending(); got != 1 {
		t.Errorf("Pending() = %d, want 1 (row lost on an encode failure)", got)
	}

	// Nothing may be published, and no in-flight file may be left behind for a
	// reader to trip over.
	partition := NewPartition(testSymbol, testDay)
	if got := partFiles(t, root, partition); len(got) != 0 {
		t.Errorf("a failed encode published %d part files, want 0", len(got))
	}
	entries, err := os.ReadDir(partition.Dir(root))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		t.Errorf("failed encode left %s behind", e.Name())
	}

	// A healthy encoder on retry must land the retained row exactly once.
	w.w.encode = parquetEncode[Candle]
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush after fixing the encoder: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := partFiles(t, root, partition); len(got) != 1 {
		t.Fatalf("retry published %d part files, want 1", len(got))
	}
	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 0)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}
	if len(got) != 1 || got[0] != candleAt(0) {
		t.Errorf("read %+v, want exactly the one retained bar", got)
	}
}

// TestTickRoundTripAndInventory writes ticks for two symbols across two days
// and asserts the row round-trips byte-for-byte, that the dataset inventory
// (Symbols, Glob) agrees with what is on disk, and that Err/Pending report the
// writer's state.
func TestTickRoundTripAndInventory(t *testing.T) {
	root := filepath.Join(t.TempDir(), "ticks")
	w := NewWriter(root, 100)

	if got := w.Pending(); got != 0 {
		t.Fatalf("Pending() = %d on a fresh writer, want 0", got)
	}
	if err := w.Err(); err != nil {
		t.Errorf("Err() = %v on a fresh writer, want nil", err)
	}

	next := testDay.Add(48 * time.Hour)
	want := []contracts.Tick{
		{Symbol: "TCS", Token: 11536, Price: 3120.25, Volume: 7, Timestamp: testDay},
		{Symbol: testSymbol, Token: 256265, Price: 2934.5, Volume: 11, Timestamp: testDay.Add(90 * time.Second)},
		{Symbol: "TCS", Token: 11536, Price: 3121.75, Volume: 3, Timestamp: next},
	}
	for _, tick := range want {
		if err := w.Append(NewTick(tick)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if got := w.Pending(); got != len(want) {
		t.Errorf("Pending() = %d, want %d", got, len(want))
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if got := w.Pending(); got != 0 {
		t.Errorf("Pending() = %d after a flush, want 0", got)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	dayOne := partFiles(t, root, NewPartition(testSymbol, testDay))
	if len(dayOne) != 1 {
		t.Fatalf("RELIANCE day 1 has %d part files, want 1", len(dayOne))
	}
	rows, err := readTickFile(dayOne[0])
	if err != nil {
		t.Fatalf("readTickFile: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("RELIANCE file holds %d rows, want 1", len(rows))
	}
	got := rows[0].Contract()
	if !got.Timestamp.Equal(want[1].Timestamp.UTC()) {
		t.Errorf("timestamp = %s, want %s", got.Timestamp, want[1].Timestamp.UTC())
	}
	if got.Token != want[1].Token || got.Price != want[1].Price || got.Volume != want[1].Volume {
		t.Errorf("round trip = %+v, want %+v", got, want[1])
	}

	// The reader's inventory must agree with what the writer put on disk.
	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	symbols, err := r.Symbols(context.Background())
	if err != nil {
		t.Fatalf("Symbols: %v", err)
	}
	if len(symbols) != 2 || symbols[0] != "RELIANCE" || symbols[1] != "TCS" {
		t.Errorf("symbols = %v, want [RELIANCE TCS]", symbols)
	}
	if got, want := r.Glob(testSymbol),
		filepath.Join(SymbolDir(root, testSymbol), "date=*", partGlob); got != want {
		t.Errorf("Glob = %s, want %s", got, want)
	}
}

// TestReaderIgnoresInflightFiles asserts a reader glob cannot pick up a file
// that is still being written. This is the property that makes the
// read-while-writing guarantee hold: the writer's temporary name is not a
// "part-" name until the batch is complete.
func TestReaderIgnoresInflightFiles(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 100)
	for i := 0; i < 3; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Drop a truncated file in the partition exactly as a mid-write batch
	// would look if the writer did not use a temporary name.
	partition := NewPartition(testSymbol, testDay)
	junk := filepath.Join(partition.Dir(root), inflightPrefix+"truncated.parquet")
	if err := os.WriteFile(junk, []byte("PAR1 not really a parquet file"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}
	got, err := r.Candles(context.Background(), testSymbol, 0)
	if err != nil {
		t.Fatalf("Candles: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("read %d candles, want 3: the in-flight file was not ignored", len(got))
	}
}

// TestStartTwiceIsAnError asserts a second Start is reported rather than
// silently spawning a second flusher that would race the first for the buffer.
func TestStartTwiceIsAnError(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 1000)

	if err := w.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := w.Start(context.Background()); err == nil {
		t.Error("second Start = nil error, want a failure")
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Closed writers refuse to restart.
	if err := w.Start(context.Background()); !errors.Is(err, ErrClosed) {
		t.Errorf("Start after Close = %v, want ErrClosed", err)
	}
}

// TestReadWhileWriting is the test for the reason the whole package is built
// this way: inference reads history while ingestion is still appending to the
// same tree (docs/contracts.md §3, "DuckDB rules"). A reader running against a
// live writer must never error and never observe a partial batch.
func TestReadWhileWriting(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 8)

	const total = 200
	done := make(chan struct{})

	go func() {
		defer close(done)
		for i := 0; i < total; i++ {
			if err := w.Append(NewCandle(candleAt(i))); err != nil {
				t.Errorf("Append: %v", err)
				return
			}
		}
	}()

	r, err := NewCandleReader(root)
	if err != nil {
		t.Fatalf("NewCandleReader: %v", err)
	}

	reads := 0
	for {
		select {
		case <-done:
			if err := w.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			if reads == 0 {
				t.Fatal("the reader never completed a query against the live writer")
			}
			final, err := r.Candles(context.Background(), testSymbol, 0)
			if err != nil {
				t.Fatalf("Candles after Close: %v", err)
			}
			if len(final) != total {
				t.Errorf("read %d candles after Close, want %d", len(final), total)
			}
			for i, c := range final {
				if !c.Timestamp.Equal(candleAt(i).Timestamp) {
					t.Fatalf("candle %d: %s is out of order, want %s", i, c.Timestamp, candleAt(i).Timestamp)
				}
			}
			return
		default:
		}

		got, err := r.Candles(context.Background(), testSymbol, 16)
		if err != nil {
			t.Fatalf("Candles during write: %v", err)
		}
		reads++
		// Whatever the reader saw must be an ordered, fully intact run of
		// bars. A partially written file would surface as a hole, a
		// mis-ordered row, or a value that does not match its minute.
		for i, c := range got {
			if c.Symbol != testSymbol {
				t.Errorf("read a row for %s mid-write, want %s", c.Symbol, testSymbol)
				break
			}
			if i > 0 && !c.Timestamp.After(got[i-1].Timestamp) {
				t.Errorf("read out-of-order candles at index %d: %s", i, c.Timestamp)
				break
			}
			minute := int(c.Timestamp.Sub(testDay) / time.Minute)
			if want := NewCandle(candleAt(minute)).Contract(); c != want {
				t.Errorf("mid-write read got %+v for minute %d, want %+v", c, minute, want)
				break
			}
		}
	}
}

// TestHistoricalRangeLayoutMatchesContract asserts the backfill dataset path
// from docs/contracts.md §3:
// data/historical/symbol=<SYMBOL>/from=<date>/to=<date>/candles.parquet.
func TestHistoricalRangeLayoutMatchesContract(t *testing.T) {
	from := testDay
	to := testDay.AddDate(0, 0, 7)
	p := NewRangePartition(testSymbol, from, to)

	wantDir := filepath.Join("data", "historical",
		"symbol="+testSymbol, "from=2026-09-29", "to=2026-10-06")
	if got := p.Dir("data/historical"); got != wantDir {
		t.Errorf("range dir = %s, want %s", got, wantDir)
	}
	if got, want := p.File("data/historical"),
		filepath.Join(wantDir, "candles.parquet"); got != want {
		t.Errorf("range file = %s, want %s", got, want)
	}
	if got, want := p.String(), "symbol="+testSymbol+"/from=2026-09-29/to=2026-10-06"; got != want {
		t.Errorf("range string = %s, want %s", got, want)
	}
}

// TestNoInflightFilesSurvive asserts a successful flush leaves only published
// part files, so a reader glob can never pick up a partial file.
func TestNoInflightFilesSurvive(t *testing.T) {
	root := filepath.Join(t.TempDir(), "candles")
	w := NewCandleWriter(root, 3)
	for i := 0; i < 12; i++ {
		if err := w.Append(NewCandle(candleAt(i))); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.Contains(d.Name(), inflightPrefix) {
			t.Errorf("in-flight file survived: %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WalkDir: %v", err)
	}
}

// TestHiveValueEscaping asserts symbols that would break a "key=value" path
// are escaped rather than silently relocating the dataset.
func TestHiveValueEscaping(t *testing.T) {
	cases := []string{"RELIANCE", "NIFTY50", "A/B", "X=Y", "SP 500", ""}
	for _, symbol := range cases {
		escaped := filepath.Join("data", "symbol="+escapeHiveValue(symbol))
		if strings.Count(escaped, string(filepath.Separator)) != 1 {
			t.Errorf("symbol %q escaped to %s, which adds a path separator", symbol, escaped)
		}
		if got := unescapeHiveValue(escapeHiveValue(symbol)); got != symbol && symbol != "" {
			t.Errorf("round trip of %q gave %q", symbol, got)
		}
		// The reader must find what the writer wrote.
		if symbol != "" {
			if !strings.HasPrefix(SymbolDir("data", symbol), "data") {
				t.Errorf("SymbolDir for %q is not under the dataset root", symbol)
			}
		}
	}
}

// readTicks decodes tick rows from the given files, in file order.
func readTicks(t *testing.T, files []string) []Tick {
	t.Helper()
	var out []Tick
	for _, f := range files {
		rows, err := readTickFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		out = append(out, rows...)
	}
	return out
}

// readTickFile decodes a published tick parquet file.
func readTickFile(path string) ([]Tick, error) {
	return parquet.ReadFile[Tick](path)
}

// snapshot returns a content hash per file, used to prove a closed file was
// not rewritten by a later flush.
func snapshot(t *testing.T, files []string) map[string]string {
	t.Helper()
	out := make(map[string]string, len(files))
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("ReadFile %s: %v", f, err)
		}
		sum := sha256.Sum256(data)
		out[filepath.Base(f)] = hex.EncodeToString(sum[:])
	}
	return out
}
