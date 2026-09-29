package analytics

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/storage"
)

const symbol = "RELIANCE"

// day is a fixed UTC session day; assertions must not depend on when the suite
// runs.
var day = time.Date(2026, 9, 29, 0, 0, 0, 0, time.UTC)

// barAt builds the n-th minute bar of day, closing at 100*(1.01^n).
func barAt(n int) contracts.Candle {
	close := 100 * math.Pow(1.01, float64(n))
	return contracts.Candle{
		Symbol:    symbol,
		Timestamp: day.Add(time.Duration(n) * time.Minute),
		Open:      close - 0.5,
		High:      close + 1,
		Low:       close - 1,
		Close:     close,
		Volume:    int64(10 + n),
	}
}

// storeWith builds a candles dataset root holding count bars on each of the
// given days, flushing every maxRows so the reader is exercised over several
// immutable files.
func storeWith(t *testing.T, days int, barsPerDay, maxRows int) (*Store, []contracts.Candle) {
	t.Helper()

	dataDir := t.TempDir()
	cfg := config.StorageConfig{
		DataDir:           dataDir,
		FlushIntervalSecs: 300,
		FlushMaxRows:      maxRows,
		PartitionBy:       storage.PartitionByDate,
	}
	w, err := storage.NewCandleWriterFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewCandleWriterFromConfig: %v", err)
	}

	var want []contracts.Candle
	for d := 0; d < days; d++ {
		for i := 0; i < barsPerDay; i++ {
			c := barAt(i)
			c.Timestamp = c.Timestamp.AddDate(0, 0, d)
			want = append(want, c)
			if err := w.Append(storage.NewCandle(c)); err != nil {
				t.Fatalf("Append: %v", err)
			}
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s, err := NewFromConfig(cfg)
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	return s, want
}

// TestContextReturnsModelReadyTail asserts Context hands inference exactly the
// last N bars, ascending — the shape TabFM is prompted with.
func TestContextReturnsModelReadyTail(t *testing.T) {
	s, _ := storeWith(t, 1, 20, 5)
	ctx := context.Background()

	got, err := s.Context(ctx, symbol, 7)
	if err != nil {
		t.Fatalf("Context: %v", err)
	}
	if len(got) != 7 {
		t.Fatalf("Context(7) returned %d candles, want 7", len(got))
	}
	if !got[0].Timestamp.Equal(barAt(13).Timestamp) {
		t.Errorf("context starts at %s, want %s", got[0].Timestamp, barAt(13).Timestamp)
	}
	for i := 1; i < len(got); i++ {
		if !got[i].Timestamp.After(got[i-1].Timestamp) {
			t.Fatalf("context is not ascending at index %d", i)
		}
	}
	if got[len(got)-1].Close != barAt(19).Close {
		t.Errorf("context ends at close %v, want %v", got[len(got)-1].Close, barAt(19).Close)
	}
}

// TestCoverageCountsWarmup asserts Coverage reports whether the 60-bar warm-up
// the feature schema requires is satisfied.
func TestCoverageCountsWarmup(t *testing.T) {
	ctx := context.Background()

	short, want := storeWith(t, 1, 10, 5)
	cov, err := short.Coverage(ctx, symbol)
	if err != nil {
		t.Fatalf("Coverage: %v", err)
	}
	if cov.Bars != len(want) {
		t.Errorf("Bars = %d, want %d", cov.Bars, len(want))
	}
	if cov.Days != 1 {
		t.Errorf("Days = %d, want 1", cov.Days)
	}
	if cov.Complete {
		t.Errorf("Complete = true for %d bars, want false below the %d-bar warm-up", cov.Bars, WarmupBars)
	}
	if !cov.First.Equal(want[0].Timestamp) || !cov.Last.Equal(want[len(want)-1].Timestamp) {
		t.Errorf("extent = [%s,%s], want [%s,%s]", cov.First, cov.Last, want[0].Timestamp, want[len(want)-1].Timestamp)
	}

	long, longWant := storeWith(t, 3, 30, 7)
	warm, err := long.Coverage(ctx, symbol)
	if err != nil {
		t.Fatalf("Coverage: %v", err)
	}
	if warm.Bars != len(longWant) || warm.Days != 3 {
		t.Errorf("Coverage = %d bars over %d days, want %d over 3",
			warm.Bars, warm.Days, len(longWant))
	}
	if !warm.Complete {
		t.Errorf("Complete = false for %d bars, want true at or above the %d-bar warm-up",
			warm.Bars, WarmupBars)
	}

	empty, err := long.Coverage(ctx, "NOBODY")
	if err != nil {
		t.Fatalf("Coverage(unknown): %v", err)
	}
	if empty.Bars != 0 || empty.Complete {
		t.Errorf("Coverage(unknown) = %+v, want an empty, incomplete result", empty)
	}
}

// TestSummaryAggregatesRange asserts OHLC, volume, VWAP and the simple return
// over an explicitly bounded range.
func TestSummaryAggregatesRange(t *testing.T) {
	s, want := storeWith(t, 1, 30, 6)
	ctx := context.Background()

	start, end := barAt(0).Timestamp, barAt(20).Timestamp
	got, err := s.Summary(ctx, symbol, start, end)
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}

	if got.Bars != 20 {
		t.Fatalf("Bars = %d, want 20 (range is half-open)", got.Bars)
	}
	if got.Open != want[0].Open {
		t.Errorf("Open = %v, want %v", got.Open, want[0].Open)
	}
	if got.Close != want[19].Close {
		t.Errorf("Close = %v, want %v", got.Close, want[19].Close)
	}
	if got.High < got.Low {
		t.Errorf("High %v < Low %v", got.High, got.Low)
	}
	if got.Low > got.Open || got.High < want[19].Close {
		t.Errorf("extent [%v,%v] does not bracket the range", got.Low, got.High)
	}

	var volume int64
	for _, c := range want[:20] {
		volume += c.Volume
	}
	if got.Volume != volume {
		t.Errorf("Volume = %d, want %d", got.Volume, volume)
	}
	if got.VWAP <= got.Low || got.VWAP >= got.High {
		t.Errorf("VWAP %v is outside the range [%v,%v]", got.VWAP, got.Low, got.High)
	}

	wantReturn := want[19].Close/want[0].Open - 1
	if math.Abs(got.Return-wantReturn) > 1e-9 {
		t.Errorf("Return = %v, want %v", got.Return, wantReturn)
	}
	// A monotonically rising series cannot have a negative drawdown.
	if got.MaxDrawdown > 0 {
		t.Errorf("MaxDrawdown = %v, want <= 0 for a rising series", got.MaxDrawdown)
	}
	// Every bar of barAt returns exactly ln(1.01), so the dispersion is zero.
	// A separate test covers a series where it is not.
	if got.Volatility > 1e-9 {
		t.Errorf("Volatility = %v, want 0 for a constant-return series", got.Volatility)
	}
}

// TestSummaryVolatilityOnVaryingSeries asserts the realised-volatility figure
// responds to a series whose returns are not all the same.
func TestSummaryVolatilityOnVaryingSeries(t *testing.T) {
	dataDir := t.TempDir()
	w, err := storage.NewCandleWriterFromConfig(config.StorageConfig{
		DataDir:           dataDir,
		FlushIntervalSecs: 300,
		FlushMaxRows:      100,
		PartitionBy:       storage.PartitionByDate,
	})
	if err != nil {
		t.Fatalf("NewCandleWriterFromConfig: %v", err)
	}

	closes := []float64{100, 110, 105, 130, 120, 150}
	for i, c := range closes {
		candle := contracts.Candle{
			Symbol:    symbol,
			Timestamp: day.Add(time.Duration(i) * time.Minute),
			Open:      c,
			High:      c + 1,
			Low:       c - 1,
			Close:     c,
			Volume:    1,
		}
		if err := w.Append(storage.NewCandle(candle)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s, err := NewFromConfig(config.StorageConfig{DataDir: dataDir})
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	got, err := s.Summary(context.Background(), symbol, day, day.Add(time.Hour))
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}
	if got.Volatility <= 0 {
		t.Errorf("Volatility = %v, want > 0 on a zig-zag series", got.Volatility)
	}
	if got.MaxDrawdown >= 0 {
		t.Errorf("MaxDrawdown = %v, want < 0: the series falls from 130 to 120", got.MaxDrawdown)
	}
}

// TestSummaryEmptyRangeIsNotAnError asserts a window with no bars returns a
// zero value: an empty market is an answer, not a failure.
func TestSummaryEmptyRangeIsNotAnError(t *testing.T) {
	s, _ := storeWith(t, 1, 5, 5)
	got, err := s.Summary(context.Background(), symbol, day.AddDate(0, 1, 0), day.AddDate(0, 2, 0))
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}
	if got.Bars != 0 || got.Symbol != symbol {
		t.Errorf("Summary of an empty range = %+v, want a zero summary for %s", got, symbol)
	}
}

// TestReturnsAlignsWithBars asserts per-bar returns line up index-for-index
// with the candle slice a backtest iterates.
func TestReturnsAlignsWithBars(t *testing.T) {
	s, _ := storeWith(t, 1, 10, 4)
	ctx := context.Background()

	got, err := s.Returns(ctx, symbol, day, day.Add(time.Hour))
	if err != nil {
		t.Fatalf("Returns: %v", err)
	}
	if len(got) != 10 {
		t.Fatalf("Returns produced %d rows, want 10", len(got))
	}
	if got[0].Log != 0 || got[0].Simple != 0 {
		t.Errorf("first bar has no predecessor: got log=%v simple=%v, want 0", got[0].Log, got[0].Simple)
	}
	for i, r := range got {
		if r.Bar != i {
			t.Errorf("row %d reports bar %d", i, r.Bar)
		}
		if i == 0 {
			continue
		}
		if want := math.Log(1.01); math.Abs(r.Log-want) > 1e-9 {
			t.Errorf("bar %d: log = %v, want %v", i, r.Log, want)
		}
		if want := 0.01; math.Abs(r.Simple-want) > 1e-9 {
			t.Errorf("bar %d: simple = %v, want %v", i, r.Simple, want)
		}
	}
}

// TestSessionsBucketsByUTCDate asserts per-day aggregation across a
// multi-day range.
func TestSessionsBucketsByUTCDate(t *testing.T) {
	s, _ := storeWith(t, 3, 12, 5)
	ctx := context.Background()

	got, err := s.Sessions(ctx, symbol, day, day.AddDate(0, 3, 0))
	if err != nil {
		t.Fatalf("Sessions: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("Sessions returned %d days, want 3", len(got))
	}
	for i, sess := range got {
		wantDate := day.AddDate(0, 0, i).Format(storage.DateLayout)
		if sess.Date != wantDate {
			t.Errorf("session %d date = %s, want %s", i, sess.Date, wantDate)
		}
		if sess.Bars != 12 {
			t.Errorf("session %d has %d bars, want 12", i, sess.Bars)
		}
		if sess.Symbol != symbol {
			t.Errorf("session %d symbol = %s, want %s", i, sess.Symbol, symbol)
		}
		if sess.Low > sess.High {
			t.Errorf("session %d extent is inverted: [%v,%v]", i, sess.Low, sess.High)
		}
		if i > 0 && !(got[i-1].Date < sess.Date) {
			t.Errorf("sessions are not in date order: %s then %s", got[i-1].Date, sess.Date)
		}
	}
}

// TestScanSQLTargetsTheSameFiles asserts the SQL handed to DuckDB addresses the
// hive layout the writer actually produces, so a research query cannot drift
// from what is on disk.
func TestScanSQLTargetsTheSameFiles(t *testing.T) {
	s, _ := storeWith(t, 2, 5, 5)

	sql, err := s.ScanSQL(symbol)
	if err != nil {
		t.Fatalf("ScanSQL: %v", err)
	}
	if !strings.Contains(sql, "read_parquet(") || !strings.Contains(sql, "hive_partitioning=true") {
		t.Errorf("ScanSQL = %q, want a read_parquet call with hive partitioning", sql)
	}
	if !strings.Contains(sql, "symbol="+symbol) || !strings.Contains(sql, "date=*") {
		t.Errorf("ScanSQL = %q, want the symbol/date hive globs", sql)
	}
	if !strings.Contains(sql, "part-*.parquet") {
		t.Errorf("ScanSQL = %q, want the part file glob readers use", sql)
	}

	if _, err := s.ScanSQL(""); err == nil {
		t.Error("ScanSQL(\"\") = nil error, want a validation failure")
	}
}

// TestWindowValidation asserts the inference window helper rejects an inverted
// range rather than returning an empty context that looks like a quiet market.
func TestWindowValidation(t *testing.T) {
	end := day.Add(10 * time.Minute)
	w := Last(end, 100*time.Minute)
	if err := w.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if !w.End.Equal(end) {
		t.Errorf("window end = %s, want %s", w.End, end)
	}
	if got, want := w.End.Sub(w.Start), 100*time.Minute; got != want {
		t.Errorf("window length = %s, want %s", got, want)
	}
	if err := (Window{Start: end, End: end}).Validate(); err == nil {
		t.Error("Validate on an empty window = nil error, want a failure")
	}
}

// TestNewRejectsEmptyRoot asserts the Store cannot be built with no dataset
// path: silently reading "" would look like an empty market forever.
func TestNewRejectsEmptyRoot(t *testing.T) {
	if _, err := New(""); err == nil {
		t.Error("New(\"\") = nil error, want a validation failure")
	}
	if _, err := New(filepath.Join(t.TempDir(), "candles")); err != nil {
		t.Errorf("New: %v", err)
	}
}

// TestContextPropagatesCancellation asserts a cancelled query stops instead of
// scanning the whole dataset.
func TestContextPropagatesCancellation(t *testing.T) {
	s, _ := storeWith(t, 2, 30, 7)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := s.Context(ctx, symbol, 10); err == nil {
		t.Error("Context(cancelled) = nil error, want context.Canceled")
	}
}
