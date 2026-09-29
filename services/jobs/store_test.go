package jobs

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/contracts"
)

func TestParquetStoreWritesTheDocumentedHiveLayout(t *testing.T) {
	dir := t.TempDir()
	store := newParquetStore(dir, "minute")
	seg := Segment{
		Symbol: "RELIANCE",
		From:   time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:     time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC),
		Candles: []contracts.Candle{
			{Symbol: "RELIANCE", Timestamp: time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC), Open: 2543, High: 2549, Low: 2535, Close: 2545.5, Volume: 123456},
		},
	}

	if err := store.WriteSegment(context.Background(), seg); err != nil {
		t.Fatalf("WriteSegment() error = %v", err)
	}

	want := filepath.Join(dir, "symbol=RELIANCE", "from=2024-01-01", "to=2024-03-01", "candles.parquet")
	path, err := store.SegmentPath(seg)
	if err != nil {
		t.Fatalf("SegmentPath() error = %v", err)
	}
	if path != want {
		t.Fatalf("SegmentPath() = %q, want %q", path, want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("segment file not written: %v", err)
	}
}

func TestParquetStoreRoundTripsRows(t *testing.T) {
	dir := t.TempDir()
	store := newParquetStore(dir, "minute")
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	seg := Segment{Symbol: "TCS", From: start, To: start.Add(60 * 24 * time.Hour)}
	seg.Candles = []contracts.Candle{
		{Symbol: "TCS", Timestamp: start.Add(time.Minute), Open: 100, High: 110, Low: 95, Close: 105, Volume: 42},
		{Symbol: "TCS", Timestamp: start.Add(2 * time.Minute), Open: 105, High: 120, Low: 104, Close: 118, Volume: 43},
	}

	if err := store.WriteSegment(context.Background(), seg); err != nil {
		t.Fatalf("WriteSegment() error = %v", err)
	}
	path, _ := store.SegmentPath(seg)
	rows, err := ReadHistoricalCandles(path)
	if err != nil {
		t.Fatalf("ReadHistoricalCandles() error = %v", err)
	}
	if len(rows) != len(seg.Candles) {
		t.Fatalf("read %d rows, wrote %d", len(rows), len(seg.Candles))
	}

	got := rows[1]
	want := seg.Candles[1]
	if got.Symbol != "TCS" {
		t.Errorf("Symbol = %q, want TCS", got.Symbol)
	}
	if got.Timestamp != want.Timestamp.UnixMilli() {
		t.Errorf("Timestamp = %d, want %d (Unix millis per docs/contracts.md §1)", got.Timestamp, want.Timestamp.UnixMilli())
	}
	if got.Open != want.Open || got.High != want.High || got.Low != want.Low || got.Close != want.Close {
		t.Errorf("OHLC = %v/%v/%v/%v, want %v/%v/%v/%v", got.Open, got.High, got.Low, got.Close, want.Open, want.High, want.Low, want.Close)
	}
	if got.Volume != want.Volume {
		t.Errorf("Volume = %d, want %d", got.Volume, want.Volume)
	}
	if got.Interval != "minute" {
		t.Errorf("Interval = %q, want minute", got.Interval)
	}
	if got.SegmentStart != seg.From.UnixMilli() || got.SegmentEnd != seg.To.UnixMilli() {
		t.Errorf("segment bounds = %d..%d, want %d..%d",
			got.SegmentStart, got.SegmentEnd, seg.From.UnixMilli(), seg.To.UnixMilli())
	}
}

func TestParquetStoreHasSegmentTracksTheFile(t *testing.T) {
	dir := t.TempDir()
	store := newParquetStore(dir, "minute")
	seg := Segment{
		Symbol:  "INFY",
		From:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:      time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
		Candles: []contracts.Candle{{Symbol: "INFY", Timestamp: time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC), Close: 1}},
	}

	present, err := store.HasSegment(context.Background(), seg)
	if err != nil {
		t.Fatalf("HasSegment() error = %v", err)
	}
	if present {
		t.Fatal("HasSegment() = true before anything was written")
	}

	if err := store.WriteSegment(context.Background(), seg); err != nil {
		t.Fatalf("WriteSegment() error = %v", err)
	}
	present, err = store.HasSegment(context.Background(), seg)
	if err != nil {
		t.Fatalf("HasSegment() error = %v", err)
	}
	if !present {
		t.Fatal("HasSegment() = false after WriteSegment()")
	}
}

func TestParquetStoreTreatsAnEmptyFileAsAbsent(t *testing.T) {
	// A zero-length file is what an interrupted pre-atomic write leaves
	// behind. Reporting it as done would silently skip a segment forever.
	dir := t.TempDir()
	store := newParquetStore(dir, "minute")
	seg := Segment{
		Symbol: "INFY",
		From:   time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:     time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
	}
	path, _ := store.SegmentPath(seg)
	if err := os.MkdirAll(filepath.Dir(path), segmentDirPerm); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, segmentFilePerm); err != nil {
		t.Fatal(err)
	}

	present, err := store.HasSegment(context.Background(), seg)
	if err != nil {
		t.Fatalf("HasSegment() error = %v", err)
	}
	if present {
		t.Fatal("HasSegment() = true for a zero-length file, want false")
	}
}

func TestParquetStoreLeavesNoPartialFileBehind(t *testing.T) {
	dir := t.TempDir()
	store := newParquetStore(dir, "minute")
	seg := Segment{
		Symbol:  "TCS",
		From:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:      time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
		Candles: []contracts.Candle{{Symbol: "TCS", Timestamp: time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC), Close: 1}},
	}
	path, _ := store.SegmentPath(seg)
	if err := os.MkdirAll(filepath.Dir(path), segmentDirPerm); err != nil {
		t.Fatal(err)
	}
	// A stale .partial from a killed process must be replaced, not skipped.
	if err := os.WriteFile(path+segmentTmpSuffix, []byte("garbage"), segmentFilePerm); err != nil {
		t.Fatal(err)
	}

	if err := store.WriteSegment(context.Background(), seg); err != nil {
		t.Fatalf("WriteSegment() error = %v", err)
	}
	if _, err := os.Stat(path + segmentTmpSuffix); !os.IsNotExist(err) {
		t.Fatalf("a .partial file survived the write (stat err = %v)", err)
	}
	if _, err := ReadHistoricalCandles(path); err != nil {
		t.Fatalf("published file is not readable: %v", err)
	}
}

func TestParquetStoreRejectsACandleOutsideItsSegment(t *testing.T) {
	dir := t.TempDir()
	store := newParquetStore(dir, "minute")
	seg := Segment{
		Symbol:  "TCS",
		From:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:      time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
		Candles: []contracts.Candle{{Symbol: "TCS", Timestamp: time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC), Close: 1}},
	}

	if err := store.WriteSegment(context.Background(), seg); err == nil {
		t.Fatal("WriteSegment() with an out-of-range candle returned nil, want an error")
	}
}

func TestParquetStoreRejectsMalformedSegments(t *testing.T) {
	store := newParquetStore(t.TempDir(), "minute")
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	cases := map[string]Segment{
		"no symbol":     {From: base, To: base.AddDate(0, 1, 0)},
		"unbound from":  {Symbol: "TCS", To: base.AddDate(0, 1, 0)},
		"ends first":    {Symbol: "TCS", From: base.AddDate(0, 1, 0), To: base},
		"zero-length":   {Symbol: "TCS", From: base, To: base},
		"blank symbol":  {Symbol: "   ", From: base, To: base.AddDate(0, 1, 0)},
		"reversed zero": {Symbol: "TCS"},
	}
	for name, seg := range cases {
		if _, err := store.SegmentPath(seg); !errors.Is(err, ErrInvalidSegment) {
			t.Errorf("%s: SegmentPath() error = %v, want it to wrap ErrInvalidSegment", name, err)
		}
	}
}

func TestParquetStoreRejectsACancelledContext(t *testing.T) {
	store := newParquetStore(t.TempDir(), "minute")
	seg := Segment{
		Symbol:  "TCS",
		From:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:      time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
		Candles: []contracts.Candle{{Symbol: "TCS", Timestamp: time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC), Close: 1}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := store.WriteSegment(ctx, seg); !errors.Is(err, context.Canceled) {
		t.Fatalf("WriteSegment() on a cancelled context = %v, want context.Canceled", err)
	}
}

func TestParquetStoreUppercasesThePartitionKey(t *testing.T) {
	store := newParquetStore("/data/historical", "minute")
	seg := Segment{
		Symbol: " reliance ",
		From:   time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:     time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC),
	}
	path, err := store.SegmentPath(seg)
	if err != nil {
		t.Fatalf("SegmentPath() error = %v", err)
	}
	if !strings.Contains(path, string(filepath.Separator)+"symbol=RELIANCE"+string(filepath.Separator)) {
		t.Fatalf("path = %q, want a symbol=RELIANCE partition", path)
	}
}

func TestParquetStoreRootTrimsTrailingSlash(t *testing.T) {
	if got := NewParquetStore("/data/historical/").Root(); got != "/data/historical" {
		t.Fatalf("Root() = %q, want /data/historical", got)
	}
}

func TestSegmentStringNamesTheSegment(t *testing.T) {
	seg := Segment{
		Symbol: "RELIANCE",
		From:   time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		To:     time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC),
	}
	if got, want := seg.String(), "RELIANCE 2024-01-01..2024-03-01"; got != want {
		t.Fatalf("String() = %q, want %q", got, want)
	}
}
