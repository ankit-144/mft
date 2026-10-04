package jobs

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"

	"github.com/mft/core/config"
)

// Parquet layout of docs/contracts.md §3 for the historical tree:
const (
	segmentSymbolDir = "symbol=%s"
	segmentFromDir   = "from=%s"
	segmentToDir     = "to=%s"
	segmentFileName  = "candles.parquet"
	segmentDirPerm   = 0o755
	segmentFilePerm  = 0o644
	segmentTmpSuffix = ".partial"
)

// HistoricalCandle is the on-disk schema of a historical candle row.
type HistoricalCandle struct {
	Symbol       string  `parquet:"symbol"`
	Instrument   int64   `parquet:"instrument_token"`
	Timestamp    int64   `parquet:"timestamp"`
	Open         float64 `parquet:"open"`
	High         float64 `parquet:"high"`
	Low          float64 `parquet:"low"`
	Close        float64 `parquet:"close"`
	Volume       int64   `parquet:"volume"`
	Interval     string  `parquet:"interval"`
	SegmentStart int64   `parquet:"segment_from"`
	SegmentEnd   int64   `parquet:"segment_to"`
}

// ParquetStore writes historical segments to the hive-partitioned Parquet tree that
// DuckDB reads natively.
type ParquetStore struct {
	dir        string
	interval   string
	segmentLen time.Duration
}

// NewParquetStore returns a store that writes under the data/historical root.
func NewParquetStore(dir string) *ParquetStore {
	return &ParquetStore{dir: strings.TrimRight(dir, "/"), interval: defaultInterval, segmentLen: segmentDays * day}
}

// NewParquetStoreFromConfig builds the store from the shared config.
func NewParquetStoreFromConfig(cfg *config.Config) *ParquetStore {
	return NewParquetStore(cfg.Jobs.HistoricalDir)
}

// newParquetStore is NewParquetStore with an explicit interval recorded on every row,
// for tests that backfill more than one interval.
func newParquetStore(dir, interval string) *ParquetStore {
	return &ParquetStore{dir: strings.TrimRight(dir, "/"), interval: interval, segmentLen: segmentDays * day}
}

// Root returns the historical directory the store writes into.
func (s *ParquetStore) Root() string { return s.dir }

// SegmentPath returns the absolute Parquet file a segment is written to.
func (s *ParquetStore) SegmentPath(seg Segment) (string, error) {
	if err := validSegment(seg); err != nil {
		return "", err
	}
	start, end := s.segmentBucket(seg.From)
	return filepath.Join(s.dir,
		fmt.Sprintf(segmentSymbolDir, segmentSymbol(seg)),
		fmt.Sprintf(segmentFromDir, start.Format(time.DateOnly)),
		fmt.Sprintf(segmentToDir, end.Format(time.DateOnly)),
		segmentFileName,
	), nil
}

func (s *ParquetStore) segmentBucket(at time.Time) (time.Time, time.Time) {
	local := at.In(indiaTimeZone)
	dayStart := time.Date(local.Year(), local.Month(), local.Day(), 0, 0, 0, 0, indiaTimeZone)
	epoch := time.Date(1970, 1, 1, 0, 0, 0, 0, indiaTimeZone)
	days := int(dayStart.Sub(epoch) / day)
	width := int(s.segmentLen / day)
	if width <= 0 {
		width = segmentDays
	}
	start := epoch.AddDate(0, 0, days/width*width)
	return start, start.AddDate(0, 0, width)
}

// HasSegment reports whether the stable bucket already covers the requested range.
func (s *ParquetStore) HasSegment(_ context.Context, seg Segment) (bool, error) {
	path, err := s.SegmentPath(seg)
	if err != nil {
		return false, err
	}
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("stat segment %s: %w", path, err)
	}
	if info.Size() == 0 {
		return false, nil
	}
	rows, err := ReadHistoricalCandles(path)
	if err != nil {
		return false, err
	}
	if len(rows) == 0 {
		_, bucketEnd := s.segmentBucket(seg.From)
		return !seg.To.Before(bucketEnd), nil
	}
	start := time.UnixMilli(rows[0].SegmentStart)
	end := time.UnixMilli(rows[0].SegmentEnd)
	return !start.After(seg.From) && !end.Before(seg.To), nil
}

// WriteSegment implements SegmentStore.
func (s *ParquetStore) WriteSegment(ctx context.Context, seg Segment) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.SegmentPath(seg)
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, segmentDirPerm); err != nil {
		return fmt.Errorf("create segment dir %s: %w", dir, err)
	}

	rows := make([]HistoricalCandle, 0, len(seg.Candles))
	for i, c := range seg.Candles {
		if c.Timestamp.Before(seg.From) || !c.Timestamp.Before(seg.To) {
			return fmt.Errorf("candle %d for %s at %s falls outside segment %s..%s",
				i, c.Symbol, c.Timestamp.Format(time.RFC3339),
				seg.From.Format(time.DateOnly), seg.To.Format(time.DateOnly))
		}
		rows = append(rows, HistoricalCandle{
			Symbol:       c.Symbol,
			Timestamp:    c.Timestamp.UTC().UnixMilli(),
			Open:         c.Open,
			High:         c.High,
			Low:          c.Low,
			Close:        c.Close,
			Volume:       c.Volume,
			Interval:     s.interval,
			SegmentStart: seg.From.UTC().UnixMilli(),
			SegmentEnd:   seg.To.UTC().UnixMilli(),
		})
	}

	tmp := path + segmentTmpSuffix
	if err := writeParquet(tmp, rows); err != nil {
		return fmt.Errorf("write segment %s: %w", seg.String(), err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("publish segment %s: %w", seg.String(), err)
	}
	return nil
}

// writeParquet serialises rows to path, replacing whatever was there.
func writeParquet(path string, rows []HistoricalCandle) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	w := parquet.NewGenericWriter[HistoricalCandle](f)
	if _, err := w.Write(rows); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return fmt.Errorf("write rows: %w", err)
	}
	if err := w.Close(); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return fmt.Errorf("close writer: %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return fmt.Errorf("close file %s: %w", path, err)
	}
	if err := os.Chmod(path, segmentFilePerm); err != nil {
		return fmt.Errorf("chmod %s: %w", path, err)
	}
	return nil
}

// ReadHistoricalCandles reads back the rows of one segment file.
func ReadHistoricalCandles(path string) ([]HistoricalCandle, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	rows, err := parquet.Read[HistoricalCandle](f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return rows, nil
}

// String renders a segment as symbol/from..to, which is the identity a log line and a
// metric label both want.
func (s Segment) String() string {
	return fmt.Sprintf("%s %s..%s", s.Symbol, s.From.UTC().Format(time.DateOnly), s.To.UTC().Format(time.DateOnly))
}

// segmentSymbol normalises a symbol for use as a directory name.
func segmentSymbol(seg Segment) string {
	return strings.ToUpper(strings.TrimSpace(seg.Symbol))
}

// validSegment rejects a segment that could not be written unambiguously.
func validSegment(seg Segment) error {
	if segmentSymbol(seg) == "" {
		return fmt.Errorf("segment has no symbol: %w", ErrInvalidSegment)
	}
	if seg.From.IsZero() || seg.To.IsZero() {
		return fmt.Errorf("segment %q has an unbound date range: %w", seg.Symbol, ErrInvalidSegment)
	}
	if !seg.To.After(seg.From) {
		return fmt.Errorf("segment %q ends before it starts: %w", seg.Symbol, ErrInvalidSegment)
	}
	return nil
}

// Compile-time proof that the Parquet store satisfies the storage seam.
var _ SegmentStore = (*ParquetStore)(nil)
