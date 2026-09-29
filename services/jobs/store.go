package jobs

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

// Parquet layout of docs/contracts.md §3 for the historical tree:
//
//	data/historical/symbol=<SYMBOL>/from=<date>/to=<date>/candles.parquet
//
// The from/to pair is the segment identity, so the path is a pure function of
// the segment and two runs of the same backfill land on the same file.
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
//
// Timestamps are Unix milliseconds, as docs/contracts.md §1 requires; the
// symbol is carried on every row so a reader that globs the whole tree does
// not have to parse the partition path to know what a bar belongs to.
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

// ParquetStore writes historical segments to the hive-partitioned Parquet tree
// that DuckDB reads natively.
//
// It is a local implementation of the [SegmentStore] seam rather than an
// adapter onto core/storage: C2's Writer is tick-shaped, appends into
// wall-clock-keyed files, and has no concept of a closed segment. See the
// package comment.
type ParquetStore struct {
	dir      string
	interval string
}

// NewParquetStore returns a store that writes under the data/historical root.
func NewParquetStore(dir string) *ParquetStore {
	return &ParquetStore{dir: strings.TrimRight(dir, "/"), interval: defaultInterval}
}

// newParquetStore is NewParquetStore with an explicit interval recorded on
// every row, for tests that backfill more than one interval.
func newParquetStore(dir, interval string) *ParquetStore {
	return &ParquetStore{dir: strings.TrimRight(dir, "/"), interval: interval}
}

// Root returns the historical directory the store writes into.
func (s *ParquetStore) Root() string { return s.dir }

// SegmentPath returns the absolute Parquet file a segment is written to.
func (s *ParquetStore) SegmentPath(seg Segment) (string, error) {
	if err := validSegment(seg); err != nil {
		return "", err
	}
	return filepath.Join(s.dir,
		fmt.Sprintf(segmentSymbolDir, segmentSymbol(seg)),
		fmt.Sprintf(segmentFromDir, seg.From.UTC().Format(time.DateOnly)),
		fmt.Sprintf(segmentToDir, seg.To.UTC().Format(time.DateOnly)),
		segmentFileName,
	), nil
}

// HasSegment implements SegmentStore. A segment counts as present only when
// its file exists and is non-empty, so a zero-length placeholder left by an
// earlier crash is refetched rather than trusted.
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
	return info.Size() > 0, nil
}

// WriteSegment implements SegmentStore.
//
// The write is atomic: rows go to a sibling .partial file which is renamed
// into place only after a successful close. A reader therefore never sees a
// half-written file, and a run interrupted mid-write leaves a .partial that
// the next run overwrites instead of a truncated candles.parquet that
// HasSegment would mistake for a finished segment.
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

// ReadHistoricalCandles reads back the rows of one segment file. It exists so
// a caller — and the test suite — can verify a landed segment without reaching
// for DuckDB; DuckDB reads the same hive layout natively.
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

// String renders a segment as symbol/from..to, which is the identity a log
// line and a metric label both want.
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
