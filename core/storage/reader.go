package storage

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/parquet-go/parquet-go"
)

const (
	// partGlob matches every published file in a partition directory. The
	// in-flight files this package writes start with a dot and are named
	// ".inflight-*.parquet", so a reader never observes a partial file.
	partGlob = "part-*.parquet"
)

// Reader queries the Parquet store. It is the interface frozen in
// docs/contracts.md §3.
type Reader interface {
	// Candles returns up to limit candles for symbol, ascending by time.
	// limit <= 0 returns every candle stored for the symbol.
	Candles(ctx context.Context, symbol string, limit int) ([]contracts.Candle, error)
}

// CandleReader is the Reader implementation over the candles dataset. It is
// read-only by construction: it opens files, never creates them.
type CandleReader struct {
	root string
}

// NewCandleReader returns a reader over an explicit dataset root, typically
// <data_dir>/candles.
func NewCandleReader(root string) (*CandleReader, error) {
	if root == "" {
		return nil, fmt.Errorf("storage: candle reader root must not be empty")
	}
	return &CandleReader{root: root}, nil
}

// NewCandleReaderFromConfig returns a reader rooted at <data_dir>/candles,
// resolved from the frozen storage config.
func NewCandleReaderFromConfig(cfg config.StorageConfig) (*CandleReader, error) {
	root, err := dataDirFor(cfg, DatasetCandles)
	if err != nil {
		return nil, err
	}
	return NewCandleReader(root)
}

// Root returns the dataset directory the reader scans.
func (r *CandleReader) Root() string { return r.root }

// Candles returns the most recent limit candles for symbol, ascending by
// time. Reading only what the caller needs is why limit is a "tail" rather
// than a "head": the inference service pulls the last N bars at every minute
// close, and materialising years of history to keep the last 100 would be
// absurd. limit <= 0 returns every candle stored for the symbol.
func (r *CandleReader) Candles(ctx context.Context, symbol string, limit int) ([]contracts.Candle, error) {
	if symbol == "" {
		return nil, fmt.Errorf("storage: symbol must not be empty")
	}

	files, err := r.Files(ctx, symbol)
	if err != nil {
		return nil, err
	}

	all, err := r.read(ctx, files)
	if err != nil {
		return nil, err
	}
	if limit > 0 && len(all) > limit {
		all = all[len(all)-limit:]
	}
	return all, nil
}

// CandleRange returns candles for symbol with start <= timestamp < end,
// ascending by time, capped at limit rows (limit <= 0 means no cap). It is the
// query backtesting replays through: any interval, any length.
func (r *CandleReader) CandleRange(ctx context.Context, symbol string, start, end time.Time, limit int) ([]contracts.Candle, error) {
	if symbol == "" {
		return nil, fmt.Errorf("storage: symbol must not be empty")
	}
	if !start.Before(end) {
		return nil, fmt.Errorf("storage: start %s must be before end %s",
			start.UTC().Format(time.RFC3339), end.UTC().Format(time.RFC3339))
	}

	files, err := r.Files(ctx, symbol)
	if err != nil {
		return nil, err
	}

	all, err := r.read(ctx, files)
	if err != nil {
		return nil, err
	}

	out := make([]contracts.Candle, 0, len(all))
	for _, c := range all {
		if c.Timestamp.Before(start) || !c.Timestamp.Before(end) {
			continue
		}
		if limit > 0 && len(out) == limit {
			break
		}
		out = append(out, c)
	}
	return out, nil
}

// Files returns every published parquet file for symbol, ordered oldest first.
// Ordering by partition date then by file timestamp keeps merges deterministic.
func (r *CandleReader) Files(ctx context.Context, symbol string) ([]string, error) {
	if symbol == "" {
		return nil, fmt.Errorf("storage: symbol must not be empty")
	}
	symDir := SymbolDir(r.root, symbol)

	entries, err := os.ReadDir(symDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("storage: read symbol dir %s: %w", symDir, err)
	}

	var dates []string
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "date=") {
			continue
		}
		dates = append(dates, strings.TrimPrefix(e.Name(), "date="))
	}
	sort.Strings(dates)

	var files []string
	for _, date := range dates {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("storage: list candles for %s: %w", symbol, err)
		}
		dir := filepath.Join(symDir, "date="+date)
		matches, err := filepath.Glob(filepath.Join(dir, partGlob))
		if err != nil {
			return nil, fmt.Errorf("storage: glob %s: %w", dir, err)
		}
		sort.Strings(matches)
		files = append(files, matches...)
	}
	return files, nil
}

// Symbols returns every symbol present in the candles dataset, sorted.
func (r *CandleReader) Symbols(ctx context.Context) ([]string, error) {
	entries, err := os.ReadDir(r.root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("storage: read dataset dir %s: %w", r.root, err)
	}

	var symbols []string
	for _, e := range entries {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("storage: list symbols: %w", err)
		}
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "symbol=") {
			continue
		}
		symbols = append(symbols, unescapeHiveValue(strings.TrimPrefix(e.Name(), "symbol=")))
	}
	sort.Strings(symbols)
	return symbols, nil
}

// Glob returns the read_parquet pattern covering every published candle file
// for symbol. It is the path expression DuckDB consumes from the Python side,
// so a research query and the Go reader address the same files.
func (r *CandleReader) Glob(symbol string) string {
	return filepath.Join(SymbolDir(r.root, symbol), "date=*", partGlob)
}

// read decodes the given files and returns their candles ascending by time.
//
// Files are read in the order supplied, so a later file wins on a duplicate
// timestamp: a re-flushed or backfilled candle supersedes the earlier copy.
// That is what makes an idempotent backfill possible.
func (r *CandleReader) read(ctx context.Context, files []string) ([]contracts.Candle, error) {
	byTS := make(map[int64]contracts.Candle)
	var order []int64

	for _, path := range files {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("storage: read candles: %w", err)
		}
		rows, err := readCandleFile(path)
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			if _, seen := byTS[row.Timestamp]; !seen {
				order = append(order, row.Timestamp)
			}
			byTS[row.Timestamp] = row.Contract()
		}
	}

	sort.Slice(order, func(i, j int) bool { return order[i] < order[j] })

	out := make([]contracts.Candle, 0, len(order))
	for _, ts := range order {
		out = append(out, byTS[ts])
	}
	return out, nil
}

// readCandleFile decodes one published parquet file.
func readCandleFile(path string) ([]Candle, error) {
	rows, err := parquet.ReadFile[Candle](path)
	if err != nil {
		return nil, fmt.Errorf("storage: read parquet %s: %w", path, err)
	}
	return rows, nil
}
