package storage

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/mft/core/config"
)

// Options configures a hive-partitioned Parquet writer.
type Options struct {
	// Root is the directory the dataset lives in, e.g.
	Root string
	// MaxRows is the number of buffered rows that triggers a flush.
	MaxRows int
	// FlushInterval is the background flush cadence.
	FlushInterval time.Duration
}

// Partition is a hive partition key: one symbol and one UTC date.
type Partition struct {
	// Symbol is the instrument symbol, unescaped.
	Symbol string
	// Date is the UTC date the row belongs to, formatted as DateLayout.
	Date string
}

// NewPartition builds the partition for a symbol at a point in time.
func NewPartition(symbol string, ts time.Time) Partition {
	return Partition{Symbol: symbol, Date: ts.UTC().Format(DateLayout)}
}

// Dir returns the hive directory for the partition under root, matching the layout in
// docs/contracts.md §3:
func (p Partition) Dir(root string) string {
	return filepath.Join(root, "symbol="+escapeHiveValue(p.Symbol), "date="+p.Date)
}

// SymbolDir returns the directory holding every date partition of a symbol:
func SymbolDir(root, symbol string) string {
	return filepath.Join(root, "symbol="+escapeHiveValue(symbol))
}

// String renders the partition as a hive path relative to a dataset root.
func (p Partition) String() string {
	return "symbol=" + escapeHiveValue(p.Symbol) + "/date=" + p.Date
}

// RangePartition is the historical dataset partition from docs/contracts.md §3:
type RangePartition struct {
	// Symbol is the instrument symbol, unescaped.
	Symbol string
	// From is the inclusive start of the requested window, DateLayout.
	From string
	// To is the exclusive end of the requested window, DateLayout.
	To string
}

// NewRangePartition builds the historical partition covering [from, to).
func NewRangePartition(symbol string, from, to time.Time) RangePartition {
	return RangePartition{
		Symbol: symbol,
		From:   from.UTC().Format(DateLayout),
		To:     to.UTC().Format(DateLayout),
	}
}

// Dir returns the hive directory for the range under root.
func (r RangePartition) Dir(root string) string {
	return filepath.Join(root,
		"symbol="+escapeHiveValue(r.Symbol),
		"from="+r.From,
		"to="+r.To)
}

// File returns the single parquet file the range partition holds.
func (r RangePartition) File(root string) string {
	return filepath.Join(r.Dir(root), "candles.parquet")
}

// String renders the range as a hive path relative to a dataset root.
func (r RangePartition) String() string {
	return "symbol=" + escapeHiveValue(r.Symbol) + "/from=" + r.From + "/to=" + r.To
}

// escapeHiveValue percent-encodes the characters that would otherwise break a hive
// "key=value" directory name.
func escapeHiveValue(s string) string {
	if s == "" {
		return "%00"
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z',
			c >= 'A' && c <= 'Z',
			c >= '0' && c <= '9',
			c == '.', c == '_', c == '-':
			b.WriteByte(c)
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

// unescapeHiveValue reverses escapeHiveValue.
func unescapeHiveValue(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] != '%' || i+2 >= len(s) {
			b.WriteByte(s[i])
			continue
		}
		var v int
		if _, err := fmt.Sscanf(s[i+1:i+3], "%02X", &v); err != nil {
			b.WriteByte(s[i])
			continue
		}
		b.WriteByte(byte(v))
		i += 2
	}
	return b.String()
}

// DatasetRoot returns the directory a dataset occupies under dataDir.
func DatasetRoot(dataDir string, dataset Dataset) (string, error) {
	if dataDir == "" {
		return "", fmt.Errorf("storage: data_dir must not be empty")
	}
	if dataset == "" {
		return "", fmt.Errorf("storage: dataset must not be empty")
	}
	return filepath.Join(dataDir, string(dataset)), nil
}

// dataDirFor returns the dataset root a config points at, defaulting data_dir to "data"
// exactly as core/config.Validate does.
func dataDirFor(cfg config.StorageConfig, dataset Dataset) (string, error) {
	dir := cfg.DataDir
	if dir == "" {
		dir = "data"
	}
	return DatasetRoot(dir, dataset)
}
