// Package analytics provides typed queries over the Parquet candle store.
package analytics

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/storage"
)

// Store is the read side of the platform: typed queries over the candles dataset.
type Store struct {
	reader *storage.CandleReader
}

// New returns a Store over an explicit dataset root, typically <data_dir>/candles.
func New(root string) (*Store, error) {
	reader, err := storage.NewCandleReader(root)
	if err != nil {
		return nil, err
	}
	return &Store{reader: reader}, nil
}

// NewFromConfig returns a Store rooted at <data_dir>/candles, resolved from the frozen
// storage config.
func NewFromConfig(cfg config.StorageConfig) (*Store, error) {
	reader, err := storage.NewCandleReaderFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &Store{reader: reader}, nil
}

// Reader exposes the underlying candle reader for callers that need the raw file list
// or a plain tail query.
func (s *Store) Reader() *storage.CandleReader { return s.reader }

// Summary aggregates a candle range into the numbers a researcher asks for first:
// extent, OHLC, volume, realised volatility and the simple return.
type Summary struct {
	Symbol      string
	Bars        int
	Start       time.Time
	End         time.Time
	Open        float64
	High        float64
	Low         float64
	Close       float64
	Volume      int64
	VWAP        float64
	Return      float64
	Volatility  float64
	MaxDrawdown float64
}

// Summary aggregates [start, end) for symbol.
func (s *Store) Summary(ctx context.Context, symbol string, start, end time.Time) (Summary, error) {
	candles, err := s.reader.CandleRange(ctx, symbol, start, end, 0)
	if err != nil {
		return Summary{}, err
	}
	if len(candles) == 0 {
		return Summary{Symbol: symbol}, nil
	}

	out := Summary{
		Symbol: symbol,
		Bars:   len(candles),
		Start:  candles[0].Timestamp,
		End:    candles[len(candles)-1].Timestamp,
		Open:   candles[0].Open,
		High:   candles[0].High,
		Low:    candles[0].Low,
		Close:  candles[len(candles)-1].Close,
	}
	if out.Open != 0 {
		out.Return = out.Close/out.Open - 1
	}

	var volume float64
	var typicalPV float64
	var sumSq, sum float64
	var n int
	peak := candles[0].Close
	for i, c := range candles {
		out.Volume += c.Volume
		typicalPV += (c.High + c.Low + c.Close) / 3 * float64(c.Volume)
		volume += float64(c.Volume)

		if c.High > out.High {
			out.High = c.High
		}
		if c.Low < out.Low {
			out.Low = c.Low
		}

		if i > 0 {
			if r := math.Log(c.Close / candles[i-1].Close); math.IsNaN(r) == false {
				sum += r
				sumSq += r * r
				n++
			}
		}
		if c.Close > peak {
			peak = c.Close
		}
		if peak > 0 {
			if dd := c.Close/peak - 1; dd < out.MaxDrawdown {
				out.MaxDrawdown = dd
			}
		}
	}

	if volume > 0 {
		out.VWAP = typicalPV / volume
	}
	if n > 1 {
		mean := sum / float64(n)
		out.Volatility = math.Sqrt(math.Max(0, sumSq/float64(n)-mean*mean))
	}
	return out, nil
}

// Return is one bar's log return against the previous bar.
type Return struct {
	Symbol    string
	Timestamp time.Time
	Bar       int
	Log       float64
	Simple    float64
}

// Returns computes per-bar returns over [start, end), oldest first.
func (s *Store) Returns(ctx context.Context, symbol string, start, end time.Time) ([]Return, error) {
	candles, err := s.reader.CandleRange(ctx, symbol, start, end, 0)
	if err != nil {
		return nil, err
	}

	out := make([]Return, 0, len(candles))
	for i, c := range candles {
		r := Return{Symbol: symbol, Timestamp: c.Timestamp, Bar: i}
		if i > 0 && candles[i-1].Close > 0 {
			r.Log = math.Log(c.Close / candles[i-1].Close)
			r.Simple = c.Close/candles[i-1].Close - 1
		}
		out = append(out, r)
	}
	return out, nil
}

// Session is one UTC trading day of a symbol's candles.
type Session struct {
	Date   string
	Symbol string
	Bars   int
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume int64
}

// Sessions buckets candles into UTC dates, oldest first.
func (s *Store) Sessions(ctx context.Context, symbol string, start, end time.Time) ([]Session, error) {
	candles, err := s.reader.CandleRange(ctx, symbol, start, end, 0)
	if err != nil {
		return nil, err
	}

	var out []Session
	for _, c := range candles {
		date := c.Timestamp.UTC().Format(storage.DateLayout)
		if len(out) == 0 || out[len(out)-1].Date != date {
			out = append(out, Session{Date: date, Symbol: symbol, Bars: 1, Open: c.Open, High: c.High, Low: c.Low, Close: c.Close, Volume: c.Volume})
			continue
		}
		sess := &out[len(out)-1]
		sess.Bars++
		sess.Close = c.Close
		sess.Volume += c.Volume
		if c.High > sess.High {
			sess.High = c.High
		}
		if c.Low < sess.Low {
			sess.Low = c.Low
		}
	}
	return out, nil
}

// Coverage reports how much history exists for a symbol, which is what the feature
// builder needs before it can claim the 60-bar warm-up is satisfied.
type Coverage struct {
	Symbol   string
	Bars     int
	Days     int
	First    time.Time
	Last     time.Time
	Complete bool
}

// WarmupBars is the number of context-only bars the feature schema requires before any
// bar may be used as a prediction target (docs/contracts.md §4).
const WarmupBars = 60

// Coverage counts the candles stored for symbol across the whole dataset.
func (s *Store) Coverage(ctx context.Context, symbol string) (Coverage, error) {
	candles, err := s.reader.Candles(ctx, symbol, 0)
	if err != nil {
		return Coverage{}, err
	}
	if len(candles) == 0 {
		return Coverage{Symbol: symbol}, nil
	}

	days := make(map[string]struct{}, 8)
	for _, c := range candles {
		days[c.Timestamp.UTC().Format(storage.DateLayout)] = struct{}{}
	}

	return Coverage{
		Symbol:   symbol,
		Bars:     len(candles),
		Days:     len(days),
		First:    candles[0].Timestamp,
		Last:     candles[len(candles)-1].Timestamp,
		Complete: len(candles) >= WarmupBars,
	}, nil
}

// ScanSQL returns the DuckDB relation that reads exactly the files holding symbol's
// candles, for handing to the Python side or `make backtest`.
func (s *Store) ScanSQL(symbol string) (string, error) {
	if symbol == "" {
		return "", fmt.Errorf("analytics: symbol must not be empty")
	}
	return "read_parquet('" + s.reader.Glob(symbol) + "', hive_partitioning=true)", nil
}

// Window is the half-open interval [Start, End) a query covers.
type Window struct {
	Start time.Time
	End   time.Time
}

// Last returns the window covering the most recent duration up to end.
func Last(end time.Time, duration time.Duration) Window {
	return Window{Start: end.Add(-duration), End: end}
}

// Validate reports whether the window is usable.
func (w Window) Validate() error {
	if !w.Start.Before(w.End) {
		return fmt.Errorf("analytics: window start %s must be before end %s",
			w.Start.UTC().Format(time.RFC3339), w.End.UTC().Format(time.RFC3339))
	}
	return nil
}

// Context returns the most recent `rows` candles for symbol, ascending by time — the
// exact shape the TabFM inference loop consumes.
func (s *Store) Context(ctx context.Context, symbol string, rows int) ([]contracts.Candle, error) {
	return s.reader.Candles(ctx, symbol, rows)
}
