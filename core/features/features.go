// Package features turns a window of 1-minute candles into the frozen
// 18-column feature table that TabFM consumes (docs/contracts.md §4). It is a
// pure function of its input: no I/O, no clock, no configuration read, no
// randomness. Every timestamp it uses comes from the candles it was handed,
// which is what makes a feature table reproducible from a Parquet pull and
// comparable between a backtest and a live minute.
//
// # Row alignment
//
// Build is row-for-row with its input: len(Rows) == len(candles) and Rows[i]
// belongs to candles[i], exactly as the contract states. The inference service
// zips a row with the candle it came from to build the regression target, so
// this is not negotiable — a table shifted by the warm-up length would be
// perfectly well-formed and perfectly wrong, training the model on a label an
// hour away.
//
// The cost of that rule is that the leading WarmupRows rows exist but say
// nothing about the market. Where a feature's trailing window is not yet full
// at row i, the cell is 0. That is one uniform rule rather than a per-feature
// exception, and WarmupRows is exactly the longest window in the schema
// (ret_60), so row WarmupRows is the first row at which every column is fully
// informed. IsWarmup reports the rest.
//
// # Numeric policy
//
// A zero is substituted, never propagated, and Build never returns NaN or an
// infinity. The degenerate denominators this schema divides by are ordinary
// states, not errors: a flat window has zero stdev, a bar that never traded has
// zero volume, a doji has zero range. "No information" is precisely what a zero
// column tells the model, and a NaN would poison an entire TabFM context table
// rather than merely degrade one row.
//
// The opposite applies to corrupt *input*. A non-finite price, a non-positive
// price, an OHLC bar whose high is below its close, a window mixing symbols or
// running backwards in time: Build returns an error instead of a table of
// zeros. Fabricating features from bad data is how a broken feed turns into a
// confidently wrong signal an hour later.
//
// # Window position does not matter
//
// Every feature is a pure function of a fixed trailing window ending at its own
// bar. None is recursive, and none is seeded from an earlier row — RSI(14)
// averages the last 14 changes rather than carrying Wilder's smoothing, which
// would make row i depend on everything back to row 14.
//
// The reason is the pull architecture (Plan.md §4). Inference reads the last
// N candles it was configured with, and backfill may read 100 or 100000. A
// feature whose value drifts with the length of the window would hand the model
// a different number for the same market state depending on how much history the
// caller happened to pull. Here, the newest row of a 100-candle build and of a
// 100000-candle build are bit-identical.
package features

import (
	"fmt"
	"time"

	"github.com/mft/core/contracts"
)

// defaultColumns is the frozen schema of docs/contracts.md §4: 18 names, in the
// order TabFM was given them. It is an array so nothing can append to it, and
// every consumer reads it through DefaultColumns, which hands back a copy.
var defaultColumns = [...]string{
	"ret_1",
	"ret_5",
	"ret_15",
	"ret_60",
	"vol_5",
	"vol_20",
	"vol_ratio",
	"range_1",
	"body_1",
	"upper_wick_1",
	"lower_wick_1",
	"volume_z_20",
	"volume_ratio",
	"momentum_rsi_14",
	"sma_gap_10",
	"minute_of_session",
	"hour_of_day",
	"spread_proxy",
}

// Column indices into a FeatureTable row. They are the iota values of
// defaultColumns, named so the build loop reads as the schema rather than as a
// scatter of literals.
const (
	colRet1 = iota
	colRet5
	colRet15
	colRet60
	colVol5
	colVol20
	colVolRatio
	colRange1
	colBody1
	colUpperWick1
	colLowerWick1
	colVolumeZ20
	colVolumeRatio
	colMomentumRSI14
	colSMAGap10
	colMinuteOfSession
	colHourOfDay
	colSpreadProxy
)

// numColumns is the width of every row.
const numColumns = len(defaultColumns)

// Trailing window lengths, in bars. Each one names the window its feature
// averages or looks back over.
//
// The guards in Build are not uniformly "i >= w": a window of w *values* needs
// only i >= w-1, while a window of w *returns* needs i >= w, because the return
// series starts at index 1 — ret_1 at bar i needs bar i-1. The value/return
// distinction is the only reason two features with the same window length are
// first defined on different rows, and it is why the guard for each is written
// out next to the feature rather than derived from a shared formula.
const (
	windowRet1  = 1
	windowRet5  = 5
	windowRet15 = 15
	windowRet60 = 60
	windowVol5  = 5
	windowVol20 = 20
	windowVolZ  = 20
	windowRSI   = 14
	windowSMA   = 10
)

// WarmupRows is the number of leading rows of a FeatureTable that are
// context-only and must never be used as a prediction target. It is exactly
// windowRet60, the longest window in the schema, so row WarmupRows is the first
// row where every feature has its full history.
//
// The constant exists so callers do not have to know the arithmetic: the
// inference service takes its targets as table.Rows[features.WarmupRows:].
const WarmupRows = windowRet60

// Builder turns a candle window into a model-ready feature table.
type Builder interface {
	// Build returns one feature row per input candle, newest last. Row i
	// corresponds to candles[i]; see the package comment for the warm-up and
	// alignment rules.
	Build(candles []contracts.Candle) (FeatureTable, error)
}

// FeatureTable is a dense numeric matrix with named columns, ready to be
// handed to a model. Rows[i] aligns with the input candle at index i.
type FeatureTable struct {
	// Columns holds the ordered feature names. It is always DefaultColumns()
	// for a table this package builds.
	Columns []string
	// Rows holds one row of len(Columns) values per input candle, in input
	// order.
	Rows [][]float64
	// AsOf is the timestamp of the last row, taken verbatim from the last
	// candle so it carries the caller's zone.
	AsOf time.Time
}

// IsWarmup reports whether row i is a context-only warm-up row. It is true for
// i < WarmupRows and false from there on, which is the exact set of rows Build
// leaves under-informed.
func IsWarmup(i int) bool {
	return i < WarmupRows
}

// DefaultColumns returns the frozen 18 feature names in their fixed order. The
// inference service calls it to assert that the table it received is the table
// TabFM was trained against.
//
// A fresh slice is returned on every call, so a caller that reorders or
// truncates the result cannot corrupt the schema for anyone else.
func DefaultColumns() []string {
	out := make([]string, len(defaultColumns))
	copy(out, defaultColumns[:])
	return out
}

// builder is the default Builder. It holds nothing but the trading location
// used for the two calendar features; everything else is derived per Build from
// the candles themselves.
type builder struct {
	loc *time.Location
}

// NewBuilder returns a Builder that labels bars with the trading location loc.
//
// The location is injected, never hardcoded, because the candles are stored in
// UTC while minute_of_session and hour_of_day are IST calendar values and the
// two disagree for 23 of every 24 hours. Pass the configured app.timezone —
// time.LoadLocation("Asia/Kolkata") — and the feature follows it.
//
// A nil location means UTC, which is the honest reading of an unset value and
// not a safe default: it makes both calendar features UTC-derived, and a UTC
// minute_of_session for the 09:15 IST open reads -345 instead of 0.
func NewBuilder(loc *time.Location) Builder {
	if loc == nil {
		loc = time.UTC
	}
	return &builder{loc: loc}
}

// location reports the trading location, tolerating a nil receiver and a
// builder that was never constructed.
func (b *builder) location() *time.Location {
	if b == nil || b.loc == nil {
		return time.UTC
	}
	return b.loc
}

// Build implements Builder. It is deterministic and allocates only; it reads
// no clock and touches no file.
//
// Errors are returned for a window too short to warm up and for input that is
// not a well-formed single-symbol run of ascending one-minute bars. There is no
// error path for a degenerate market — flat windows, zero volume and doji bars
// produce zeros, not failures.
func (b *builder) Build(candles []contracts.Candle) (FeatureTable, error) {
	if len(candles) < WarmupRows {
		return FeatureTable{}, fmt.Errorf("features: %d candles is under the %d-bar warm-up, refusing to build a table with no predictable row",
			len(candles), WarmupRows)
	}
	if err := validate(candles); err != nil {
		return FeatureTable{}, err
	}

	loc := b.location()
	s := newSeries(candles)
	rows := make([][]float64, len(candles))

	for i := range candles {
		row := make([]float64, numColumns)

		// Returns. ret_1 is the only one whose window is a single bar, and it
		// is first defined at bar 1; ret_60 at bar 60, which is exactly where
		// the warm-up ends.
		if i >= windowRet1 {
			row[colRet1] = s.ret1(i)
		}
		if i >= windowRet5 {
			row[colRet5] = s.ret(windowRet5, i)
		}
		if i >= windowRet15 {
			row[colRet15] = s.ret(windowRet15, i)
		}
		if i >= windowRet60 {
			row[colRet60] = s.ret(windowRet60, i)
		}

		// Realised volatility and its ratio.
		if i >= windowVol5 {
			row[colVol5] = s.vol(windowVol5, i)
		}
		if i >= windowVol20 {
			row[colVol20] = s.vol(windowVol20, i)
			// vol_5 is already populated here: the 20-bar guard is stricter
			// than the 5-bar one it depends on.
			row[colVolRatio] = div(row[colVol5], row[colVol20])
		}

		// Candle geometry. Every one of these reads only the bar it belongs
		// to, so all of them are defined from bar 0.
		row[colRange1] = s.range1(i)
		row[colBody1] = s.body1(i)
		row[colUpperWick1] = s.upperWick1(i)
		row[colLowerWick1] = s.lowerWick1(i)
		row[colSpreadProxy] = s.spreadProxy(i)

		// Volume. The 20-bar window counts the current bar, the same window
		// volume_ratio's moving average does.
		if i >= windowVolZ-1 {
			row[colVolumeZ20] = s.volumeZ20(i)
			row[colVolumeRatio] = s.volumeRatio(i)
		}

		// Momentum and trend.
		if i >= windowRSI {
			row[colMomentumRSI14] = s.rsi14(i)
		}
		if i >= windowSMA-1 {
			row[colSMAGap10] = s.smaGap10(i)
		}

		// Session clock, in the injected trading location.
		row[colMinuteOfSession] = minuteOfSession(candles[i].Timestamp, loc)
		row[colHourOfDay] = hourOfDay(candles[i].Timestamp, loc)

		for j := range row {
			row[j] = sanitize(row[j])
		}
		rows[i] = row
	}

	return FeatureTable{
		Columns: DefaultColumns(),
		Rows:    rows,
		AsOf:    candles[len(candles)-1].Timestamp,
	}, nil
}

// validate rejects input that would make the table confidently wrong.
//
// The checks are the ones whose failure is invisible downstream: a window that
// mixes two symbols averages two unrelated price series into a plausible-looking
// return; a window that repeats or rewinds a timestamp double-counts a bar;
// an inconsistent OHLC bar yields a wick measured from a high that never
// happened. Every one of those produces finite numbers, so nothing but an
// explicit check stops it reaching the model.
//
// Gaps are not a failure. A missing minute is a fact about the market, and it
// leaves the trailing windows shorter than the clock would suggest — which is
// exactly what a fixed-window feature should do.
func validate(candles []contracts.Candle) error {
	symbol := candles[0].Symbol
	for i := range candles {
		c := &candles[i]
		at := c.Timestamp.Format(time.RFC3339)

		if c.Symbol != symbol {
			return fmt.Errorf("features: candle %d is %q but the window opened with %q: a feature window must hold one symbol",
				i, c.Symbol, symbol)
		}
		if i > 0 && !c.Timestamp.After(candles[i-1].Timestamp) {
			return fmt.Errorf("features: candle %d at %s does not follow %s: candles must ascend strictly in time",
				i, at, candles[i-1].Timestamp.Format(time.RFC3339))
		}
		if !finite(c.Open) || !finite(c.High) || !finite(c.Low) || !finite(c.Close) {
			return fmt.Errorf("features: candle %d at %s has a non-finite price", i, at)
		}
		if c.Open <= 0 || c.High <= 0 || c.Low <= 0 || c.Close <= 0 {
			return fmt.Errorf("features: candle %d at %s has a non-positive price: open=%g high=%g low=%g close=%g",
				i, at, c.Open, c.High, c.Low, c.Close)
		}
		if c.High < c.Low || c.High < c.Open || c.High < c.Close || c.Low > c.Open || c.Low > c.Close {
			return fmt.Errorf("features: candle %d at %s is not a well-formed OHLC bar: open=%g high=%g low=%g close=%g",
				i, at, c.Open, c.High, c.Low, c.Close)
		}
		if c.Volume < 0 {
			return fmt.Errorf("features: candle %d at %s has negative volume %d", i, at, c.Volume)
		}
	}
	return nil
}
