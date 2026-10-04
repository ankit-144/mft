// Package features turns a window of 1-minute candles into the frozen 18-column feature
// table that TabFM consumes (docs/contracts.md §4).
package features

import (
	"fmt"
	"time"

	"github.com/mft/core/contracts"
)

// defaultColumns is the frozen schema of docs/contracts.md §4: 18 names, in the order
// TabFM was given them.
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

// Column indices into a FeatureTable row.
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

// Trailing window lengths, in bars.
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

// WarmupRows is the number of leading rows of a FeatureTable that are context-only and
// must never be used as a prediction target.
const WarmupRows = windowRet60

// Builder turns a candle window into a model-ready feature table.
type Builder interface {
	// Build returns one feature row per input candle, newest last.
	Build(candles []contracts.Candle) (FeatureTable, error)
}

// FeatureTable is a dense numeric matrix with named columns, ready to be handed to a
// model.
type FeatureTable struct {
	// Columns holds the ordered feature names.
	Columns []string
	// Rows holds one row of len(Columns) values per input candle, in input order.
	Rows [][]float64
	// AsOf is the timestamp of the last row, taken verbatim from the last candle so it
	// carries the caller's zone.
	AsOf time.Time
}

// IsWarmup reports whether row i is a context-only warm-up row.
func IsWarmup(i int) bool {
	return i < WarmupRows
}

// DefaultColumns returns the frozen 18 feature names in their fixed order.
func DefaultColumns() []string {
	out := make([]string, len(defaultColumns))
	copy(out, defaultColumns[:])
	return out
}

// builder is the default Builder.
type builder struct {
	loc *time.Location
}

// NewBuilder returns a Builder that labels bars with the trading location loc.
func NewBuilder(loc *time.Location) Builder {
	if loc == nil {
		loc = time.UTC
	}
	return &builder{loc: loc}
}

// location reports the trading location, tolerating a nil receiver and a builder that
// was never constructed.
func (b *builder) location() *time.Location {
	if b == nil || b.loc == nil {
		return time.UTC
	}
	return b.loc
}

// Build implements Builder.
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

		if i >= windowVol5 {
			row[colVol5] = s.vol(windowVol5, i)
		}
		if i >= windowVol20 {
			row[colVol20] = s.vol(windowVol20, i)

			row[colVolRatio] = div(row[colVol5], row[colVol20])
		}

		row[colRange1] = s.range1(i)
		row[colBody1] = s.body1(i)
		row[colUpperWick1] = s.upperWick1(i)
		row[colLowerWick1] = s.lowerWick1(i)
		row[colSpreadProxy] = s.spreadProxy(i)

		if i >= windowVolZ-1 {
			row[colVolumeZ20] = s.volumeZ20(i)
			row[colVolumeRatio] = s.volumeRatio(i)
		}

		if i >= windowRSI {
			row[colMomentumRSI14] = s.rsi14(i)
		}
		if i >= windowSMA-1 {
			row[colSMAGap10] = s.smaGap10(i)
		}

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
