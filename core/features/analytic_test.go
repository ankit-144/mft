package features

import (
	"fmt"
	"math"
	"testing"
)

// TestTrendSeriesMatchesClosedForm is the analytic golden: a series that rises
// by exactly 1% per bar under a fixed OHLC envelope at constant volume, so
// every column reduces to a closed form a reviewer can check with a calculator.
//
//	close[i]  = 100 * 1.01^i
//	open[i]   = close[i] * 0.999
//	high[i]   = close[i] * 1.002
//	low[i]    = close[i] * 0.998
//	volume[i] = 500
//
// which, with L = ln(1.01), gives at every row:
//
//	ret_k           = k * L                    (0 before the window is full)
//	vol_5, vol_20   = 0                        (every 1-bar return is the same)
//	range_1         = 0.004                    (1.002 - 0.998)
//	body_1          = 0.001 / 0.999
//	upper_wick_1    = (1.002 - 1) / 0.004      = 0.5
//	lower_wick_1    = (0.999 - 0.998) / 0.004  = 0.25
//	volume_z_20     = 0                        (constant volume)
//	volume_ratio    = 1
//	momentum_rsi_14 = 1                        (every change is up)
//	sma_gap_10      = (1 - m) / m, m = mean of 1.01^-k for k in 0..9
//	spread_proxy    = 0.001 * close[i] / 500
//	minute_of_session = i                      (the series starts at 09:15 IST)
//	hour_of_day     = (555 + i) / 60
//
// vol_ratio is the one column absent from that list, on purpose. This fixture
// has no dispersion to divide, so both volatility columns are floating-point
// noise around zero and their ratio is arbitrary; pinning it here would pin
// noise. TestVolRatioOnAlternatingReturns pins it in closed form and
// TestMixedSeriesMatchesGolden pins it on a real price path.
func TestTrendSeriesMatchesClosedForm(t *testing.T) {
	const bars = 64
	candles := geometricCandles(bars, 100, 1.01)

	b := NewBuilder(tradingZone(t))
	table, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	l := math.Log(1.01)
	// m is the mean of 1.01^-k over a 10-bar window, which is the closed-form
	// geometric sum: the 10-bar average of a geometric ramp scales with the
	// ramp itself, so sma_gap_10 is the same at every row.
	r := 1 / 1.01
	m := (1 - math.Pow(r, windowSMA)) / (windowSMA * (1 - r))
	gap := (1 - m) / m

	// retAt is the k-bar log return of a 1% ramp, zero while its window is
	// still filling.
	retAt := func(k, i int) float64 {
		if i < k {
			return 0
		}
		return float64(k) * l
	}
	// after is the warm-up rule as an expectation: 0 until the window of
	// windowSize values is full, and want after that.
	after := func(i, windowSize int, want float64) float64 {
		if i < windowSize {
			return 0
		}
		return want
	}

	const (
		wantRange     = 0.004
		wantBody      = 0.001 / 0.999
		wantUpperWick = 0.5
		wantLowerWick = 0.25
	)

	columns := []struct {
		name string
		at   func(i int) float64
	}{
		{name: "ret_1", at: func(i int) float64 { return retAt(1, i) }},
		{name: "ret_5", at: func(i int) float64 { return retAt(5, i) }},
		{name: "ret_15", at: func(i int) float64 { return retAt(15, i) }},
		{name: "ret_60", at: func(i int) float64 { return retAt(60, i) }},
		{name: "vol_5", at: func(int) float64 { return 0 }},
		{name: "vol_20", at: func(int) float64 { return 0 }},
		{name: "range_1", at: func(int) float64 { return wantRange }},
		{name: "body_1", at: func(int) float64 { return wantBody }},
		{name: "upper_wick_1", at: func(int) float64 { return wantUpperWick }},
		{name: "lower_wick_1", at: func(int) float64 { return wantLowerWick }},
		{name: "volume_z_20", at: func(i int) float64 { return after(i, 19, 0) }},
		{name: "volume_ratio", at: func(i int) float64 { return after(i, 19, 1) }},
		{name: "momentum_rsi_14", at: func(i int) float64 { return after(i, 14, 1) }},
		{name: "sma_gap_10", at: func(i int) float64 { return after(i, 9, gap) }},
		{name: "spread_proxy", at: func(i int) float64 { return 0.001 * candles[i].Close / 500 }},
		{name: "minute_of_session", at: func(i int) float64 { return float64(i) }},
		{name: "hour_of_day", at: func(i int) float64 { return float64((9*60 + 15 + i) / 60) }},
	}

	for i := range candles {
		for _, c := range columns {
			assertClose(t, col(t, table, i, c.name), c.at(i), fmt.Sprintf("row %d %s", i, c.name))
		}
	}
}
