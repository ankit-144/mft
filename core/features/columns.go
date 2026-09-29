package features

import (
	"math"

	"github.com/mft/core/contracts"
)

// series is the columnar view of a candle window that every feature reads.
//
// The OHLCV values are copied into parallel arrays once per Build because a
// feature reads them by offset rather than by value, and the two derived
// series — the 1-bar log return and the 1-bar simple change — are computed in
// the same pass so that a window of returns is a slice of a series rather than
// a loop that recomputes them.
//
// Index 0 of logRet and change is not meaningful: neither a return nor a change
// exists without a previous bar. It is left at 0 and every read of those two
// series is guarded by a window that starts at index 1 or later.
type series struct {
	opens   []float64
	highs   []float64
	lows    []float64
	closes  []float64
	volumes []float64
	logRet  []float64
	change  []float64
}

// newSeries copies a validated candle window into a series.
func newSeries(candles []contracts.Candle) *series {
	s := &series{
		opens:   make([]float64, len(candles)),
		highs:   make([]float64, len(candles)),
		lows:    make([]float64, len(candles)),
		closes:  make([]float64, len(candles)),
		volumes: make([]float64, len(candles)),
		logRet:  make([]float64, len(candles)),
		change:  make([]float64, len(candles)),
	}
	for i := range candles {
		c := &candles[i]
		s.opens[i] = c.Open
		s.highs[i] = c.High
		s.lows[i] = c.Low
		s.closes[i] = c.Close
		s.volumes[i] = float64(c.Volume)
		if i > 0 {
			s.logRet[i] = math.Log(c.Close / candles[i-1].Close)
			s.change[i] = c.Close - candles[i-1].Close
		}
	}
	return s
}

// ret1 returns the 1-bar log return ending at i:
//
//	log(close[i] / close[i-1])
//
// Log rather than a plain ratio: it is symmetric, so an up move and a down
// move of equal size have equal magnitude and opposite sign, and it composes
// over a window — log(close[i]/close[i-60]) is exactly the sum of the 60
// one-bar log returns between them.
func (s *series) ret1(i int) float64 {
	return s.logRet[i]
}

// ret returns the k-bar log return ending at i:
//
//	log(close[i] / close[i-k])
//
// Requires i >= k: there is no k-bar return without a bar k places back.
func (s *series) ret(k, i int) float64 {
	return math.Log(s.closes[i] / s.closes[i-k])
}

// vol returns the sample standard deviation of the last n 1-bar log returns,
// i.e. of logRet[i-n+1..i].
//
// Requires i >= n, the return series having no value at index 0.
//
// The window counts returns, not bars: n returns span n+1 closes. That is the
// reading of "stdev of 1-bar returns, n-bar window" that keeps the value a
// function of returns only, and it is why vol_5 first appears at bar 5.
func (s *series) vol(n, i int) float64 {
	return stdev(s.logRet, i-n+1, i)
}

// range1 returns the bar's high-low range as a fraction of its close:
//
//	(high[i] - low[i]) / close[i]
//
// Normalising by close is what makes the number comparable across a price level
// of 20 or 2000; the raw range is not a feature, it is a price.
func (s *series) range1(i int) float64 {
	return div(s.highs[i]-s.lows[i], s.closes[i])
}

// body1 returns the signed body as a fraction of the open:
//
//	(close[i] - open[i]) / open[i]
//
// Signed on purpose: the sign is the direction of the bar and the model should
// not have to recover it from a magnitude.
func (s *series) body1(i int) float64 {
	return div(s.closes[i]-s.opens[i], s.opens[i])
}

// upperWick1 returns the upper wick as a fraction of the bar's full range:
//
//	(high[i] - max(open[i], close[i])) / (high[i] - low[i])
//
// The wick is measured against the range, not the close, because range_1
// already carries the range in price-relative form. Splitting the range into
// upper, lower and body fractions this way makes the three an exact partition:
// they sum to 1 for any well-formed bar, so the model can read selling and
// buying pressure relative to each other without three separate scalings.
//
// A doji has zero range and yields 0 for all three, which is the honest
// answer: there was no wick to measure.
func (s *series) upperWick1(i int) float64 {
	body := math.Max(s.opens[i], s.closes[i])
	return div(s.highs[i]-body, s.highs[i]-s.lows[i])
}

// lowerWick1 returns the lower wick as a fraction of the bar's full range:
//
//	(min(open[i], close[i]) - low[i]) / (high[i] - low[i])
//
// As with upperWick1, the denominator is the range. See its comment.
func (s *series) lowerWick1(i int) float64 {
	body := math.Min(s.opens[i], s.closes[i])
	return div(body-s.lows[i], s.highs[i]-s.lows[i])
}

// volumeZ20 returns the bar's volume as a z-score against the trailing 20-bar
// volume window, which includes the bar itself:
//
//	(volume[i] - mean(volume[i-19..i])) / stdev(volume[i-19..i])
//
// The current bar is inside its own window so that the column reads as "how
// unusual is this bar for its recent history", the same convention
// volume_ratio uses. A window with no dispersion — constant volume over 20
// bars — yields 0 rather than an infinity.
func (s *series) volumeZ20(i int) float64 {
	return zscore(s.volumes[i], s.volumes, i-windowVolZ+1, i)
}

// volumeRatio returns the bar's volume over the trailing 20-bar mean volume,
// itself including the bar:
//
//	volume[i] / mean(volume[i-19..i])
//
// 1.0 means the bar is typical; 0 that it never traded. A window of nothing but
// zero volume yields 0, not a division by zero.
func (s *series) volumeRatio(i int) float64 {
	return div(s.volumes[i], mean(s.volumes, i-windowVolZ+1, i))
}

// rsi14 returns RSI(14) normalised to [-1, 1]:
//
//	avgGain = mean(max(close[j] - close[j-1], 0))  for j in (i-14, i]
//	avgLoss = mean(max(close[j-1] - close[j], 0))  for j in (i-14, i]
//	RSI     = 100 - 100 / (1 + avgGain/avgLoss)
//	result  = 2*RSI/100 - 1
//
// Requires i >= 14, the change series having no value at index 0.
//
// The averages are simple means over the last 14 changes, not Wilder's
// exponential smoothing. Wilder's variant seeds a running average at bar 14 and
// carries it forward, so its value at bar i depends on every bar back to 14 —
// and on where the caller's query started. A 100-candle pull and a 100000-bar
// backfill would then disagree about the same market state, which is precisely
// the drift the pull architecture in Plan.md §4 cannot tolerate.
//
// The mapping to [-1, 1] puts a flat window at 0, an unbroken run of up bars at
// +1 and an unbroken run of down bars at -1, matching the [-1, 1] score range
// the inference service reports. The all-up and all-down cases are written out
// because avgGain/avgLoss is an infinity there and the arithmetic below would
// otherwise have to rely on IEEE 754 to land on the right answer.
func (s *series) rsi14(i int) float64 {
	var gains, losses float64
	for j := i - windowRSI + 1; j <= i; j++ {
		switch d := s.change[j]; {
		case d > 0:
			gains += d
		case d < 0:
			losses -= d
		}
	}

	n := float64(windowRSI)
	avgGain, avgLoss := gains/n, losses/n

	var rsi float64
	switch {
	case avgGain == 0 && avgLoss == 0:
		rsi = 50 // no movement at all: neutral, not undefined
	case avgLoss == 0:
		rsi = 100 // every change was up
	default:
		rsi = 100 - 100/(1+avgGain/avgLoss)
	}
	return 2*rsi/100 - 1
}

// smaGap10 returns the close's distance from its own 10-bar simple moving
// average, as a fraction of that average:
//
//	(close[i] - mean(close[i-9..i])) / mean(close[i-9..i])
//
// Requires i >= 9. The window includes the current close, which is what makes
// the column comparable across price levels: a gap of 1% reads 0.01 whether the
// stock trades at 20 or at 2000.
func (s *series) smaGap10(i int) float64 {
	sma := mean(s.closes, i-windowSMA+1, i)
	return div(s.closes[i]-sma, sma)
}

// spreadProxy returns the bar's absolute body over its traded volume:
//
//	abs(close[i] - open[i]) / volume[i]
//
// A cost proxy, in rupees of price movement per share traded: the same body on
// a tenth of the volume is a wider effective spread and more slippage. It is
// zero for a bar that never traded, and it is deliberately unscaled — a raw
// ratio, not a fraction of anything, because there is no per-share turnover
// figure available to divide by instead.
func (s *series) spreadProxy(i int) float64 {
	return div(math.Abs(s.closes[i]-s.opens[i]), s.volumes[i])
}
