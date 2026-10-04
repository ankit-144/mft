package features

import (
	"math"

	"github.com/mft/core/contracts"
)

// series is the columnar view of a candle window that every feature reads.
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
func (s *series) ret1(i int) float64 {
	return s.logRet[i]
}

// ret returns the k-bar log return ending at i:
func (s *series) ret(k, i int) float64 {
	return math.Log(s.closes[i] / s.closes[i-k])
}

// vol returns the sample standard deviation of the last n 1-bar log returns, i.e.
func (s *series) vol(n, i int) float64 {
	return stdev(s.logRet, i-n+1, i)
}

// range1 returns the bar's high-low range as a fraction of its close:
func (s *series) range1(i int) float64 {
	return div(s.highs[i]-s.lows[i], s.closes[i])
}

// body1 returns the signed body as a fraction of the open:
func (s *series) body1(i int) float64 {
	return div(s.closes[i]-s.opens[i], s.opens[i])
}

// upperWick1 returns the upper wick as a fraction of the bar's full range:
func (s *series) upperWick1(i int) float64 {
	body := math.Max(s.opens[i], s.closes[i])
	return div(s.highs[i]-body, s.highs[i]-s.lows[i])
}

// lowerWick1 returns the lower wick as a fraction of the bar's full range:
func (s *series) lowerWick1(i int) float64 {
	body := math.Min(s.opens[i], s.closes[i])
	return div(body-s.lows[i], s.highs[i]-s.lows[i])
}

// volumeZ20 returns the bar's volume as a z-score against the trailing 20-bar volume
// window, which includes the bar itself:
func (s *series) volumeZ20(i int) float64 {
	return zscore(s.volumes[i], s.volumes, i-windowVolZ+1, i)
}

// volumeRatio returns the bar's volume over the trailing 20-bar mean volume, itself
// including the bar:
func (s *series) volumeRatio(i int) float64 {
	return div(s.volumes[i], mean(s.volumes, i-windowVolZ+1, i))
}

// rsi14 returns RSI(14) normalised to [-1, 1]:
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
		rsi = 50
	case avgLoss == 0:
		rsi = 100
	default:
		rsi = 100 - 100/(1+avgGain/avgLoss)
	}
	return 2*rsi/100 - 1
}

// smaGap10 returns the close's distance from its own 10-bar simple moving average, as a
// fraction of that average:
func (s *series) smaGap10(i int) float64 {
	sma := mean(s.closes, i-windowSMA+1, i)
	return div(s.closes[i]-sma, sma)
}

// spreadProxy returns the bar's absolute body over its traded volume:
func (s *series) spreadProxy(i int) float64 {
	return div(math.Abs(s.closes[i]-s.opens[i]), s.volumes[i])
}
