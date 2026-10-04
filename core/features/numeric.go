package features

import "math"

// finite reports whether v is an ordinary finite number: neither NaN nor an infinity.
func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// sanitize applies the numeric policy to a computed feature: a value that is not finite
// is replaced by 0.
func sanitize(v float64) float64 {
	if !finite(v) {
		return 0
	}
	return v
}

// div returns num/den, or 0 when den is 0 or the quotient is not finite.
func div(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return sanitize(num / den)
}

// mean returns the arithmetic mean of xs[lo:hi+1] inclusive, or 0 for an empty range.
func mean(xs []float64, lo, hi int) float64 {
	if hi < lo {
		return 0
	}
	var sum float64
	for i := lo; i <= hi; i++ {
		sum += xs[i]
	}
	return sum / float64(hi-lo+1)
}

// stdev returns the sample standard deviation of xs[lo:hi+1] inclusive, with the n-1
// (Bessel) denominator, or 0 for a range of fewer than two values.
func stdev(xs []float64, lo, hi int) float64 {
	n := hi - lo + 1
	if n < 2 {
		return 0
	}
	mu := mean(xs, lo, hi)
	var ss float64
	for i := lo; i <= hi; i++ {
		d := xs[i] - mu
		ss += d * d
	}
	return math.Sqrt(ss / float64(n-1))
}

// zscore returns (x - mean(xs[lo:hi+1])) / stdev(xs[lo:hi+1]), or 0 when the window has
// no dispersion.
func zscore(x float64, xs []float64, lo, hi int) float64 {
	return div(x-mean(xs, lo, hi), stdev(xs, lo, hi))
}
