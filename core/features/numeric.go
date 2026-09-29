package features

import "math"

// finite reports whether v is an ordinary finite number: neither NaN nor an
// infinity. Build refuses corrupt input rather than sanitising it, so this is
// the input guard, not the output policy.
func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// sanitize applies the numeric policy to a computed feature: a value that is
// not finite is replaced by 0.
//
// This is the last line of defence, not the first. Every feature is computed
// through div, which already returns 0 for a zero denominator, so reaching a
// non-finite value here means an intermediate overflowed. Dropping it to 0 keeps
// the table dense and model-consumable; a single NaN in a TabFM context table is
// not a degraded row, it is a poisoned one.
func sanitize(v float64) float64 {
	if !finite(v) {
		return 0
	}
	return v
}

// div returns num/den, or 0 when den is 0 or the quotient is not finite.
//
// A zero denominator is the normal case in this domain, not an exceptional
// one: a flat window has zero stdev, a bar that never traded has zero volume, a
// doji has zero range. Substituting 0 is the documented policy — see the package
// comment — and it is the right one because every one of those states means "no
// information", which is exactly what a zero column says to the model.
func div(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return sanitize(num / den)
}

// mean returns the arithmetic mean of xs[lo:hi+1] inclusive, or 0 for an empty
// range.
//
// Windows are summed per row rather than prefix-summed. A 60-bar mean is 60
// additions; prefix sums would make it O(1) but subtract two large nearly equal
// totals, and the rounding error of that subtraction grows with the length of
// the history. A feature whose value drifts as the history grows is a bug that
// looks like noise, so the cost is paid and the precision is kept.
//
// The summation itself is a plain left-to-right loop, not a compensated one.
// Over at most 60 prices or volumes the accumulated error is around 1e-13
// relative, far below anything a model can act on, and the loop is the version
// a reader can check against the formula in the contract.
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

// stdev returns the sample standard deviation of xs[lo:hi+1] inclusive, with
// the n-1 (Bessel) denominator, or 0 for a range of fewer than two values.
//
// The sample estimator, not the population one: it is the conventional meaning
// of "stdev" of a trailing sample of returns, and it is consistent with the
// volume z-score below, which shares this function.
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

// zscore returns (x - mean(xs[lo:hi+1])) / stdev(xs[lo:hi+1]), or 0 when the
// window has no dispersion. A window in which every value is identical is a
// constant, and the z-score of a point drawn from a constant is undefined; 0 is
// the neutral answer.
func zscore(x float64, xs []float64, lo, hi int) float64 {
	return div(x-mean(xs, lo, hi), stdev(xs, lo, hi))
}
