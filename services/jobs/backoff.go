package jobs

import (
	"math"
	"math/rand"
	"time"
)

// Backoff defaults.
const (
	defaultBackoffBase = 500 * time.Millisecond
	defaultBackoffMax  = 30 * time.Second
)

// Backoff is an exponential retry schedule with full jitter.
type Backoff struct {
	Base   time.Duration
	Max    time.Duration
	Jitter func(time.Duration) time.Duration
}

// DefaultBackoff returns the schedule backfill uses when config is silent.
func DefaultBackoff() Backoff {
	return Backoff{Base: defaultBackoffBase, Max: defaultBackoffMax, Jitter: FullJitter}
}

// Delay returns how long to wait before retry number attempt, counting the first retry
// as attempt zero.
func (b Backoff) Delay(attempt int) time.Duration {
	base := b.Base
	if base <= 0 {
		base = defaultBackoffBase
	}
	max := b.Max
	if max <= 0 {
		max = defaultBackoffMax
	}
	if attempt < 0 {
		attempt = 0
	}

	d := float64(base) * math.Pow(2, float64(attempt))
	if d >= float64(max) || math.IsInf(d, 1) {
		d = float64(max)
	}
	out := time.Duration(d)
	if b.Jitter == nil {
		return out
	}
	jittered := b.Jitter(out)
	if jittered < 0 {
		return 0
	}
	return jittered
}

// FullJitter returns a random duration in [d/2, d].
func FullJitter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	half := d / 2
	if half <= 0 {
		return d
	}
	return half + time.Duration(rand.Int63n(int64(half)+1))
}
