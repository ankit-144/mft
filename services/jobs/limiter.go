package jobs

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// defaultBurstDivisor converts a per-second rate into a bucket capacity: a rate of 3
// requests per second gets a burst of 3, so a client that has been idle can spend a
// full second of quota at once but no more.
const defaultBurstDivisor = 1

// Limiter is a token bucket that paces calls to the broker historical API.
type Limiter struct {
	mu sync.Mutex

	rate  float64
	burst float64

	tokens float64
	last   time.Time
	now    func() time.Time
	sleep  func(ctx context.Context, d time.Duration) error
}

// NewLimiter returns a token bucket refilling at rate tokens per second.
func NewLimiter(rate float64) *Limiter {
	return newLimiter(rate, math.Ceil(rate/defaultBurstDivisor))
}

// newLimiter is NewLimiter with an explicit burst capacity, for tests and for callers
// that want a stricter or looser window than one second of quota.
func newLimiter(rate, burst float64) *Limiter {
	if rate <= 0 {
		rate = math.Inf(1)
	}
	if burst <= 0 {
		burst = rate
	}
	return &Limiter{
		rate:   rate,
		burst:  burst,
		tokens: burst,
		now:    time.Now,
		sleep:  sleepCtx,
	}
}

// Rate returns the configured refill rate in tokens per second.
func (l *Limiter) Rate() float64 { return l.rate }

// TakeAt refills the bucket as of now and consumes one token if one is available,
// reporting whether it was granted.
func (l *Limiter) TakeAt(now time.Time) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.last.IsZero() {
		l.last = now
	}
	if elapsed := now.Sub(l.last).Seconds(); elapsed > 0 {
		l.tokens += elapsed * l.rate
		if l.tokens > l.burst {
			l.tokens = l.burst
		}
		l.last = now
	}

	if l.tokens < 1 {
		return false
	}
	l.tokens--
	return true
}

// Take consumes one token, blocking until one is available or ctx is done.
func (l *Limiter) Take(ctx context.Context) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if l.TakeAt(l.now()) {
			return nil
		}
		if err := l.sleep(ctx, l.retryAfter()); err != nil {
			return err
		}
	}
}

// retryAfter returns how long to wait for the bucket to hold a full token.
func (l *Limiter) retryAfter() time.Duration {
	l.mu.Lock()
	defer l.mu.Unlock()
	if math.IsInf(l.rate, 1) || l.rate <= 0 {
		return 0
	}
	missing := 1 - l.tokens
	if missing <= 0 {
		return 0
	}
	return time.Duration(missing / l.rate * float64(time.Second))
}

// sleepCtx waits for d, or until ctx is done, whichever comes first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return fmt.Errorf("rate limiter: %w", ctx.Err())
	case <-t.C:
		return nil
	}
}
