package jobs

import (
	"context"
	"sync"
	"testing"
	"time"
)

// step is the resolution at which the token-bucket tests advance the simulated
// clock. Nothing sleeps: TakeAt is handed a time, so a test that walks a whole
// minute of a window takes microseconds and cannot be flaky.
//
// 125ms is a power of two in seconds, so elapsed*rate is exact in float64 for
// every rate these tests use. A step like 10ms would accumulate rounding error
// and make a window-boundary assertion depend on which way the error fell.
const step = 125 * time.Millisecond

// grantedIn walks [from, to] inclusive in step-sized ticks and returns how
// many tokens the bucket actually handed out. The window is closed at both
// ends on purpose: a one-second window at rate 3 has to be measured at
// t+1s exactly, not at t+990ms, or the assertion silently drifts by a token.
func grantedIn(t *testing.T, l *Limiter, from, to time.Time) int {
	t.Helper()
	n := 0
	for ts := from; !ts.After(to); ts = ts.Add(step) {
		if l.TakeAt(ts) {
			n++
		}
	}
	return n
}

func TestLimiterGrantsBurstThenThrottles(t *testing.T) {
	l := newLimiter(3, 3)
	start := time.Unix(1_700_000_000, 0).UTC()

	// The bucket starts full, so the first three calls ride the burst and the
	// fourth has to wait for a refill.
	for i, want := range []bool{true, true, true, false, false} {
		if got := l.TakeAt(start); got != want {
			t.Fatalf("TakeAt #%d = %v, want %v", i, got, want)
		}
	}
}

func TestLimiterGrantsExactlyRateTokensPerWindow(t *testing.T) {
	l := newLimiter(3, 3)
	start := time.Unix(1_700_000_000, 0).UTC()

	// Drain the burst, then measure a closed one-second window that contains
	// no burst: the bucket must hand out exactly rate tokens in it, no more
	// and no fewer.
	for range 3 {
		l.TakeAt(start)
	}
	if n := grantedIn(t, l, start, start.Add(time.Second)); n != 3 {
		t.Fatalf("granted %d tokens in a 1s window at rate 3, want 3", n)
	}
}

func TestLimiterRefillsContinuouslyAcrossSubSecondWindows(t *testing.T) {
	// Refill is continuous, not per-second: a whole-second-step implementation
	// would grant nothing in this 500ms window.
	l := newLimiter(6, 6)
	start := time.Unix(1_700_000_000, 0).UTC()

	for range 6 {
		l.TakeAt(start)
	}
	if n := grantedIn(t, l, start, start.Add(500*time.Millisecond)); n != 3 {
		t.Fatalf("granted %d tokens in 500ms at rate 6, want 3", n)
	}
}

func TestLimiterRefillIsCappedAtBurst(t *testing.T) {
	// An idle bucket must not accumulate more than a burst of quota, or a
	// long-idle backfill would open with a request storm.
	l := newLimiter(3, 3)
	start := time.Unix(1_700_000_000, 0).UTC()

	for range 3 {
		l.TakeAt(start)
	}
	// Ten idle minutes is 1800 tokens' worth of refill, all discarded.
	idle := start.Add(10 * time.Minute)
	for i := range 3 {
		if !l.TakeAt(idle) {
			t.Fatalf("TakeAt #%d after a long idle was denied, want the burst of 3", i)
		}
	}
	if l.TakeAt(idle) {
		t.Fatal("TakeAt granted past the burst after a long idle period")
	}
}

func TestLimiterHigherBurstAllowsLargerInitialBurst(t *testing.T) {
	l := newLimiter(3, 10)
	start := time.Unix(1_700_000_000, 0).UTC()

	for i := range 10 {
		if !l.TakeAt(start) {
			t.Fatalf("TakeAt #%d denied, want the burst of 10 to cover it", i)
		}
	}
	if l.TakeAt(start) {
		t.Fatal("TakeAt #10 granted, want the bucket to be empty")
	}
}

func TestLimiterUnlimitedRateAlwaysGrants(t *testing.T) {
	// A non-positive rate disables limiting rather than deadlocking the job.
	l := NewLimiter(0)
	start := time.Unix(1_700_000_000, 0).UTC()

	if n := grantedIn(t, l, start, start.Add(time.Hour)); n != int(time.Hour/step)+1 {
		t.Fatalf("granted %d tokens, want every tick", n)
	}
}

func TestLimiterRateReportsConfiguredValue(t *testing.T) {
	if got := NewLimiter(7).Rate(); got != 7 {
		t.Fatalf("Rate() = %v, want 7", got)
	}
}

func TestLimiterTakeBlocksUntilTheBucketRefills(t *testing.T) {
	// A high rate keeps the refill sub-millisecond, so this exercises the
	// blocking path without spending wall-clock time.
	l := NewLimiter(1000)
	ctx := context.Background()

	for range 1000 {
		if err := l.Take(ctx); err != nil {
			t.Fatalf("Take() error = %v", err)
		}
	}
	if err := l.Take(ctx); err != nil {
		t.Fatalf("Take() after the bucket drained: error = %v", err)
	}
}

func TestLimiterTakeRespectsCancelledContext(t *testing.T) {
	l := newLimiter(1, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := l.Take(ctx); err == nil {
		t.Fatal("Take() on a cancelled context returned nil, want ctx.Err()")
	}
}

func TestLimiterIsSafeForConcurrentUse(t *testing.T) {
	l := newLimiter(1000, 1000)
	start := time.Unix(1_700_000_000, 0).UTC()

	var wg sync.WaitGroup
	for i := range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = l.TakeAt(start.Add(time.Duration(i) * time.Microsecond))
		}()
	}
	wg.Wait()
}

func TestLimiterRetryAfterShrinksAsTheBucketFills(t *testing.T) {
	l := newLimiter(2, 2)
	start := time.Unix(1_700_000_000, 0).UTC()

	l.TakeAt(start)
	l.TakeAt(start)
	if got := l.retryAfter(); got != 500*time.Millisecond {
		t.Fatalf("retryAfter() on an empty 2/s bucket = %v, want 500ms", got)
	}

	// A quarter-second of refill leaves half a token, so the wait halves. The
	// take is denied, but the refill it performed is what is being measured.
	l.TakeAt(start.Add(250 * time.Millisecond))
	if got := l.retryAfter(); got != 250*time.Millisecond {
		t.Fatalf("retryAfter() after a quarter-second of refill = %v, want 250ms", got)
	}
}
