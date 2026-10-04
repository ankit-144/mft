package jobs

import (
	"testing"
	"time"
)

func TestBackoffDelayGrowsExponentiallyAndCaps(t *testing.T) {
	b := Backoff{Base: 100 * time.Millisecond, Max: 800 * time.Millisecond}

	want := []time.Duration{
		100 * time.Millisecond, // attempt 0
		200 * time.Millisecond, // attempt 1
		400 * time.Millisecond, // attempt 2
		800 * time.Millisecond, // attempt 3
		800 * time.Millisecond, // attempt 4, capped
		800 * time.Millisecond,
	}
	for i, w := range want {
		if got := b.Delay(i); got != w {
			t.Fatalf("Delay(%d) = %v, want %v", i, got, w)
		}
	}
}

func TestBackoffDelayClampsNegativeAttempt(t *testing.T) {
	b := Backoff{Base: time.Second, Max: time.Minute}
	if got, want := b.Delay(-5), time.Second; got != want {
		t.Fatalf("Delay(-5) = %v, want the base %v", got, want)
	}
}

func TestBackoffDelaySurvivesHugeAttemptCounts(t *testing.T) {
	// An exponent large enough to overflow time.Duration must clamp, not wrap
	// round to a negative (i.e. immediate) delay.
	b := Backoff{Base: time.Second, Max: 30 * time.Second}
	if got := b.Delay(200); got != 30*time.Second {
		t.Fatalf("Delay(200) = %v, want the 30s cap", got)
	}
}

func TestBackoffAppliesDefaultsForZeroFields(t *testing.T) {
	var b Backoff
	if got := b.Delay(0); got != defaultBackoffBase {
		t.Fatalf("Delay(0) on a zero Backoff = %v, want the %v default", got, defaultBackoffBase)
	}
	if got := b.Delay(20); got != defaultBackoffMax {
		t.Fatalf("Delay(20) on a zero Backoff = %v, want the %v cap", got, defaultBackoffMax)
	}
}

func TestDefaultBackoffIsCapped(t *testing.T) {
	b := DefaultBackoff()
	if got := b.Delay(30); got > b.Max {
		t.Fatalf("DefaultBackoff().Delay(30) = %v, above the %v cap", got, b.Max)
	}
}

func TestFullJitterStaysInTheUpperHalfOfTheWindow(t *testing.T) {
	const d = time.Second
	for range 500 {
		got := FullJitter(d)
		if got < d/2 || got > d {
			t.Fatalf("FullJitter(%v) = %v, want it in [%v, %v]", d, got, d/2, d)
		}
	}
}

func TestFullJitterSpreadsRatherThanCollapsing(t *testing.T) {
	// Without spread, a throttled fleet retries in lockstep and re-throttles
	// itself. Assert the sample actually varies.
	const d = time.Second
	seen := make(map[time.Duration]struct{}, 64)
	for range 200 {
		seen[FullJitter(d)] = struct{}{}
	}
	if len(seen) < 10 {
		t.Fatalf("FullJitter produced only %d distinct values in 200 draws, want a real spread", len(seen))
	}
}

func TestFullJitterHandlesNonPositiveDelay(t *testing.T) {
	if got := FullJitter(0); got != 0 {
		t.Fatalf("FullJitter(0) = %v, want 0", got)
	}
	if got := FullJitter(-time.Second); got != 0 {
		t.Fatalf("FullJitter(-1s) = %v, want 0", got)
	}
}

func TestBackoffDelayNeverNegativeWithAnAdversarialJitter(t *testing.T) {
	b := Backoff{Base: time.Second, Max: time.Minute, Jitter: func(time.Duration) time.Duration {
		return -time.Hour
	}}
	if got := b.Delay(1); got != 0 {
		t.Fatalf("Delay(1) with a negative jitter = %v, want 0", got)
	}
}

func TestBackoffDelayUsesTheJitter(t *testing.T) {
	b := Backoff{Base: time.Second, Max: time.Minute, Jitter: func(time.Duration) time.Duration {
		return 7 * time.Millisecond
	}}
	if got := b.Delay(3); got != 7*time.Millisecond {
		t.Fatalf("Delay(3) = %v, want the jittered 7ms", got)
	}
}
