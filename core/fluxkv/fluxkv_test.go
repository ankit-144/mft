package fluxkv

import (
	"sync"
	"testing"
	"time"
)

func TestKVSetGet(t *testing.T) {
	kv := New()
	kv.Set("key", "value", time.Minute)

	got, ok := kv.Get("key")
	if !ok {
		t.Fatal("expected key to exist")
	}
	if got != "value" {
		t.Fatalf("expected value %q, got %v", "value", got)
	}
}

func TestKVGetExpired(t *testing.T) {
	kv := New()
	kv.Set("key", "value", -time.Second)

	if _, ok := kv.Get("key"); ok {
		t.Fatal("expected expired key to be absent")
	}
}

func TestKVDelete(t *testing.T) {
	kv := New()
	kv.Set("key", "value", time.Minute)
	kv.Delete("key")

	if _, ok := kv.Get("key"); ok {
		t.Fatal("expected deleted key to be absent")
	}
}

func TestUpdateCandleAggregatesWithinMinute(t *testing.T) {
	kv := New()
	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)

	kv.UpdateCandle("RELIANCE", base, 100, 1)                     // open
	kv.UpdateCandle("RELIANCE", base.Add(10*time.Second), 110, 1) // high
	kv.UpdateCandle("RELIANCE", base.Add(20*time.Second), 90, 1)  // low
	kv.UpdateCandle("RELIANCE", base.Add(30*time.Second), 105, 1) // close

	c := kv.Candle("RELIANCE")
	if c == nil {
		t.Fatal("expected candle to exist")
	}
	if c.Open != 100 || c.High != 110 || c.Low != 90 || c.Close != 105 {
		t.Fatalf("unexpected OHLC: %+v", c)
	}
	if c.Volume != 4 {
		t.Fatalf("expected volume 4, got %d", c.Volume)
	}
}

func TestUpdateCandleRollsOverMinute(t *testing.T) {
	kv := New()
	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)

	kv.UpdateCandle("RELIANCE", base, 100, 1)
	kv.UpdateCandle("RELIANCE", base.Add(time.Minute), 200, 1)

	c := kv.Candle("RELIANCE")
	if c == nil {
		t.Fatal("expected candle to exist")
	}
	if c.Open != 200 || c.Close != 200 {
		t.Fatalf("expected fresh candle after rollover, got %+v", c)
	}
	if c.Volume != 1 {
		t.Fatalf("expected volume 1 after rollover, got %d", c.Volume)
	}
}

func TestCandlesSnapshot(t *testing.T) {
	kv := New()
	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)

	kv.UpdateCandle("A", base, 10, 1)
	kv.UpdateCandle("B", base, 20, 1)

	got := kv.Candles()
	if len(got) != 2 {
		t.Fatalf("expected 2 candles, got %d", len(got))
	}
}

func TestSweepRemovesOnlyExpiredEntries(t *testing.T) {
	kv := New()
	now := time.Now()
	kv.Set("dead", 1, -time.Second)
	kv.Set("live", 2, time.Hour)

	if got := kv.Sweep(now); got != 1 {
		t.Fatalf("Sweep() = %d, want 1", got)
	}
	if _, ok := kv.Get("dead"); ok {
		t.Fatal("the expired entry should be gone")
	}
	if _, ok := kv.Get("live"); !ok {
		t.Fatal("the live entry should have survived the sweep")
	}
	if got := kv.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}
}

func TestSweepLeavesCandlesAlone(t *testing.T) {
	kv := New()
	kv.UpdateCandle("RELIANCE", time.Now(), 100, 1)

	kv.Sweep(time.Now())

	if kv.Candle("RELIANCE") == nil {
		t.Fatal("sweeping the TTL store must not drop candles")
	}
}

func TestDeleteRemovesCandles(t *testing.T) {
	kv := New()
	kv.UpdateCandle("RELIANCE", time.Now(), 100, 1)
	kv.Set("EXEC:RELIANCE:BUY", true, time.Hour)

	kv.Delete("RELIANCE")
	kv.Delete("EXEC:RELIANCE:BUY")

	if kv.Candle("RELIANCE") != nil {
		t.Fatal("Delete should have removed the candle")
	}
	if _, ok := kv.Get("EXEC:RELIANCE:BUY"); ok {
		t.Fatal("Delete should have removed the TTL entry")
	}
}

// TestCandleReturnsACopy is the regression test for the aliasing bug: a
// caller holding a candle pointer could write through it into the map while
// the producer was folding ticks in under the lock.
func TestCandleReturnsACopy(t *testing.T) {
	kv := New()
	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)
	kv.UpdateCandle("RELIANCE", base, 100, 1)

	snapshot := kv.Candle("RELIANCE")
	if snapshot == nil {
		t.Fatal("expected a candle")
	}
	snapshot.Close = 9999
	snapshot.Volume = 9999

	live := kv.Candle("RELIANCE")
	if live.Close != 100 || live.Volume != 1 {
		t.Fatalf("writing to a snapshot reached the store: %+v", live)
	}
}

func TestUpdateCandleReturnsACopy(t *testing.T) {
	kv := New()
	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)

	first := kv.UpdateCandle("RELIANCE", base, 100, 1)
	first.High = 9999

	if got := kv.Candle("RELIANCE").High; got != 100 {
		t.Fatalf("the value returned by UpdateCandle aliases the store: high = %v", got)
	}
}

func TestCandlesSnapshotIsDetached(t *testing.T) {
	kv := New()
	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)
	kv.UpdateCandle("A", base, 10, 1)
	kv.UpdateCandle("B", base, 20, 1)

	for _, c := range kv.Candles() {
		c.Close = -1
	}
	for symbol, want := range map[string]float64{"A": 10, "B": 20} {
		if got := kv.Candle(symbol).Close; got != want {
			t.Fatalf("%s close = %v, want %v", symbol, got, want)
		}
	}
}

// TestKVConcurrentAccess is the -race workhorse: producers fold ticks while
// readers take snapshots and sweep, which is what the ingestion and execution
// services actually do to one cache in one process.
func TestKVConcurrentAccess(t *testing.T) {
	kv := New()
	const symbols = 8

	var wg sync.WaitGroup
	for s := range symbols {
		wg.Add(1)
		go func() {
			defer wg.Done()
			symbol := string(rune('A' + s))
			base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)
			for i := range 500 {
				kv.UpdateCandle(symbol, base.Add(time.Duration(i)*time.Minute), float64(100+i), 1)
				if i%50 == 0 {
					kv.Set("EXEC:"+symbol, i, time.Hour)
				}
			}
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			symbol := string(rune('A' + s))
			for range 500 {
				_ = kv.Candle(symbol)
				_ = kv.Candles()
				_, _ = kv.Get("EXEC:" + symbol)
				kv.Sweep(time.Now())
				kv.Len()
			}
		}()
	}
	wg.Wait()
}
