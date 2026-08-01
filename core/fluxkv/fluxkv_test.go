package fluxkv

import (
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

	kv.UpdateCandle("RELIANCE", base, 100, 1)          // open
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
