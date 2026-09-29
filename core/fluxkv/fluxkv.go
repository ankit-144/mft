// Package fluxkv implements an in-memory cache with TTL-based eviction and
// 1-minute candle aggregation. It is the hot-path state store for the MFT
// platform.
//
// # Scope: a per-process cache, never a bus
//
// fluxkv lives inside one process and one process only. Ingestion aggregates
// candles in the ingestion process; the execution engine debounces orders in
// the execution process. Neither can observe the other's keys, and that is
// deliberate (see Plan.md §4): cross-service data moves through Parquet via
// DuckDB and over HTTP, never through this map. What fluxkv holds is state
// that is correct only inside a single process and cheap to lose on restart —
// risk limits, debounce windows, idempotency keys, and the live candle per
// symbol.
//
// # Concurrency contract
//
// Every method is safe for concurrent use. Candles are the one value type the
// store mutates in place, so UpdateCandle folds a tick under the write lock
// and every read hands back a copy. A caller can never hold a pointer into
// the map; this is why Candle, Candles, and UpdateCandle all return values
// rather than the internal pointer.
//
// # Expiry
//
// A TTL is enforced on read: an expired entry is reported as absent. Set
// overwrites in place, so a hot key is never duplicated and the map is bounded
// by the number of distinct keys written — but one-shot keys (an order
// debounce that is never re-armed, an idempotency key that expires) linger as
// dead weight until swept. Sweep drops them; the execution service runs it
// periodically. Candles are not swept: they hold one entry per symbol and are
// overwritten every tick.
package fluxkv

import (
	"sync"
	"time"
)

type entry struct {
	value     any
	expiresAt time.Time
}

// Candle is an aggregated one-minute OHLCV candle.
type Candle struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    int64
}

// clone returns a detached copy of the candle. The zero value is preserved so
// a nil receiver still yields nil.
func (c *Candle) clone() *Candle {
	if c == nil {
		return nil
	}
	cp := *c
	return &cp
}

// KV is a concurrency-safe in-memory key-value store with TTL support.
type KV struct {
	mu      sync.RWMutex
	items   map[string]entry
	candles map[string]*Candle
}

// New creates an empty KV store.
func New() *KV {
	return &KV{
		items:   make(map[string]entry),
		candles: make(map[string]*Candle),
	}
}

// Set stores value under key with the given TTL. A non-positive TTL stores an
// already-expired entry: Get reports it absent immediately and the next Sweep
// reclaims it. Values are stored and returned as given — Set does not deep-copy
// mutable payloads, so callers storing maps or slices own their immutability.
func (k *KV) Set(key string, value any, ttl time.Duration) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.items[key] = entry{value: value, expiresAt: time.Now().Add(ttl)}
}

// Get returns the value stored under key, or nil if missing or expired. An
// expired entry is left in place for Sweep to reclaim; Get does not mutate.
func (k *KV) Get(key string) (any, bool) {
	k.mu.RLock()
	defer k.mu.RUnlock()
	e, ok := k.items[key]
	if !ok {
		return nil, false
	}
	if time.Now().After(e.expiresAt) {
		return nil, false
	}
	return e.value, true
}

// Delete removes key from both the TTL map and the candle map. The two share
// one key space, so deleting a symbol drops both its TTL entries and its
// live candle. Deleting a key that is absent is a no-op.
func (k *KV) Delete(key string) {
	k.mu.Lock()
	defer k.mu.Unlock()
	delete(k.items, key)
	delete(k.candles, key)
}

// Sweep drops every expired TTL entry and returns how many were removed.
// Entries that are still live and all candles are left untouched. Sweep is
// safe to call concurrently with Get, Set, and UpdateCandle, and safe to call
// often — the TTL map holds one entry per distinct key.
func (k *KV) Sweep(now time.Time) int {
	k.mu.Lock()
	defer k.mu.Unlock()
	removed := 0
	for key, e := range k.items {
		if now.After(e.expiresAt) {
			delete(k.items, key)
			removed++
		}
	}
	return removed
}

// Len returns the number of TTL entries currently held, including any that
// have expired but not yet been swept. It is a diagnostic for tests and
// metrics, not a consistency primitive.
func (k *KV) Len() int {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return len(k.items)
}

// UpdateCandle folds a tick price/volume into the current 1-minute candle
// for symbol, creating a new candle when the minute rolls over. The returned
// candle is a snapshot taken under the write lock; later ticks do not mutate
// it.
func (k *KV) UpdateCandle(symbol string, ts time.Time, price float64, volume int64) *Candle {
	minute := ts.Truncate(time.Minute)

	k.mu.Lock()
	defer k.mu.Unlock()

	cur, ok := k.candles[symbol]
	if !ok || !cur.Timestamp.Equal(minute) {
		cur = &Candle{
			Symbol:    symbol,
			Timestamp: minute,
			Open:      price,
			High:      price,
			Low:       price,
			Close:     price,
			Volume:    volume,
		}
		k.candles[symbol] = cur
		return cur.clone()
	}

	if price > cur.High {
		cur.High = price
	}
	if price < cur.Low {
		cur.Low = price
	}
	cur.Close = price
	cur.Volume += volume
	return cur.clone()
}

// Candle returns a snapshot of the current candle for symbol, or nil if the
// symbol has not traded since the store was created. The snapshot is detached
// from the store: mutating it does not corrupt the candle the producer is
// still folding ticks into.
func (k *KV) Candle(symbol string) *Candle {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.candles[symbol].clone()
}

// Candles returns a snapshot of every current candle, in unspecified order.
// Each candle is detached from the store, as with Candle.
func (k *KV) Candles() []*Candle {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]*Candle, 0, len(k.candles))
	for _, c := range k.candles {
		out = append(out, c.clone())
	}
	return out
}
