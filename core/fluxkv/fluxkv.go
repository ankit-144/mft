// Package fluxkv implements an in-memory cache with TTL-based eviction and
// 1-minute candle aggregation. It is the hot-path state store for the MFT
// platform.
package fluxkv

import (
	"sync"
	"time"
)

type entry struct {
	value      any
	expiresAt  time.Time
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

// KV is a concurrency-safe in-memory key-value store with TTL support.
type KV struct {
	mu     sync.RWMutex
	items  map[string]entry
	candles map[string]*Candle
}

// New creates an empty KV store.
func New() *KV {
	return &KV{
		items:   make(map[string]entry),
		candles: make(map[string]*Candle),
	}
}

// Set stores value under key with the given TTL.
func (k *KV) Set(key string, value any, ttl time.Duration) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.items[key] = entry{value: value, expiresAt: time.Now().Add(ttl)}
}

// Get returns the value stored under key, or nil if missing or expired.
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

// Delete removes key from the store.
func (k *KV) Delete(key string) {
	k.mu.Lock()
	defer k.mu.Unlock()
	delete(k.items, key)
}

// UpdateCandle folds a tick price/volume into the current 1-minute candle
// for symbol, creating a new candle when the minute rolls over.
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
		return cur
	}

	if price > cur.High {
		cur.High = price
	}
	if price < cur.Low {
		cur.Low = price
	}
	cur.Close = price
	cur.Volume += volume
	return cur
}

// Candle returns the current candle for symbol, or nil.
func (k *KV) Candle(symbol string) *Candle {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.candles[symbol]
}

// Candles returns a snapshot of all current candles.
func (k *KV) Candles() []*Candle {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]*Candle, 0, len(k.candles))
	for _, c := range k.candles {
		out = append(out, c)
	}
	return out
}
