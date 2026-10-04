package broker

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// RESTCategory selects a Kite Connect request quota.
type RESTCategory uint8

const (
	RESTOther RESTCategory = iota
	RESTQuote
	RESTHistorical
	RESTOrder
)

var restIntervals = map[RESTCategory]time.Duration{
	RESTQuote:      time.Second,
	RESTHistorical: time.Second / 3,
	RESTOrder:      100 * time.Millisecond,
	RESTOther:      100 * time.Millisecond,
}

// RESTPacer serializes request starts at Kite's per-endpoint rates.
type RESTPacer struct {
	mu    sync.Mutex
	next  map[RESTCategory]time.Time
	now   func() time.Time
	sleep func(context.Context, time.Duration) error
}

// NewRESTPacer creates an independent set of per-category gates.
func NewRESTPacer() *RESTPacer {
	return &RESTPacer{
		next:  make(map[RESTCategory]time.Time),
		now:   time.Now,
		sleep: waitContext,
	}
}

// Wait reserves a request slot or returns when ctx is canceled.
func (p *RESTPacer) Wait(ctx context.Context, category RESTCategory) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	interval, ok := restIntervals[category]
	if !ok {
		return fmt.Errorf("broker: unknown REST request category %d", category)
	}
	p.mu.Lock()
	now := p.now()
	due := p.next[category]
	if due.Before(now) {
		due = now
	}
	p.next[category] = due.Add(interval)
	p.mu.Unlock()
	if delay := due.Sub(now); delay > 0 {
		if err := p.sleep(ctx, delay); err != nil {
			p.releaseLast(category, due, interval)
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		p.releaseLast(category, due, interval)
		return err
	}
	return nil
}

func (p *RESTPacer) releaseLast(category RESTCategory, due time.Time, interval time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	reservedEnd := due.Add(interval)
	if p.next[category].Equal(reservedEnd) {
		p.next[category] = due
	}
}

func waitContext(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
