// Package jobs schedules bounded historical backfills with broker rate limits.
package jobs

import (
	"context"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/contracts"
)

// HistoricalClient is the narrow slice of broker history that backfill needs.
type HistoricalClient interface {
	// HistoricalCandles returns the candles for symbol over [from, to).
	HistoricalCandles(ctx context.Context, symbol string, from, to time.Time, interval string) ([]contracts.Candle, error)
}

// InstrumentResolver is the narrow slice of the broker instrument master that backfill
// needs: a symbol to exchange and instrument token mapping.
type InstrumentResolver interface {
	// Instruments returns the resolved instruments for the configured watchlist.
	Instruments(ctx context.Context) ([]contracts.Instrument, error)
}

// Segment is one bounded [From,To) request to the broker.
type Segment struct {
	// Symbol is the upper-case tradable symbol, e.g.
	Symbol string
	// From is the inclusive start of the segment, UTC.
	From time.Time
	// To is the exclusive end of the segment, UTC.
	To time.Time
	// Candles are the bars in the segment, ascending by Timestamp.
	Candles []contracts.Candle
}

// SegmentStore is the narrow slice of the Parquet cold store that backfill needs.
type SegmentStore interface {
	// WriteSegment durably writes one segment, replacing any file already at that location.
	WriteSegment(ctx context.Context, seg Segment) error
	// HasSegment reports whether a segment is already present on disk.
	HasSegment(ctx context.Context, seg Segment) (bool, error)
}

// Compile-time proof that C1's connector satisfies the resolver seam without any
// adapter: if core/broker ever changes its method set, this file stops compiling rather
// than the jobs service failing at startup.
var _ InstrumentResolver = (*broker.Kite)(nil)
