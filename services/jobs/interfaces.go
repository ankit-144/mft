// Package jobs implements Service 4 of the MFT platform: background jobs and
// historical backfilling. It uses netresearch/go-cron for in-process
// scheduling and enforces rate limits against the broker historical API.
//
// # Seams
//
// Backfill needs three things that live outside this package, and none of them
// has the shape docs/contracts.md froze for it. Rather than widen a frozen
// contract or edit another component's package, this file declares the narrow
// interfaces backfill actually calls and adapts to them:
//
//   - [HistoricalClient] — one method of historical candle history. C1 built
//     core/broker for streaming and orders; Kite's /data/historical REST API was
//     out of scope there, so the Kite-backed implementation
//     [NewKiteHistory] lives here in services/jobs and speaks the Kite wire
//     format directly.
//   - [InstrumentResolver] — already satisfied by *broker.Kite as-is. It exists
//     so the historical client can resolve a symbol to an exchange and an
//     instrument token without depending on the whole connector.
//   - [SegmentStore] — the Parquet layout of docs/contracts.md §3 for
//     data/historical/. C2 owns core/storage, whose Writer is tick-shaped and
//     append-oriented: it has no notion of a bounded, immutable, resumable
//     segment, and its flush path is keyed on wall-clock time rather than the
//     date range the request asked for. See the report for what C2 would need
//     to add for backfill to depend on it directly.
//
// The interfaces are deliberately one- and two-method. A narrow seam is
// substitutable in a test without a mock framework, and it states exactly what
// this component assumes of a component it does not own.
package jobs

import (
	"context"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/contracts"
)

// HistoricalClient is the narrow slice of broker history that backfill needs.
//
// Implementations return the candles whose bar-open timestamp falls in
// [from, to], ascending by time. A failure caused by throttling must wrap
// broker.ErrRateLimit so the caller can back off; an authentication failure
// must wrap broker.ErrAuth so the caller can stop rather than retry.
type HistoricalClient interface {
	// HistoricalCandles returns the candles for symbol over [from, to).
	HistoricalCandles(ctx context.Context, symbol string, from, to time.Time, interval string) ([]contracts.Candle, error)
}

// InstrumentResolver is the narrow slice of the broker instrument master that
// backfill needs: a symbol to exchange and instrument token mapping.
//
// *broker.Kite satisfies this interface unchanged, so binding it costs nothing
// and keeps C1's connector the single source of truth for symbol resolution.
type InstrumentResolver interface {
	// Instruments returns the resolved instruments for the configured watchlist.
	Instruments(ctx context.Context) ([]contracts.Instrument, error)
}

// Segment is one bounded, immutable slice of a symbol's history: a single
// [From, To) request to the broker, written to exactly one Parquet file.
//
// The From/To pair is the segment identity. It is also the directory name in
// the data/historical layout of docs/contracts.md §3, so resuming a run is a
// question about which segments already exist on disk.
type Segment struct {
	// Symbol is the upper-case tradable symbol, e.g. "RELIANCE".
	Symbol string
	// From is the inclusive start of the segment, UTC.
	From time.Time
	// To is the exclusive end of the segment, UTC.
	To time.Time
	// Candles are the bars in the segment, ascending by Timestamp.
	Candles []contracts.Candle
}

// SegmentStore is the narrow slice of the Parquet cold store that backfill
// needs. It is deliberately segment-shaped rather than row-shaped: the
// historical tree is a set of immutable files, and a store that could append
// into one would break the DuckDB rule that readers only ever see closed files.
//
// A WriteSegment must be atomic. A run interrupted mid-write must leave either
// the complete file or none at all, because HasSegment is what a resumed run
// uses to decide what to skip.
type SegmentStore interface {
	// WriteSegment durably writes one segment, replacing any file already at
	// that location.
	WriteSegment(ctx context.Context, seg Segment) error
	// HasSegment reports whether a segment is already present on disk.
	HasSegment(ctx context.Context, seg Segment) (bool, error)
}

// Compile-time proof that C1's connector satisfies the resolver seam without
// any adapter: if core/broker ever changes its method set, this file stops
// compiling rather than the jobs service failing at startup.
var _ InstrumentResolver = (*broker.Kite)(nil)
