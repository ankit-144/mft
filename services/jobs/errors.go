package jobs

import "errors"

// Sentinel errors returned by the backfill job. Callers distinguish failure
// modes with errors.Is; every error the job returns wraps one of these.
//
// The broker sentinels (broker.ErrAuth, broker.ErrRateLimit) are re-wrapped
// from core/broker rather than redeclared, so a single errors.Is works whether
// the error came from this package or from C1's connector.
var (
	// ErrInvalidSegment means a segment could not be described unambiguously:
	// no symbol, an unbound date, or a range that ends before it starts.
	ErrInvalidSegment = errors.New("jobs: invalid historical segment")

	// ErrNoHistory means the run made no progress: every configured symbol
	// failed for the same reason, usually broker.ErrAuth. A partial run is
	// not an error, because the checkpoint makes the remainder cheap.
	ErrNoHistory = errors.New("jobs: no historical data was backfilled")
)
