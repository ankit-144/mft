package jobs

import "errors"

// Sentinel errors returned by the backfill job.
var (
	ErrInvalidSegment = errors.New("jobs: invalid historical segment")

	ErrNoHistory = errors.New("jobs: no historical data was backfilled")
)
