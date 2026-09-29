package risk

import "time"

// Clock reports the current time. Every check that reasons about wall-clock
// time takes a Clock rather than calling time.Now itself, so tests are
// deterministic and never depend on the machine's local timezone.
type Clock func() time.Time

// SystemClock returns the current instant. All timestamps are UTC on the wire
// (docs/contracts.md §1); only MarketHoursCheck converts to IST, at the last
// possible moment, so that no other check has to think about timezones.
func SystemClock() time.Time { return time.Now().UTC() }

// now reads the clock, falling back to SystemClock when the check was built
// without one. A nil Clock is a normal value here, not a programming error:
// Policy{} with no clock is a valid, wall-clock-reading policy.
func (c Clock) now() time.Time {
	if c == nil {
		return SystemClock()
	}
	return c()
}
