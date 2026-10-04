package risk

import "time"

// Clock reports the current time.
type Clock func() time.Time

// SystemClock returns the current instant.
func SystemClock() time.Time { return time.Now().UTC() }

// now reads the clock, falling back to SystemClock when the check was built without
// one.
func (c Clock) now() time.Time {
	if c == nil {
		return SystemClock()
	}
	return c()
}
