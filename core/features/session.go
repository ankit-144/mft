package features

import "time"

// SessionOpenMinute is the NSE cash session open as minutes past midnight in the
// injected trading location: 09:15.
const SessionOpenMinute = 9*60 + 15

// minuteOfSession returns bars elapsed since the session open, for a candle timestamp t
// expressed in loc.
func minuteOfSession(t time.Time, loc *time.Location) float64 {
	local := t.In(loc)
	return float64(local.Hour()*60 + local.Minute() - SessionOpenMinute)
}

// hourOfDay returns the hour of the day, 0-23, of t expressed in loc.
func hourOfDay(t time.Time, loc *time.Location) float64 {
	return float64(t.In(loc).Hour())
}
