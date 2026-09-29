package features

import "time"

// SessionOpenMinute is the NSE cash session open as minutes past midnight in
// the injected trading location: 09:15. It is the anchor for
// minute_of_session and matches services/execution/risk's market-hours check,
// which refuses signals outside 09:15-15:30 in the same zone.
const SessionOpenMinute = 9*60 + 15

// minuteOfSession returns bars elapsed since the session open, for a candle
// timestamp t expressed in loc.
//
// The value is the clock offset from 09:15 in loc, not a count of the bars seen
// so far. For a gapless session the two are identical — the NSE cash session
// runs continuously from 09:15 to 15:30, so bar 0 is 09:15 and bar 375 is 15:30
// — but the clock form is the one that keeps the feature a pure function of a
// single candle. A bar counter would shift every row whenever the storage layer
// dropped a candle, which would make the value of the newest row depend on
// where in the history the caller started reading.
//
// The value is deliberately not clamped to the session. A bar before the open
// yields a negative number and one after the close yields more than 375; both
// say "this bar is out of session", which is exactly the fact a model wants
// during the 09:00-09:15 auction window and the post-close block trade. Clamping
// would fold those onto real session minutes.
func minuteOfSession(t time.Time, loc *time.Location) float64 {
	local := t.In(loc)
	return float64(local.Hour()*60 + local.Minute() - SessionOpenMinute)
}

// hourOfDay returns the hour of the day, 0-23, of t expressed in loc.
//
// The candle timestamps the platform stores are UTC. This feature is a calendar
// clock, so it is computed in the trading location: 00:30 UTC is 06:00 IST and
// must be labelled 6. The two are equal only for the 05:30-06:00 UTC window.
func hourOfDay(t time.Time, loc *time.Location) float64 {
	return float64(t.In(loc).Hour())
}
