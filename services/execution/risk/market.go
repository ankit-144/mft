package risk

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/mft/core/contracts"
)

// IST returns the Asia/Kolkata location used for every session decision.
//
// The result is resolved once. India observes no daylight saving, so when the
// host has no tz database the fixed +05:30 offset is exactly equivalent to the
// named zone and the session bounds below stay correct.
func IST() *time.Location {
	return istLocation()
}

var istLocation = sync.OnceValue(func() *time.Location {
	loc, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		return time.FixedZone("IST", 5*60*60+30*60)
	}
	return loc
})

// NSE cash session bounds, as minutes past IST midnight. The contract fixes
// the window at 09:15–15:30 IST inclusive.
const (
	marketOpenMinute  = 9*60 + 15
	marketCloseMinute = 15*60 + 30
)

// HolidayLayout is the date format of StaticCalendar keys: an IST calendar
// date, "2006-01-02".
const HolidayLayout = "2006-01-02"

// Calendar reports NSE trading holidays.
//
// The interface exists so the session check has no network dependency and no
// hidden state. A nil Calendar, or one built from an empty set, reports no
// holidays and the session check then treats every weekday as a trading day.
// That is the documented default: it errs towards trading on a holiday, which
// the broker will reject, rather than blocking a valid signal on stale local
// data.
type Calendar interface {
	// IsHoliday reports whether the IST calendar date of t is a trading
	// holiday. t may be in any zone; only its IST date is consulted.
	IsHoliday(t time.Time) bool
}

// StaticCalendar is a Calendar backed by an explicit set of holiday dates.
// The zero value and a nil map report no holidays.
type StaticCalendar struct {
	holidays map[string]struct{}
}

// NewStaticCalendar builds a calendar from IST dates written as
// "2006-01-02". Unparseable dates are dropped rather than failing startup —
// a typo in a holiday list must not be able to stop the risk engine from
// booting. Pass none for a calendar that reports no holidays.
func NewStaticCalendar(dates ...string) StaticCalendar {
	c := StaticCalendar{holidays: make(map[string]struct{}, len(dates))}
	for _, d := range dates {
		if _, err := time.Parse(HolidayLayout, d); err != nil {
			continue
		}
		c.holidays[d] = struct{}{}
	}
	return c
}

// IsHoliday implements Calendar.
func (c StaticCalendar) IsHoliday(t time.Time) bool {
	if len(c.holidays) == 0 {
		return false
	}
	_, ok := c.holidays[t.In(IST()).Format(HolidayLayout)]
	return ok
}

// MarketHoursCheck refuses signals outside the NSE cash session: 09:15–15:30
// IST inclusive, weekdays, and not a holiday in the injected Calendar.
type MarketHoursCheck struct {
	// Clock supplies "now". nil means wall clock.
	Clock Clock
	// Calendar supplies holidays. nil means "assume every weekday trades".
	Calendar Calendar
}

// Check implements contracts.Checker.
func (c MarketHoursCheck) Check(_ context.Context, _ contracts.Signal, _ contracts.Portfolio) error {
	now := c.Clock.now().In(IST())

	switch now.Weekday() {
	case time.Saturday, time.Sunday:
		return contracts.Reject(contracts.ReasonMarketClosed,
			fmt.Sprintf("market closed: %s is a %s", now.Format(HolidayLayout), now.Weekday()))
	}

	if c.Calendar != nil && c.Calendar.IsHoliday(now) {
		return contracts.Reject(contracts.ReasonMarketClosed,
			fmt.Sprintf("market closed: %s is an NSE holiday", now.Format(HolidayLayout)))
	}

	minute := now.Hour()*60 + now.Minute()
	if minute < marketOpenMinute || minute > marketCloseMinute {
		return contracts.Reject(contracts.ReasonMarketClosed,
			fmt.Sprintf("market closed at %s IST; session is 09:15-15:30", now.Format("15:04")))
	}
	return nil
}
