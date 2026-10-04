package features

import (
	"testing"
	"time"

	"github.com/mft/core/contracts"
)

// sessionCandles returns n bars of one-minute spacing starting at start. It is
// the only generator the session tests need: the price path is irrelevant to
// the calendar features, so every bar is the same doji.
func sessionCandles(start time.Time, n int) []contracts.Candle {
	out := make([]contracts.Candle, n)
	for i := range out {
		out[i] = contracts.Candle{
			Symbol:    "RELIANCE",
			Timestamp: start.Add(time.Duration(i) * time.Minute),
			Open:      100,
			High:      100,
			Low:       100,
			Close:     100,
			Volume:    500,
		}
	}
	return out
}

// TestMinuteOfSessionCountsBarsFromTheOpen pins the anchor. A builder that
// counted from midnight, or that read the UTC timestamp, would be off by
// hundreds of minutes here rather than subtly wrong.
func TestMinuteOfSessionCountsBarsFromTheOpen(t *testing.T) {
	tests := []struct {
		name  string
		start time.Time // UTC
		want  []float64
	}{
		{
			name:  "the session open, 09:15 IST",
			start: time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC),
			want:  []float64{0, 1, 2, 3},
		},
		{
			name:  "one minute before the open, 09:14 IST",
			start: time.Date(2026, 9, 29, 3, 44, 0, 0, time.UTC),
			want:  []float64{-1, 0, 1, 2},
		},
		{
			name:  "mid-session, 13:20 IST",
			start: time.Date(2026, 9, 29, 7, 50, 0, 0, time.UTC),
			want:  []float64{245, 246, 247, 248},
		},
		{
			name:  "the close, 15:30 IST",
			start: time.Date(2026, 9, 29, 10, 0, 0, 0, time.UTC),
			want:  []float64{375, 376, 377, 378},
		},
	}

	b := NewBuilder(tradingZone(t))
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := b.Build(sessionCandles(tc.start, WarmupRows+len(tc.want)))
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			for i, want := range tc.want {
				if got := col(t, table, i, "minute_of_session"); got != want {
					t.Errorf("row %d (bar %d, %s IST) minute_of_session = %v, want %v",
						i, i, tc.start.Add(time.Duration(i)*time.Minute).In(tradingZone(t)).Format("15:04"), got, want)
				}
			}
		})
	}
}

// TestSessionFeaturesSurviveUTCMidnight walks a window across midnight in both
// zones. The platform stores UTC, so the rollover this test guards against is
// the one that happens five and a half hours before the IST one: a builder that
// formatted a timestamp in UTC, or that rebuilt a date from a year/month/day
// triple, ends up with a wrong or doubled hour here.
func TestSessionFeaturesSurviveUTCMidnight(t *testing.T) {
	tests := []struct {
		name        string
		start       time.Time // UTC
		offset      int       // first asserted bar
		wantHours   []float64
		wantMinutes []float64
	}{
		{
			// 23:45 IST is 18:15 UTC. Bar 15 is 00:00 IST, so the asserted
			// bars straddle midnight in IST while the UTC date never moves.
			name:        "IST rolls over midnight while UTC does not",
			start:       time.Date(2026, 9, 29, 18, 15, 0, 0, time.UTC),
			offset:      13,
			wantHours:   []float64{23, 23, 0, 0},
			wantMinutes: []float64{883, 884, -555, -554},
		},
		{
			// 05:15 IST is 23:45 UTC. Bar 15 is 00:00 UTC, so the asserted
			// bars straddle midnight in UTC while the IST clock does not move.
			name:        "UTC rolls over midnight while IST does not",
			start:       time.Date(2026, 9, 29, 23, 45, 0, 0, time.UTC),
			offset:      13,
			wantHours:   []float64{5, 5, 5, 5},
			wantMinutes: []float64{-227, -226, -225, -224},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := NewBuilder(tradingZone(t))
			candles := sessionCandles(tc.start, WarmupRows+tc.offset+len(tc.wantHours))
			table, err := b.Build(candles)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			for i, wantHour := range tc.wantHours {
				row := tc.offset + i
				if got := col(t, table, row, "hour_of_day"); got != wantHour {
					t.Errorf("row %d (%s IST) hour_of_day = %v, want %v",
						row, candles[row].Timestamp.In(tradingZone(t)).Format("02 Jan 15:04"), got, wantHour)
				}
				if got := col(t, table, row, "minute_of_session"); got != tc.wantMinutes[i] {
					t.Errorf("row %d (%s IST) minute_of_session = %v, want %v",
						row, candles[row].Timestamp.In(tradingZone(t)).Format("02 Jan 15:04"), got, tc.wantMinutes[i])
				}
			}
		})
	}
}

// TestISTAndUTCDisagreeOnTheseBars is the direct proof that the zone is the one
// the contract asks for. It fails loudly if a builder ever hardcodes UTC, and
// it fails loudly if a builder ever hardcodes IST, because both are exercised.
func TestISTAndUTCDisagreeOnTheseBars(t *testing.T) {
	start := time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC) // 09:15 IST
	candles := sessionCandles(start, WarmupRows+1)

	ist := NewBuilder(tradingZone(t))
	utc := NewBuilder(time.UTC)

	istTable, err := ist.Build(candles)
	if err != nil {
		t.Fatalf("Build(IST): %v", err)
	}
	utcTable, err := utc.Build(candles)
	if err != nil {
		t.Fatalf("Build(UTC): %v", err)
	}

	last := len(candles) - 1
	if got, want := col(t, istTable, last, "minute_of_session"), float64(last); got != want {
		t.Errorf("IST minute_of_session at the last bar = %v, want %v", got, want)
	}
	// In UTC the same bar is 04:45, which is 285 minutes past midnight and
	// 270 minutes before the 09:15 IST open.
	if got, want := col(t, utcTable, last, "minute_of_session"), float64(4*60+45-555); got != want {
		t.Errorf("UTC minute_of_session at the last bar = %v, want %v", got, want)
	}
	if col(t, istTable, last, "hour_of_day") == col(t, utcTable, last, "hour_of_day") {
		t.Error("IST and UTC report the same hour; the test window no longer separates the zones")
	}
}

// TestNilLocationMeansUTC documents the one silent degradation NewBuilder
// allows: no location means UTC, which is a wrong answer for the session
// columns rather than a missing one. It is tested so the behaviour is at least
// deliberate and visible.
func TestNilLocationMeansUTC(t *testing.T) {
	candles := sessionCandles(time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC), WarmupRows+1)

	withNil, err := NewBuilder(nil).Build(candles)
	if err != nil {
		t.Fatalf("Build(nil): %v", err)
	}
	explicit, err := NewBuilder(time.UTC).Build(candles)
	if err != nil {
		t.Fatalf("Build(UTC): %v", err)
	}

	last := len(candles) - 1
	if withNil.Rows[last][colMinuteOfSession] != explicit.Rows[last][colMinuteOfSession] {
		t.Fatal("a nil location does not mean UTC")
	}
	if withNil.Rows[last][colMinuteOfSession] == float64(last) {
		t.Fatal("the test window no longer separates UTC from IST")
	}
}

// TestZeroValueBuilderStillBuilds keeps a Builder that was never constructed
// from panicking. It has to be UTC, but it has to work.
func TestZeroValueBuilderStillBuilds(t *testing.T) {
	var b builder
	candles := sessionCandles(time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC), WarmupRows+1)
	table, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build on the zero builder: %v", err)
	}
	if len(table.Rows) != len(candles) {
		t.Fatalf("got %d rows, want %d", len(table.Rows), len(candles))
	}
}
