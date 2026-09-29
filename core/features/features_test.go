package features

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/contracts"
)

// tradingZone returns Asia/Kolkata, the location every session feature is
// specified in. When the host has no tz database it returns the fixed +05:30
// offset instead: India observes no daylight saving, so the two are identical
// for every timestamp this package is handed, and a container without tzdata
// should not fail a test about IST.
func tradingZone(t *testing.T) *time.Location {
	t.Helper()
	loc, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		return time.FixedZone("IST", 5*60*60+30*60)
	}
	return loc
}

// col returns the value of one named column of row i, so a test can assert a
// feature without counting positions in the schema by hand.
func col(t *testing.T, table FeatureTable, i int, name string) float64 {
	t.Helper()
	if i < 0 || i >= len(table.Rows) {
		t.Fatalf("row %d out of range: table has %d rows", i, len(table.Rows))
	}
	for j, c := range table.Columns {
		if c == name {
			if j >= len(table.Rows[i]) {
				t.Fatalf("row %d has %d values, want at least %d", i, len(table.Rows[i]), j+1)
			}
			return table.Rows[i][j]
		}
	}
	t.Fatalf("unknown column %q", name)
	return 0
}

// assertClose compares two feature values with a relative tolerance, which is
// the right yardstick for columns that range over several orders of magnitude.
func assertClose(t *testing.T, got, want float64, name string) {
	t.Helper()
	tol := 1e-9 * math.Max(1, math.Abs(want))
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %.17g, want %.17g (tolerance %.3g)", name, got, want, tol)
	}
}

// flatCandles returns n identical bars: the degenerate input that every
// zero-denominator policy in this package exists for.
func flatCandles(n int) []contracts.Candle {
	start := time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC)
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

// spikeCandles returns n bars pinned at 100 apart from the one at index spike,
// which closes at 200. A single distinctive bar is the clearest way to prove
// that row i was computed from candle i.
func spikeCandles(n, spike int) []contracts.Candle {
	start := time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC)
	out := make([]contracts.Candle, n)
	for i := range out {
		px := 100.0
		if i == spike {
			px = 200
		}
		out[i] = contracts.Candle{
			Symbol:    "RELIANCE",
			Timestamp: start.Add(time.Duration(i) * time.Minute),
			Open:      px,
			High:      px,
			Low:       px,
			Close:     px,
			Volume:    500,
		}
	}
	return out
}

func TestDefaultColumnsAreFrozen(t *testing.T) {
	want := []string{
		"ret_1", "ret_5", "ret_15", "ret_60",
		"vol_5", "vol_20", "vol_ratio",
		"range_1", "body_1", "upper_wick_1", "lower_wick_1",
		"volume_z_20", "volume_ratio",
		"momentum_rsi_14", "sma_gap_10",
		"minute_of_session", "hour_of_day", "spread_proxy",
	}
	got := DefaultColumns()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DefaultColumns() =\n%v\nwant\n%v", got, want)
	}

	// The returned slice is a copy: a caller that reorders it cannot corrupt
	// the schema for the next caller.
	got[0] = "mutated"
	if DefaultColumns()[0] != "ret_1" {
		t.Fatal("DefaultColumns returned a slice aliasing the schema")
	}
}

func TestBuildRejectsWindowShorterThanWarmup(t *testing.T) {
	tests := []struct {
		name  string
		nbars int
	}{
		{name: "empty", nbars: 0},
		{name: "one bar", nbars: 1},
		{name: "half a window", nbars: 30},
		{name: "one short of the warm-up", nbars: WarmupRows - 1},
	}
	b := NewBuilder(tradingZone(t))
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table, err := b.Build(flatCandles(tc.nbars))
			if err == nil {
				t.Fatalf("Build(%d candles) returned a table, want an error", tc.nbars)
			}
			if !strings.Contains(err.Error(), "warm-up") {
				t.Fatalf("error %q does not mention the warm-up", err)
			}
			if table.Columns != nil || table.Rows != nil {
				t.Fatal("a rejected build must not return a partial table")
			}
		})
	}
}

func TestBuildAcceptsExactlyWarmupBars(t *testing.T) {
	b := NewBuilder(tradingZone(t))
	table, err := b.Build(flatCandles(WarmupRows))
	if err != nil {
		t.Fatalf("Build(%d candles): %v", WarmupRows, err)
	}
	if len(table.Rows) != WarmupRows {
		t.Fatalf("got %d rows, want %d", len(table.Rows), WarmupRows)
	}
	if !IsWarmup(WarmupRows - 1) {
		t.Fatal("the last row of a minimal table is still a warm-up row")
	}
}

func TestBuildRejectsCorruptInput(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func([]contracts.Candle)
		wantErr string
	}{
		{
			name:    "NaN close",
			mutate:  func(c []contracts.Candle) { c[30].Close = math.NaN() },
			wantErr: "non-finite",
		},
		{
			name:    "infinite high",
			mutate:  func(c []contracts.Candle) { c[12].High = math.Inf(1) },
			wantErr: "non-finite",
		},
		{
			name:    "zero open",
			mutate:  func(c []contracts.Candle) { c[7].Open = 0 },
			wantErr: "non-positive",
		},
		{
			name:    "negative close",
			mutate:  func(c []contracts.Candle) { c[41].Close = -1 },
			wantErr: "non-positive",
		},
		{
			name:    "high below the close",
			mutate:  func(c []contracts.Candle) { c[19].High = c[19].Close - 1 },
			wantErr: "well-formed OHLC",
		},
		{
			name:    "low above the open",
			mutate:  func(c []contracts.Candle) { c[23].Low = c[23].Open + 1 },
			wantErr: "well-formed OHLC",
		},
		{
			name:    "a high below the low",
			mutate:  func(c []contracts.Candle) { c[5].High = c[5].Low - 0.5 },
			wantErr: "well-formed OHLC",
		},
		{
			name:    "negative volume",
			mutate:  func(c []contracts.Candle) { c[33].Volume = -1 },
			wantErr: "negative volume",
		},
		{
			name:    "a second symbol in the window",
			mutate:  func(c []contracts.Candle) { c[44].Symbol = "TCS" },
			wantErr: "one symbol",
		},
		{
			name:    "a repeated timestamp",
			mutate:  func(c []contracts.Candle) { c[28].Timestamp = c[27].Timestamp },
			wantErr: "ascend strictly",
		},
		{
			name: "a window running backwards",
			mutate: func(c []contracts.Candle) {
				c[40], c[41] = c[41], c[40]
			},
			wantErr: "ascend strictly",
		},
	}

	b := NewBuilder(tradingZone(t))
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			candles := spikeCandles(WarmupRows+4, WarmupRows)
			tc.mutate(candles)
			table, err := b.Build(candles)
			if err == nil {
				t.Fatalf("Build accepted corrupt input and returned %d rows", len(table.Rows))
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not mention %q", err, tc.wantErr)
			}
			if table.Rows != nil {
				t.Fatal("a rejected build must not return a partial table")
			}
		})
	}
}

// TestRowsAlignOneForOneWithCandles is the alignment guarantee, stated as a
// test: the row index is the candle index, with no offset by the warm-up.
//
// The inference service zips table.Rows[i] with candles[i] to build the
// regression target. An implementation that shifted rows by the warm-up length
// would return a table of the right width and completely wrong labels, so this
// pins the row index to a distinctive single bar rather than to a range.
func TestRowsAlignOneForOneWithCandles(t *testing.T) {
	// The distinctive bar is the first predictable one: it moves ret_1 at
	// rows 60 and 61, and it is the only thing that can make ret_60 non-zero
	// at row 60, because that row is the first to see candle 0.
	const spike = WarmupRows
	b := NewBuilder(tradingZone(t))
	candles := spikeCandles(WarmupRows+6, spike)

	table, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if len(table.Rows) != len(candles) {
		t.Fatalf("got %d rows for %d candles; Build must not drop or shift rows",
			len(table.Rows), len(candles))
	}

	var moved []int
	for i := range table.Rows {
		if col(t, table, i, "ret_1") != 0 {
			moved = append(moved, i)
		}
	}
	want := []int{spike, spike + 1} // the jump up, then the jump back down
	if !reflect.DeepEqual(moved, want) {
		t.Fatalf("rows with a non-zero 1-bar return: %v, want %v — rows are not aligned with their candles", moved, want)
	}

	// ret_60 at row WarmupRows can only be non-zero if that row looked back to
	// candle 0, which is exactly the assertion an offset-by-59 build fails.
	if got, want := col(t, table, WarmupRows, "ret_60"), math.Log(2); math.Abs(got-want) > 1e-12 {
		t.Fatalf("ret_60 = %v at row %d, want ln(2): that row did not see candle 0", got, WarmupRows)
	}
	if got := col(t, table, WarmupRows-1, "ret_60"); got != 0 {
		t.Fatalf("ret_60 = %v on the last warm-up row, want 0", got)
	}

	if !table.AsOf.Equal(candles[len(candles)-1].Timestamp) {
		t.Fatalf("AsOf = %s, want the last candle at %s",
			table.AsOf, candles[len(candles)-1].Timestamp)
	}
}

func TestBuildIsDeterministic(t *testing.T) {
	b := NewBuilder(tradingZone(t))
	candles := spikeCandles(WarmupRows+10, WarmupRows+5)

	first, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	second, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("two builds of the same candles differ; Build must be a pure function")
	}
}

// TestBuildDoesNotDependOnWindowPosition protects the pull architecture in
// Plan.md §4: inference reads the last N candles it was configured with, a
// backfill reads a year, and the newest row must describe the same market state
// either way. A feature that carried state forward — Wilder's RSI, a cumulative
// sum, an expanding mean — would fail here.
func TestBuildDoesNotDependOnWindowPosition(t *testing.T) {
	candles := spikeCandles(WarmupRows+40, 70)
	b := NewBuilder(tradingZone(t))

	full, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build(full): %v", err)
	}

	const tail = WarmupRows + 5
	tailTable, err := b.Build(candles[len(candles)-tail:])
	if err != nil {
		t.Fatalf("Build(tail): %v", err)
	}

	// Rows before WarmupRows of the tail are legitimately different: their
	// trailing windows run past the start of the slice and the warm-up rule
	// fills them with zeros. From row WarmupRows onward — the first row whose
	// longest window is inside the slice — the two builds must agree exactly.
	for i := WarmupRows; i < tail; i++ {
		if !reflect.DeepEqual(tailTable.Rows[i], full.Rows[len(full.Rows)-tail+i]) {
			t.Fatalf("row %d of the tail build differs from the full build", i)
		}
	}
}

func TestIsWarmup(t *testing.T) {
	for _, tc := range []struct {
		row  int
		want bool
	}{
		{row: 0, want: true},
		{row: 1, want: true},
		{row: WarmupRows - 1, want: true},
		{row: WarmupRows, want: false},
		{row: WarmupRows + 1, want: false},
	} {
		if got := IsWarmup(tc.row); got != tc.want {
			t.Errorf("IsWarmup(%d) = %v, want %v", tc.row, got, tc.want)
		}
	}
}

// TestNoNonFiniteValuesAnywhere is the output half of the numeric policy: no
// NaN and no infinity ever reaches a model, whatever the input looks like.
func TestNoNonFiniteValuesAnywhere(t *testing.T) {
	b := NewBuilder(tradingZone(t))
	series := map[string][]contracts.Candle{
		"flat":              flatCandles(WarmupRows + 10),
		"spike":             spikeCandles(WarmupRows+10, WarmupRows+5),
		"mixed":             mixedCandles(),
		"zero volume":       zeroVolumeCandles(WarmupRows + 5),
		"every bar a doji":  dojiCandles(WarmupRows + 5),
		"alternating close": alternatingCandles(WarmupRows + 5),
	}
	for name, candles := range series {
		t.Run(name, func(t *testing.T) {
			table, err := b.Build(candles)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			for i, row := range table.Rows {
				if len(row) != len(table.Columns) {
					t.Fatalf("row %d has %d values, want %d", i, len(row), len(table.Columns))
				}
				for j, v := range row {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Fatalf("row %d column %s is %v", i, table.Columns[j], v)
					}
				}
			}
		})
	}
}

// TestZeroDenominatorsSubstituteZero pins the policy on the degenerate series:
// a market that does not move must produce a table of zeros, not an error and
// not a NaN.
//
// The two exceptions are deliberate. volume_ratio divides by a mean volume that
// is present, not zero, so it reads 1 — a constant market is a typical one —
// and the two session columns are calendar arithmetic, always real.
func TestZeroDenominatorsSubstituteZero(t *testing.T) {
	b := NewBuilder(tradingZone(t))
	table, err := b.Build(flatCandles(WarmupRows + 5))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	i := len(table.Rows) - 1
	for _, name := range []string{
		"ret_1", "ret_5", "ret_15", "ret_60",
		"vol_5", "vol_20", "vol_ratio",
		"range_1", "body_1", "upper_wick_1", "lower_wick_1",
		"volume_z_20",
		"momentum_rsi_14", "sma_gap_10",
		"spread_proxy",
	} {
		if got := col(t, table, i, name); got != 0 {
			t.Errorf("%s = %v on a flat series, want the substituted 0", name, got)
		}
	}
	if got := col(t, table, i, "volume_ratio"); got != 1 {
		t.Errorf("volume_ratio = %v at constant volume, want 1", got)
	}
}

// TestZeroVolumeIsZeroNotInfinite checks the two columns that divide by volume
// on a bar that never traded.
func TestZeroVolumeIsZeroNotInfinite(t *testing.T) {
	b := NewBuilder(tradingZone(t))
	candles := zeroVolumeCandles(WarmupRows + 5)
	table, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	i := WarmupRows + 4
	for _, name := range []string{"spread_proxy", "volume_ratio"} {
		if got := col(t, table, i, name); got != 0 {
			t.Errorf("%s = %v on a zero-volume bar, want 0", name, got)
		}
	}
}

// TestVolRatioOnAlternatingReturns pins vol_ratio to an exact closed form.
//
// The trend golden cannot: its returns are identical to within floating point,
// so both volatilities are noise around zero and their ratio is meaningless.
// Alternating returns make both windows exactly solvable. With 1-bar log
// returns of alternating sign and equal magnitude L:
//
//	vol_5  of a 5-window  (3 up, 2 down)  = L*sqrt(6/5)
//	vol_20 of a 20-window (10 up, 10 down) = L*sqrt(20/19)
//	vol_ratio                             = sqrt(1.14)
func TestVolRatioOnAlternatingReturns(t *testing.T) {
	b := NewBuilder(tradingZone(t))
	candles := alternatingCandles(WarmupRows + 5)
	table, err := b.Build(candles)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	l := math.Log(1.01)
	i := len(table.Rows) - 1
	assertClose(t, col(t, table, i, "vol_5"), l*math.Sqrt(6.0/5.0), "vol_5")
	assertClose(t, col(t, table, i, "vol_20"), l*math.Sqrt(20.0/19.0), "vol_20")
	assertClose(t, col(t, table, i, "vol_ratio"), math.Sqrt(1.14), "vol_ratio")
}

// TestRSISaturatesAtBothEnds covers the two branches of RSI(14) that a mixed
// price path never reaches: an unbroken run of up bars and of down bars.
func TestRSISaturatesAtBothEnds(t *testing.T) {
	tests := []struct {
		name string
		step float64
		want float64
	}{
		{name: "every bar up", step: 1.01, want: 1},
		{name: "every bar down", step: 0.99, want: -1},
		{name: "no movement at all", step: 1, want: 0},
	}
	b := NewBuilder(tradingZone(t))
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			candles := geometricCandles(WarmupRows+5, 100, tc.step)
			table, err := b.Build(candles)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			if got := col(t, table, len(table.Rows)-1, "momentum_rsi_14"); got != tc.want {
				t.Fatalf("momentum_rsi_14 = %v, want %v", got, tc.want)
			}
		})
	}
}

// firstMinute is 2026-09-29 03:45 UTC, which is 09:15 IST — the session open.
// Every generated series starts there, so minute_of_session equals the bar index
// for as long as the series lasts, and a UTC/IST mix-up is impossible to miss.
func firstMinute() time.Time {
	return time.Date(2026, 9, 29, 3, 45, 0, 0, time.UTC)
}

// geometricCandles returns n bars with close[i] = base*ratio^i, a fixed OHLC
// envelope around each close and a constant volume.
func geometricCandles(n int, base, ratio float64) []contracts.Candle {
	start := firstMinute()
	out := make([]contracts.Candle, n)
	for i := range out {
		c := base * math.Pow(ratio, float64(i))
		out[i] = contracts.Candle{
			Symbol:    "RELIANCE",
			Timestamp: start.Add(time.Duration(i) * time.Minute),
			Open:      c * 0.999,
			High:      c * 1.002,
			Low:       c * 0.998,
			Close:     c,
			Volume:    500,
		}
	}
	return out
}

// alternatingCandles returns n bars whose close alternates between 100 and 101,
// so every 1-bar log return is the same magnitude with alternating sign.
func alternatingCandles(n int) []contracts.Candle {
	start := firstMinute()
	out := make([]contracts.Candle, n)
	for i := range out {
		c := 100.0
		if i%2 == 1 {
			c = 101
		}
		o := c
		if i > 0 && out[i-1].Close != c {
			o = out[i-1].Close
		}
		out[i] = contracts.Candle{
			Symbol:    "RELIANCE",
			Timestamp: start.Add(time.Duration(i) * time.Minute),
			Open:      o,
			High:      math.Max(o, c) * 1.001,
			Low:       math.Min(o, c) * 0.999,
			Close:     c,
			Volume:    500,
		}
	}
	return out
}

// zeroVolumeCandles is a moving market on which nothing trades.
func zeroVolumeCandles(n int) []contracts.Candle {
	candles := geometricCandles(n, 100, 1.01)
	for i := range candles {
		candles[i].Volume = 0
	}
	return candles
}

// dojiCandles is a moving market where every bar opens, highs, lows and closes
// at the same price: no range, no body, no wicks.
func dojiCandles(n int) []contracts.Candle {
	candles := geometricCandles(n, 100, 1.01)
	for i := range candles {
		candles[i].Open = candles[i].Close
		candles[i].High = candles[i].Close
		candles[i].Low = candles[i].Close
	}
	return candles
}
