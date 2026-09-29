package jobs

import (
	"context"
	"errors"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/testutil"
	cron "github.com/netresearch/go-cron"
	"github.com/prometheus/client_golang/prometheus"
)

// fixedNow anchors every segment boundary in these tests. The worker reads the
// clock through BackfillWorker.now, so a lookback is a deterministic set of
// date ranges rather than something that depends on when the suite ran.
var fixedNow = time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)

// recordedSleeps captures the backoff schedule a retry loop asked for instead
// of waiting it out, so a 429 test asserts on the schedule without spending
// real time on it.
type recordedSleeps struct {
	mu     sync.Mutex
	delays []time.Duration
}

// sleep records d and returns immediately.
func (r *recordedSleeps) sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.delays = append(r.delays, d)
	return nil
}

// all returns the recorded delays in order.
func (r *recordedSleeps) all() []time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]time.Duration(nil), r.delays...)
}

// backfillFixture is a worker wired to a fake Kite over loopback and a
// Parquet store on disk.
type backfillFixture struct {
	worker *BackfillWorker
	fake   *fakeKite
	dir    string
	reg    *prometheus.Registry
	sleeps *recordedSleeps
}

// newBackfillFixture builds a fixture over a fresh temporary directory. The
// lookback and chunk width are set small so the suite stays fast; the
// production chunking rule is asserted separately in
// TestSegmentsCoverTheLookbackContiguously.
func newBackfillFixture(t *testing.T, fake *fakeKite, lookbackDays, chunkDays int, symbols ...string) backfillFixture {
	t.Helper()
	return newBackfillFixtureIn(t, fake, t.TempDir(), lookbackDays, chunkDays, symbols...)
}

// newBackfillFixtureIn builds a fixture over an existing directory, which is
// how a resumed run reuses the segments a previous run landed.
func newBackfillFixtureIn(t *testing.T, fake *fakeKite, dir string, lookbackDays, chunkDays int, symbols ...string) backfillFixture {
	t.Helper()

	baseURL := startFakeKite(t, fake)
	cfg := &config.Config{}
	cfg.Broker.APIKey = "test-key"
	cfg.Broker.AccessToken = "test-token"
	cfg.Broker.RequestTimeoutSeconds = 5
	cfg.Jobs.RateLimitPerSecond = 1000
	cfg.Jobs.HistoricalDir = dir
	cfg.Jobs.BackfillLookbackDays = lookbackDays
	cfg.Broker.Instruments = symbols

	reg := testutil.NewRegistry()
	worker := NewBackfillWorker(cfg,
		newKiteHistory(cfg, masterResolver(), baseURL, testutil.NewLogger()),
		newParquetStore(dir, "minute"),
		NewLimiter(1000),
		testutil.NewLogger(), reg)
	worker.segmentLen = time.Duration(chunkDays) * day
	worker.now = func() time.Time { return fixedNow }
	sleeps := &recordedSleeps{}
	worker.sleep = sleeps.sleep
	// A deterministic retry schedule; the jitter is covered in backoff_test.go.
	worker.backoff = Backoff{Base: 100 * time.Millisecond, Max: time.Second}

	return backfillFixture{worker: worker, fake: fake, dir: dir, reg: reg, sleeps: sleeps}
}

// run invokes a backfill.
func (f backfillFixture) run(ctx context.Context) error { return f.worker.RunBackfill(ctx) }

// landed returns every candles.parquet under the store, sorted by path.
func (f backfillFixture) landed(t *testing.T) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(f.dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, segmentFileName) {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", f.dir, err)
	}
	sort.Strings(out)
	return out
}

// segmentPath returns the path a symbol's single segment is expected at.
func (f backfillFixture) segmentPath(symbol, from, to string) string {
	return filepath.Join(f.dir, "symbol="+symbol, "from="+from, "to="+to, segmentFileName)
}

// candleRows returns the total number of candle rows across every landed
// segment.
func (f backfillFixture) candleRows(t *testing.T) int {
	t.Helper()
	n := 0
	for _, path := range f.landed(t) {
		rows, err := ReadHistoricalCandles(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		n += len(rows)
	}
	return n
}

// metric reads an integer metric value from the fixture registry.
func (f backfillFixture) metric(t *testing.T, name string) int {
	t.Helper()
	return testutil.MetricValue(t, name, f.reg, nil)
}

// threeRowsPerRequest answers a request with three bars inside its range.
func threeRowsPerRequest(t *testing.T) func(*http.Request) reply {
	return func(r *http.Request) reply {
		from, to := rangeOf(t, r)
		var rows [][]any
		for d := 1; d <= 3; d++ {
			ts := from.AddDate(0, 0, d).Add(3*time.Hour + 45*time.Minute)
			if !ts.Before(to) {
				break
			}
			rows = append(rows, []any{
				ts.Format(time.RFC3339), "2543.0", "2549.0", "2535.0", "2545.5", 123456.0,
			})
		}
		return reply{rows: rows}
	}
}

// failsFor answers a request with a status only when the symbol matches, and
// with one in-range bar otherwise.
func failsFor(t *testing.T, status int, body, symbol string) func(*http.Request) reply {
	return func(r *http.Request) reply {
		if strings.Contains(r.URL.Path, "/"+symbol+"/") {
			return reply{status: status, body: body}
		}
		from, _ := rangeOf(t, r)
		return reply{rows: [][]any{
			{from.Add(25 * time.Hour).Format(time.RFC3339), "1", "2", "0.5", "1.5", "10"},
		}}
	}
}

func TestRunBackfillLandsSegmentsInTheParquetStore(t *testing.T) {
	fake := &fakeKite{responder: threeRowsPerRequest(t)}
	fx := newBackfillFixture(t, fake, 120, 60, "RELIANCE")

	if err := fx.run(context.Background()); err != nil {
		t.Fatalf("RunBackfill() error = %v", err)
	}

	// A 120-day lookback at a 60-day chunk is two segments.
	files := fx.landed(t)
	if len(files) != 2 {
		t.Fatalf("landed %d segment files, want 2: %v", len(files), files)
	}
	if want := fx.segmentPath("RELIANCE", "2025-11-01", "2025-12-31"); files[0] != want {
		t.Errorf("first segment = %q, want %q", files[0], want)
	}
	if want := fx.segmentPath("RELIANCE", "2025-12-31", "2026-03-01"); files[1] != want {
		t.Errorf("second segment = %q, want %q", files[1], want)
	}

	if n := fx.candleRows(t); n != 6 {
		t.Errorf("read back %d candle rows, want 6 (3 per segment)", n)
	}
	if got := fx.metric(t, "mft_jobs_backfill_segments_total"); got != 2 {
		t.Errorf("segments_total = %d, want 2", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_candles_total"); got != 6 {
		t.Errorf("candles_total = %d, want 6", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_requests_total"); got != 2 {
		t.Errorf("requests_total = %d, want 2", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_runs_total"); got != 1 {
		t.Errorf("runs_total = %d, want 1", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_last_success_timestamp_seconds"); got != int(fixedNow.Unix()) {
		t.Errorf("last_success = %d, want %d", got, fixedNow.Unix())
	}
}

func TestRunBackfillSecondRunFetchesNothing(t *testing.T) {
	// The resumability contract: a completed backfill is free to repeat.
	fake := &fakeKite{responder: threeRowsPerRequest(t)}
	fx := newBackfillFixture(t, fake, 120, 60, "RELIANCE")

	if err := fx.run(context.Background()); err != nil {
		t.Fatalf("first RunBackfill() error = %v", err)
	}
	firstRequests := fake.count()
	if firstRequests == 0 {
		t.Fatal("the first run made no requests, want the segments fetched")
	}

	if err := fx.run(context.Background()); err != nil {
		t.Fatalf("second RunBackfill() error = %v", err)
	}
	if got := fake.count(); got != firstRequests {
		t.Fatalf("the second run made %d extra requests, want 0: every segment was already on disk", got-firstRequests)
	}
	if got := fx.metric(t, "mft_jobs_backfill_segments_skipped_total"); got != 2 {
		t.Errorf("segments_skipped_total = %d, want 2", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_segments_total"); got != 2 {
		t.Errorf("segments_total = %d, want it to stay at 2 across the second run", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_runs_total"); got != 2 {
		t.Errorf("runs_total = %d, want 2", got)
	}
}

func TestRunBackfillResumesAfterAnInterruption(t *testing.T) {
	// Interrupt the run after its first segment lands, then resume. The
	// landed segment must survive and the resumed run must fetch only the rest.
	dir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel on the second request: the first segment has already landed, so
	// the interruption leaves one durable segment and one unfinished one.
	var seen atomic.Int32
	fake := &fakeKite{responder: threeRowsPerRequest(t)}
	fake.onRequest = func(*http.Request) {
		if seen.Add(1) == 2 {
			cancel()
		}
	}

	fx := newBackfillFixtureIn(t, fake, dir, 180, 60, "RELIANCE")
	// A partial run is not an error — the checkpoint makes the remainder the
	// whole of the next run's work — so the contract under test here is what
	// survived the interrupt, not the return value.
	if err := fx.run(ctx); err != nil {
		t.Fatalf("interrupted RunBackfill() error = %v, want a partial run to be tolerated", err)
	}
	if n := len(fx.landed(t)); n != 1 {
		t.Fatalf("landed %d segments before the interrupt, want 1", n)
	}
	if got := fx.metric(t, "mft_jobs_backfill_segments_total"); got != 1 {
		t.Fatalf("segments_total = %d, want only the first segment to be durable", got)
	}

	resume := newBackfillFixtureIn(t, fake, dir, 180, 60, "RELIANCE")
	if err := resume.run(context.Background()); err != nil {
		t.Fatalf("resumed RunBackfill() error = %v", err)
	}
	if n := len(resume.landed(t)); n != 3 {
		t.Fatalf("the resumed run has %d segments on disk, want 3", n)
	}
	if got := resume.metric(t, "mft_jobs_backfill_requests_total"); got != 2 {
		t.Fatalf("the resumed run made %d requests, want 2: the landed segment was skipped", got)
	}
	if got := resume.metric(t, "mft_jobs_backfill_segments_skipped_total"); got != 1 {
		t.Fatalf("segments_skipped_total = %d, want 1", got)
	}
}

func TestRunBackfillRetriesAfterThrottling(t *testing.T) {
	// Two 429s then success: the request must be retried on an exponential
	// schedule, and the segment must still land.
	fake := &fakeKite{throttles: 2, responder: threeRowsPerRequest(t)}
	fx := newBackfillFixture(t, fake, 60, 60, "RELIANCE")

	if err := fx.run(context.Background()); err != nil {
		t.Fatalf("RunBackfill() error = %v", err)
	}

	if got := fake.count(); got != 3 {
		t.Errorf("made %d requests, want 3 (two throttled, one served)", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_throttles_total"); got != 2 {
		t.Errorf("throttles_total = %d, want 2", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_requests_total"); got != 3 {
		t.Errorf("requests_total = %d, want 3: retried requests still count", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_segments_total"); got != 1 {
		t.Errorf("segments_total = %d, want 1", got)
	}
	if n := len(fx.landed(t)); n != 1 {
		t.Errorf("landed %d segments, want 1", n)
	}

	// The backoff schedule is asserted, not merely implied: 100ms then 200ms.
	want := []time.Duration{100 * time.Millisecond, 200 * time.Millisecond}
	got := fx.sleeps.all()
	if len(got) != len(want) {
		t.Fatalf("backoff delays = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("backoff delay #%d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRunBackfillGivesUpAfterMaxAttempts(t *testing.T) {
	fake := &fakeKite{throttles: maxAttempts + 5}
	fx := newBackfillFixture(t, fake, 60, 60, "RELIANCE")

	err := fx.run(context.Background())
	if !errors.Is(err, broker.ErrRateLimit) {
		t.Fatalf("RunBackfill() error = %v, want it to wrap broker.ErrRateLimit", err)
	}
	if !errors.Is(err, ErrNoHistory) {
		t.Fatalf("RunBackfill() error = %v, want it to wrap ErrNoHistory", err)
	}
	if got := fake.count(); got != maxAttempts {
		t.Errorf("made %d requests, want it to stop at maxAttempts (%d)", got, maxAttempts)
	}
	if n := len(fx.landed(t)); n != 0 {
		t.Errorf("landed %d segments, want 0", n)
	}
	if got := fx.metric(t, "mft_jobs_backfill_runs_failed_total"); got != 1 {
		t.Errorf("runs_failed_total = %d, want 1", got)
	}
}

func TestRunBackfillStopsImmediatelyOnAuthFailure(t *testing.T) {
	// A dead access token will not fix itself on the next segment, so the run
	// must fail fast rather than burning four attempts per segment.
	fake := &fakeKite{defaultReply: reply{status: http.StatusUnauthorized}}
	fx := newBackfillFixture(t, fake, 180, 60, "RELIANCE", "TCS")

	err := fx.run(context.Background())
	if !errors.Is(err, broker.ErrAuth) {
		t.Fatalf("RunBackfill() error = %v, want it to wrap broker.ErrAuth", err)
	}
	if !errors.Is(err, ErrNoHistory) {
		t.Fatalf("RunBackfill() error = %v, want it to wrap ErrNoHistory", err)
	}
	if got := fake.count(); got != 2 {
		t.Errorf("made %d requests, want 2 (one per symbol, no retries)", got)
	}
	if got := len(fx.sleeps.all()); got != 0 {
		t.Errorf("backed off %d times on an auth failure, want 0", got)
	}
	if n := len(fx.landed(t)); n != 0 {
		t.Errorf("landed %d segments on an auth failure, want 0", n)
	}
}

func TestRunBackfillContinuesPastOneFailingSymbol(t *testing.T) {
	// A delisted ticker must not cost the healthy ones their history.
	fake := &fakeKite{responder: failsFor(t, http.StatusNotFound, errorBody, "TCS")}
	fx := newBackfillFixture(t, fake, 60, 60, "RELIANCE", "TCS")

	err := fx.run(context.Background())
	if err != nil {
		t.Fatalf("RunBackfill() error = %v, want a partial run to be tolerated", err)
	}
	if errors.Is(err, ErrNoHistory) {
		t.Fatal("a run that landed a candle must not report ErrNoHistory")
	}
	if got := fake.count(); got != 2 {
		t.Errorf("made %d requests, want 2 (one attempt per symbol)", got)
	}
	if n := len(fx.landed(t)); n != 1 {
		t.Errorf("landed %d segments, want RELIANCE's 1", n)
	}
	if got := fx.metric(t, "mft_jobs_backfill_candles_total"); got != 1 {
		t.Errorf("candles_total = %d, want 1", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_runs_failed_total"); got != 0 {
		t.Errorf("runs_failed_total = %d, want 0 for a partial success", got)
	}
}

func TestRunBackfillIsANoOpWithNoInstruments(t *testing.T) {
	fake := &fakeKite{responder: threeRowsPerRequest(t)}
	fx := newBackfillFixture(t, fake, 730, 60)

	if err := fx.run(context.Background()); err != nil {
		t.Fatalf("RunBackfill() error = %v", err)
	}
	if got := fake.count(); got != 0 {
		t.Fatalf("made %d requests with an empty watchlist, want 0", got)
	}
	if got := fx.metric(t, "mft_jobs_backfill_runs_total"); got != 1 {
		t.Errorf("runs_total = %d, want 1", got)
	}
}

func TestRunBackfillRejectsACancelledContext(t *testing.T) {
	fake := &fakeKite{responder: threeRowsPerRequest(t)}
	fx := newBackfillFixture(t, fake, 730, 60, "RELIANCE")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := fx.run(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("RunBackfill() error = %v, want it to wrap context.Canceled", err)
	}
	if got := fake.count(); got != 0 {
		t.Fatalf("made %d requests on a cancelled context, want 0", got)
	}
}

func TestRunBackfillCoversEveryConfiguredInstrument(t *testing.T) {
	fake := &fakeKite{responder: threeRowsPerRequest(t)}
	fx := newBackfillFixture(t, fake, 60, 60, "RELIANCE", "TCS", "INFY", " reliance ")

	if err := fx.run(context.Background()); err != nil {
		t.Fatalf("RunBackfill() error = %v", err)
	}
	// Three distinct symbols after de-duplicating and normalising the
	// watchlist; the fourth entry repeats the first.
	if n := len(fx.landed(t)); n != 3 {
		t.Fatalf("landed %d segments, want one per distinct symbol (3): %v", n, fx.landed(t))
	}
	for _, symbol := range []string{"RELIANCE", "TCS", "INFY"} {
		path := fx.segmentPath(symbol, "2025-12-31", "2026-03-01")
		if _, err := os.Stat(path); err != nil {
			t.Errorf("missing segment for %s at %s: %v", symbol, path, err)
		}
	}
}

func TestSegmentsCoverTheLookbackContiguously(t *testing.T) {
	cfg := &config.Config{}
	cfg.Jobs.RateLimitPerSecond = 3
	cfg.Jobs.BackfillLookbackDays = 730
	w := NewBackfillWorker(cfg, nil, nil, nil, testutil.NewLogger(), testutil.NewRegistry())

	start := fixedNow.Add(-730 * day)
	segs := w.segmentsFor("RELIANCE", start, fixedNow)
	if len(segs) != 13 {
		t.Fatalf("got %d segments, want 13 (730 days / 60-day chunks)", len(segs))
	}
	if !segs[0].From.Equal(start) {
		t.Errorf("first From = %s, want %s", segs[0].From, start)
	}
	if !segs[len(segs)-1].To.Equal(fixedNow) {
		t.Errorf("last To = %s, want %s", segs[len(segs)-1].To, fixedNow)
	}
	for i, s := range segs {
		if s.Symbol != "RELIANCE" {
			t.Errorf("segment %d symbol = %q, want RELIANCE", i, s.Symbol)
		}
		if s.From.Before(start) || s.To.After(fixedNow) {
			t.Errorf("segment %d (%s) falls outside the lookback", i, s)
		}
		if i > 0 && !s.From.Equal(segs[i-1].To) {
			t.Errorf("segment %d starts at %s but segment %d ends at %s: chunks must be contiguous",
				i, s.From, i-1, segs[i-1].To)
		}
	}
}

func TestSegmentsHandlesALookbackShorterThanOneChunk(t *testing.T) {
	cfg := &config.Config{}
	cfg.Jobs.BackfillLookbackDays = 5
	w := NewBackfillWorker(cfg, nil, nil, nil, testutil.NewLogger(), testutil.NewRegistry())

	segs := w.segmentsFor("TCS", fixedNow.Add(-5*day), fixedNow)
	if len(segs) != 1 {
		t.Fatalf("got %d segments, want 1", len(segs))
	}
	if got := segs[0].To.Sub(segs[0].From); got != 5*day {
		t.Fatalf("the short chunk spans %v, want %v", got, 5*day)
	}
}

func TestSegmentsHandlesAZeroRange(t *testing.T) {
	w := NewBackfillWorker(&config.Config{}, nil, nil, nil, testutil.NewLogger(), testutil.NewRegistry())
	if got := w.segmentsFor("TCS", fixedNow, fixedNow); len(got) != 0 {
		t.Fatalf("segmentsFor() over an empty range = %v, want none", got)
	}
}

func TestNewBackfillWorkerUsesConfig(t *testing.T) {
	cfg := &config.Config{}
	cfg.Jobs.RateLimitPerSecond = 7
	cfg.Jobs.BackfillLookbackDays = 30
	cfg.Jobs.HistoricalDir = "/tmp/hist"
	cfg.Broker.Instruments = []string{"RELIANCE", "TCS"}

	w := NewBackfillWorker(cfg, nil, nil, nil, testutil.NewLogger(), testutil.NewRegistry())
	if got := w.limiter.Rate(); got != 7 {
		t.Errorf("limiter rate = %v, want 7", got)
	}
	if w.lookback != 30*day {
		t.Errorf("lookback = %v, want 720h", w.lookback)
	}
	if got := w.store.(*ParquetStore).Root(); got != "/tmp/hist" {
		t.Errorf("store root = %q, want /tmp/hist", got)
	}
	if got := w.symbols(); len(got) != 2 || got[0] != "RELIANCE" || got[1] != "TCS" {
		t.Errorf("symbols() = %v, want [RELIANCE TCS]", got)
	}
	if w.interval != defaultInterval {
		t.Errorf("interval = %q, want %q", w.interval, defaultInterval)
	}
}

func TestNewBackfillWorkerAppliesDefaultsForAZeroConfig(t *testing.T) {
	w := NewBackfillWorker(&config.Config{}, nil, nil, nil, testutil.NewLogger(), testutil.NewRegistry())
	if got := w.limiter.Rate(); got != defaultRateLimitPerSecond {
		t.Errorf("limiter rate = %v, want the %d default", got, defaultRateLimitPerSecond)
	}
	if w.lookback != defaultLookbackDays*day {
		t.Errorf("lookback = %v, want %d days", w.lookback, defaultLookbackDays)
	}
	if w.segmentLen != segmentDays*day {
		t.Errorf("segment = %v, want %d days", w.segmentLen, segmentDays)
	}
	if got := w.store.(*ParquetStore).Root(); got != "/historical" {
		t.Errorf("store root = %q, want /historical", got)
	}
}

func TestNewLimiterFromConfig(t *testing.T) {
	cfg := &config.Config{}
	cfg.Jobs.RateLimitPerSecond = 4
	if got := NewLimiterFromConfig(cfg).Rate(); got != 4 {
		t.Fatalf("rate = %v, want 4", got)
	}
	cfg.Jobs.RateLimitPerSecond = 0
	if got := NewLimiterFromConfig(cfg).Rate(); got != defaultRateLimitPerSecond {
		t.Fatalf("rate for a zero config = %v, want the %d default", got, defaultRateLimitPerSecond)
	}
}

func TestAllJobMetricsCarryThePlatformPrefix(t *testing.T) {
	reg := testutil.NewRegistry()
	_ = NewBackfillWorker(&config.Config{}, nil, nil, nil, testutil.NewLogger(), reg)

	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	if len(families) == 0 {
		t.Fatal("no metrics were registered")
	}
	for _, f := range families {
		if !strings.HasPrefix(f.GetName(), "mft_") {
			t.Errorf("metric %q is missing the mandatory mft_ prefix", f.GetName())
		}
		if f.GetHelp() == "" {
			t.Errorf("metric %q has no Help string", f.GetName())
		}
	}
}

func TestIsFatalClassifiesBrokerSentinels(t *testing.T) {
	if !isFatal(broker.ErrAuth) {
		t.Error("broker.ErrAuth should be fatal")
	}
	if !isFatal(broker.ErrInstrumentNotFound) {
		t.Error("broker.ErrInstrumentNotFound should be fatal")
	}
	if isFatal(broker.ErrRateLimit) {
		t.Error("broker.ErrRateLimit must not be fatal, it is retried")
	}
	if isFatal(broker.ErrUnavailable) {
		t.Error("broker.ErrUnavailable must not be fatal, it is retried")
	}
}

func TestNewSchedulerIsWiredAndRecoversFromAPanic(t *testing.T) {
	// cron.Recover is the reason a panicking backfill cannot take down the
	// process that also serves the metrics a human needs to diagnose it.
	s := NewScheduler(testutil.NewLogger())
	if s.cron == nil {
		t.Fatal("NewScheduler() returned a nil cron instance")
	}
	if _, err := s.cron.AddFunc("@every 1h", func() {}); err != nil {
		t.Fatalf("AddFunc() error = %v", err)
	}

	logger := cron.VerbosePrintfLogger(cronLogWriter{testutil.NewLogger()})
	ran := false
	wrapped := cron.Recover(logger)(cron.FuncJob(func() {
		ran = true
		panic("backfill exploded")
	}))
	cron.RunJob(context.Background(), wrapped)

	if !ran {
		t.Fatal("the wrapped job never ran")
	}
}
