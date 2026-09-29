package ingestion

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/storage"
	"github.com/mft/core/testutil"
	"github.com/parquet-go/parquet-go"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// base is a fixed trading minute. Every test builds tick timestamps from it,
// so no assertion depends on the wall clock and no test waits for a real
// minute boundary.
var base = time.Date(2026, 8, 3, 10, 0, 0, 0, time.UTC)

// token is the instrument token the fake broker stamps on every tick. A tick
// persisted with a different token cannot be joined to the instrument master.
const token = int64(256265)

// harness is a pipeline wired to temporary storage, a fake clock and a fake
// broker. No test touches the network.
type harness struct {
	t   *testing.T
	p   *Pipeline
	reg *prometheus.Registry
	dir string

	nowMu sync.Mutex
	now   time.Time
}

// clock reads the harness clock. Tests advance it explicitly, which is what
// makes minute rollover deterministic.
func (h *harness) clock() time.Time {
	h.nowMu.Lock()
	defer h.nowMu.Unlock()
	return h.now
}

// advance moves the harness clock forward by d.
func (h *harness) advance(d time.Duration) {
	h.nowMu.Lock()
	defer h.nowMu.Unlock()
	h.now = h.now.Add(d)
}

// metric reads a counter or gauge by name from the pipeline's registry.
func (h *harness) metric(name string) int {
	h.t.Helper()
	return testutil.MetricValue(h.t, name, h.reg, nil)
}

// candles returns every candle persisted under data/candles for symbol,
// ascending by time. Reading from disk rather than from memory is the point:
// it proves the rows actually landed.
func (h *harness) candles(symbol string) []contracts.Candle {
	h.t.Helper()
	root, err := storage.DatasetRoot(h.dir, storage.DatasetCandles)
	if err != nil {
		h.t.Fatalf("dataset root: %v", err)
	}
	reader, err := storage.NewCandleReader(root)
	if err != nil {
		h.t.Fatalf("candle reader: %v", err)
	}
	got, err := reader.Candles(context.Background(), symbol, 0)
	if err != nil {
		h.t.Fatalf("read candles: %v", err)
	}
	return got
}

// flushCandles writes the candle buffer so a reader sees the rows.
func (h *harness) flushCandles() {
	h.t.Helper()
	if err := h.p.candles.Flush(context.Background()); err != nil {
		h.t.Fatalf("flush candles: %v", err)
	}
}

// newHarness builds a pipeline over a temp directory with a fake clock and
// fast backoff, and a sweep cadence long enough that it never fires unless a
// test asks for it.
func newHarness(t *testing.T, streamer broker.Streamer) *harness {
	t.Helper()
	dir := t.TempDir()

	cfg := &config.Config{
		Broker:  config.BrokerConfig{Instruments: []string{"RELIANCE", "TCS"}},
		Storage: config.StorageConfig{DataDir: dir, FlushIntervalSecs: 1, FlushMaxRows: 10000},
	}
	store, err := NewCandleStore(cfg)
	if err != nil {
		t.Fatalf("candle store: %v", err)
	}

	h := &harness{
		t:   t,
		reg: testutil.NewRegistry(),
		dir: dir,
		now: base,
	}

	opts := defaultOptions(cfg)
	opts.clock = h.clock
	opts.sweepEvery = time.Hour
	opts.ageEvery = 5 * time.Millisecond
	opts.reconnectBase = time.Millisecond
	opts.reconnectMax = 5 * time.Millisecond

	h.p = newPipeline(
		streamer,
		fluxkv.New(),
		storage.NewWriter(filepath.Join(dir, "ticks"), 10000),
		store,
		cfg,
		h.reg,
		testutil.NewLogger(),
		opts,
	)
	return h
}

// start brings the pipeline up through the real lifecycle hook.
func (h *harness) start() {
	h.t.Helper()
	if err := h.p.Start(context.Background()); err != nil {
		h.t.Fatalf("start: %v", err)
	}
}

// stop runs the full shutdown and fails on any surfaced error.
func (h *harness) stop() {
	h.t.Helper()
	if err := h.p.Stop(context.Background()); err != nil {
		h.t.Fatalf("stop: %v", err)
	}
}

// runProcessor drives the single writer directly over a closed channel, which
// is exactly the shape the drain path takes at shutdown.
func (h *harness) runProcessor(ticks ...broker.Tick) {
	h.t.Helper()
	queue := make(chan broker.Tick, len(ticks)+1)
	for _, tick := range ticks {
		queue <- tick
	}
	close(queue)
	h.p.process(queue)
}

// waitFor polls cond until it holds or the test gives up.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// scriptedStreamer is a Streamer that records the subscription it was given,
// plays a scripted burst of ticks, and then blocks until its context ends —
// which is what a healthy live stream does.
type scriptedStreamer struct {
	ticks []broker.Tick

	mu      sync.Mutex
	symbols []string
	calls   int
	// failFirst makes the first N calls return err instead of streaming, so a
	// test can prove the supervisor re-invokes Stream.
	failFirst int
	err       error
}

func (s *scriptedStreamer) Stream(ctx context.Context, symbols []string, out chan<- broker.Tick) error {
	s.mu.Lock()
	s.calls++
	n := s.calls
	s.symbols = append([]string(nil), symbols...)
	s.mu.Unlock()

	if n <= s.failFirst {
		return s.err
	}
	for _, tick := range s.ticks {
		select {
		case out <- tick:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	<-ctx.Done()
	return ctx.Err()
}

func (s *scriptedStreamer) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func (s *scriptedStreamer) subscribed() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.symbols...)
}

// idleStreamer blocks forever without emitting a tick.
type idleStreamer struct{}

func (idleStreamer) Stream(ctx context.Context, _ []string, _ chan<- broker.Tick) error {
	<-ctx.Done()
	return ctx.Err()
}

// tickAt builds a RELIANCE tick at ts with the given price.
func tickAt(price float64, ts time.Time) broker.Tick {
	return broker.Tick{Symbol: "RELIANCE", Token: token, Price: price, Volume: 1, Timestamp: ts}
}

// TestStreamSubscribesToConfiguredInstruments is the regression for the nil
// symbols defect: the pipeline must hand core/broker the configured watchlist.
func TestStreamSubscribesToConfiguredInstruments(t *testing.T) {
	streamer := &scriptedStreamer{}
	h := newHarness(t, streamer)

	h.start()
	h.stop()

	want := []string{"RELIANCE", "TCS"}
	got := streamer.subscribed()
	if len(got) != len(want) {
		t.Fatalf("subscribed to %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("subscribed to %v, want %v", got, want)
		}
	}
}

// TestStreamReturnsAndRestarts is the regression for the silent-death defect: a
// Stream that returns while the context is live must be re-invoked, and the
// pipeline must stay usable afterwards.
func TestStreamReturnsAndRestarts(t *testing.T) {
	streamer := &scriptedStreamer{
		ticks:     []broker.Tick{tickAt(100, base)},
		failFirst: 3,
		err:       errors.New("websocket: read: unexpected EOF"),
	}
	h := newHarness(t, streamer)

	h.start()
	waitFor(t, "the tick delivered after the restarts", func() bool {
		return h.metric("mft_ingestion_ticks_processed_total") == 1
	})
	h.stop()

	if got := streamer.callCount(); got < 4 {
		t.Fatalf("Stream called %d times, want at least 4 (3 failures + 1 live)", got)
	}
	if got := h.metric("mft_ingestion_stream_restarts_total"); got != 3 {
		t.Fatalf("stream restarts = %d, want 3", got)
	}
}

// TestStreamNotRestartedAfterStop asserts the supervisor stops calling Stream
// once the context is done, so shutdown does not fight a reconnect loop.
func TestStreamNotRestartedAfterStop(t *testing.T) {
	streamer := &scriptedStreamer{failFirst: 1 << 20, err: errors.New("connection refused")}
	h := newHarness(t, streamer)

	h.start()
	h.stop()
	after := streamer.callCount()

	time.Sleep(50 * time.Millisecond)
	if got := streamer.callCount(); got != after {
		t.Fatalf("Stream called %d times after stop, want %d", got, after)
	}
}

// TestFoldBuildsCandle covers the ordinary path: OHLCV folding into fluxKV, the
// raw tick reaching the tick store, and the live candle being readable.
func TestFoldBuildsCandle(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	h.runProcessor(
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 5, Timestamp: base},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 105, Volume: 3, Timestamp: base.Add(20 * time.Second)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 95, Volume: 2, Timestamp: base.Add(40 * time.Second)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 101, Volume: 1, Timestamp: base.Add(50 * time.Second)},
	)

	c := h.p.cache.Candle("RELIANCE")
	if c == nil {
		t.Fatal("expected a live candle in the cache")
	}
	want := fluxkv.Candle{
		Symbol: "RELIANCE", Timestamp: base,
		Open: 100, High: 105, Low: 95, Close: 101, Volume: 11,
	}
	if *c != want {
		t.Fatalf("candle = %+v, want %+v", *c, want)
	}
	if got := h.metric("mft_ingestion_ticks_processed_total"); got != 4 {
		t.Fatalf("ticks processed = %d, want 4", got)
	}
	if got := h.p.ticks.Pending(); got != 4 {
		t.Fatalf("tick rows buffered = %d, want 4", got)
	}
}

// TestAppendErrorIsSurfaced is the regression for the discarded Append return
// value: a refused write must reach Err rather than vanish.
func TestAppendErrorIsSurfaced(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	// Closing the writer makes every subsequent Append return ErrClosed.
	if err := h.p.ticks.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}

	h.runProcessor(tickAt(100, base))

	err := h.p.Err()
	if err == nil {
		t.Fatal("Err = nil, want the refused append to be surfaced")
	}
	if !errors.Is(err, storage.ErrClosed) {
		t.Fatalf("Err = %v, want it to wrap storage.ErrClosed", err)
	}
	if got := h.metric("mft_ingestion_storage_append_errors_total"); got != 1 {
		t.Fatalf("append errors = %d, want 1", got)
	}
	// The candle is still folded: a storage failure on the tick row must not
	// cost the aggregation, which lives in fluxKV and is the hot path.
	if h.p.cache.Candle("RELIANCE") == nil {
		t.Fatal("expected the candle to be folded despite the append failure")
	}
}

// TestCandleAppendErrorIsSurfaced is the same regression for the candle path.
func TestCandleAppendErrorIsSurfaced(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	if err := h.p.candles.Close(); err != nil {
		t.Fatalf("close candle writer: %v", err)
	}

	h.runProcessor(
		tickAt(100, base),
		tickAt(101, base.Add(time.Minute)),
		tickAt(102, base.Add(2*time.Minute)),
	)

	if err := h.p.Err(); !errors.Is(err, storage.ErrClosed) {
		t.Fatalf("Err = %v, want it to wrap storage.ErrClosed", err)
	}
	if got := h.metric("mft_ingestion_candles_completed_total"); got != 0 {
		t.Fatalf("candles completed = %d, want 0 when the append failed", got)
	}
	if got := h.metric("mft_ingestion_storage_append_errors_total"); got != 2 {
		t.Fatalf("append errors = %d, want 2", got)
	}
}

// TestRolloverWritesCandleExactlyOnce is the core invariant: every finished bar
// lands on disk, and none lands twice.
func TestRolloverWritesCandleExactlyOnce(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	const minutes = 5
	ticks := make([]broker.Tick, 0, minutes*5)
	for minute := range minutes {
		for _, offset := range []time.Duration{0, 15 * time.Second, 30 * time.Second, 45 * time.Second} {
			ticks = append(ticks, broker.Tick{
				Symbol: "RELIANCE", Token: token,
				Price:     float64(100 + minute),
				Volume:    2,
				Timestamp: base.Add(time.Duration(minute)*time.Minute + offset),
			})
		}
	}
	// A tick in the sixth minute is what closes the fifth bar.
	ticks = append(ticks, tickAt(105, base.Add(time.Duration(minutes)*time.Minute)))

	h.runProcessor(ticks...)
	h.flushCandles()

	got := h.candles("RELIANCE")
	if len(got) != minutes {
		t.Fatalf("persisted %d candles, want %d: %+v", len(got), minutes, got)
	}
	for i, candle := range got {
		want := base.Add(time.Duration(i) * time.Minute)
		if !candle.Timestamp.Equal(want) {
			t.Fatalf("candle %d at %s, want %s", i, candle.Timestamp, want)
		}
		if candle.Open != float64(100+i) || candle.Close != float64(100+i) {
			t.Fatalf("candle %d = O%.0f C%.0f, want O%.0f C%.0f",
				i, candle.Open, candle.Close, float64(100+i), float64(100+i))
		}
		if candle.High != float64(100+i) || candle.Low != float64(100+i) {
			t.Fatalf("candle %d high/low = %.0f/%.0f, want %.0f/%.0f",
				i, candle.High, candle.Low, float64(100+i), float64(100+i))
		}
		if candle.Volume != 8 {
			t.Fatalf("candle %d volume = %d, want 8", i, candle.Volume)
		}
	}
	if got := h.metric("mft_ingestion_candles_completed_total"); got != minutes {
		t.Fatalf("candles completed = %d, want %d", got, minutes)
	}
}

// TestBoundaryTickLandsInTheRightBar pins minute truncation. A tick at
// 10:00:59.999 and one at 10:01:00.000 are different bars, and the second must
// not leak backwards into the first.
func TestBoundaryTickLandsInTheRightBar(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	h.runProcessor(
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 10, Volume: 1,
			Timestamp: base.Add(59*time.Second + 999*time.Millisecond)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 20, Volume: 4,
			Timestamp: base.Add(time.Minute)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 30, Volume: 1,
			Timestamp: base.Add(2 * time.Minute)},
	)
	h.flushCandles()

	got := h.candles("RELIANCE")
	if len(got) != 2 {
		t.Fatalf("persisted %d candles, want 2: %+v", len(got), got)
	}
	first, second := got[0], got[1]
	if !first.Timestamp.Equal(base) || first.Open != 10 || first.Close != 10 || first.Volume != 1 {
		t.Fatalf("first bar = %+v, want {O10 C10 V1} at %s", first, base)
	}
	if !second.Timestamp.Equal(base.Add(time.Minute)) || second.Open != 20 || second.Volume != 4 {
		t.Fatalf("second bar = %+v, want {O20 V4} at %s", second, base.Add(time.Minute))
	}
}

// TestLateTickDoesNotRewriteAClosedBar covers a reordered frame or a reconnect
// replaying the tail of the session. Reopening a persisted minute would write
// the bar twice.
func TestLateTickDoesNotRewriteAClosedBar(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	h.runProcessor(
		tickAt(10, base),
		tickAt(20, base.Add(time.Minute)),
		// Late: belongs to the minute already written.
		tickAt(99, base.Add(30*time.Second)),
		tickAt(30, base.Add(2*time.Minute)),
	)
	h.flushCandles()

	got := h.candles("RELIANCE")
	if len(got) != 2 {
		t.Fatalf("persisted %d candles, want 2: %+v", len(got), got)
	}
	if got[0].Volume != 1 || got[0].Close != 10 {
		t.Fatalf("first bar = %+v, want it untouched by the late tick", got[0])
	}
	if n := h.metric("mft_ingestion_ticks_late_total"); n != 1 {
		t.Fatalf("late ticks = %d, want 1", n)
	}
	// Only the aggregation rejects it; the raw tick is still archived.
	if n := h.metric("mft_ingestion_ticks_processed_total"); n != 4 {
		t.Fatalf("ticks processed = %d, want 4", n)
	}
	if got := h.p.ticks.Pending(); got != 4 {
		t.Fatalf("tick rows buffered = %d, want 4: the late tick is still archived", got)
	}
}

// TestSweepClosesAQuietBar is the case a tick-driven rollover cannot catch: a
// symbol that stops trading mid-minute produces no later tick, so nothing ever
// rolls over. The clock sweep is what writes the bar.
func TestSweepClosesAQuietBar(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	h.runProcessor(tickAt(10, base))
	h.flushCandles()
	if got := h.candles("RELIANCE"); len(got) != 0 {
		t.Fatalf("persisted %d candles before the minute rolled, want 0", len(got))
	}

	h.p.sweep(h.clock().Add(time.Minute))
	h.flushCandles()

	got := h.candles("RELIANCE")
	if len(got) != 1 {
		t.Fatalf("persisted %d candles after the sweep, want 1", len(got))
	}
	if !got[0].Timestamp.Equal(base) || got[0].Close != 10 {
		t.Fatalf("swept bar = %+v, want it at %s with close 10", got[0], base)
	}

	// A second sweep in the same minute must not write it again.
	h.p.sweep(h.clock().Add(time.Minute))
	h.flushCandles()
	if got := h.candles("RELIANCE"); len(got) != 1 {
		t.Fatalf("persisted %d candles after a repeat sweep, want 1", len(got))
	}
	if n := h.metric("mft_ingestion_candles_completed_total"); n != 1 {
		t.Fatalf("candles completed = %d, want 1", n)
	}
}

// TestSweepUsesTheInjectedClock proves rollover is driven by the Clock seam
// and not by wall time, which is what makes it testable at all.
func TestSweepUsesTheInjectedClock(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	h.runProcessor(
		tickAt(10, base),
		tickAt(11, base.Add(time.Minute)),
	)

	// A sweep taken at the open bar's own minute closes nothing.
	h.p.sweep(h.clock())
	h.flushCandles()
	if got := h.candles("RELIANCE"); len(got) != 1 {
		t.Fatalf("persisted %d candles, want 1 before the minute rolls", len(got))
	}

	// Moving the injected clock past the boundary closes the second bar.
	h.advance(2 * time.Minute)
	h.p.sweep(h.clock())
	h.flushCandles()
	if got := h.candles("RELIANCE"); len(got) != 2 {
		t.Fatalf("persisted %d candles after the clock advanced, want 2", len(got))
	}
}

// TestProcessorDrainsOnShutdown runs the whole lifecycle: ticks in through the
// real broker channel, then a full stop. Every tick must be folded, and every
// finished bar must be on disk afterwards.
func TestProcessorDrainsOnShutdown(t *testing.T) {
	const (
		symbol  = "RELIANCE"
		minutes = 4
		perMin  = 25
	)

	ticks := make([]broker.Tick, 0, minutes*perMin)
	for minute := range minutes {
		for i := range perMin {
			ticks = append(ticks, broker.Tick{
				Symbol: symbol, Token: token,
				Price:     float64(100 + minute + i%3),
				Volume:    1,
				Timestamp: base.Add(time.Duration(minute)*time.Minute + time.Duration(i)*time.Second),
			})
		}
	}

	h := newHarness(t, &scriptedStreamer{ticks: ticks})
	h.start()
	waitFor(t, "every tick to be processed", func() bool {
		return h.metric("mft_ingestion_ticks_processed_total") == minutes*perMin
	})
	h.stop()

	// The final minute has no later tick and the sweep never fired, so it is
	// deliberately not persisted: it is not a complete bar. Every earlier
	// minute is on disk exactly once.
	got := h.candles(symbol)
	want := minutes - 1
	if len(got) != want {
		t.Fatalf("persisted %d candles, want %d: %+v", len(got), want, got)
	}
	for i, candle := range got {
		ts := base.Add(time.Duration(i) * time.Minute)
		if !candle.Timestamp.Equal(ts) {
			t.Fatalf("candle %d at %s, want %s", i, candle.Timestamp, ts)
		}
		if candle.Volume != int64(perMin) {
			t.Fatalf("candle %d volume = %d, want %d", i, candle.Volume, perMin)
		}
	}
	if n := h.metric("mft_ingestion_candles_completed_total"); n != want {
		t.Fatalf("candles completed = %d, want %d", n, want)
	}
	if h.p.Err() != nil {
		t.Fatalf("Err after a clean shutdown = %v, want nil", h.p.Err())
	}
}

// TestStartSurfacesWriterFailure is the regression for the ignored Start error.
func TestStartSurfacesWriterFailure(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	h.start()
	defer h.stop()

	if err := h.p.Start(context.Background()); err == nil {
		t.Fatal("second Start = nil, want an error naming the already-started writer")
	}
}

// TestTickRowCarriesInstrumentToken is the regression for the hardcoded
// InstrumentToken: a zero token makes the tick dataset unjoinable with the
// instrument master.
func TestTickRowCarriesInstrumentToken(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	h.runProcessor(
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 7, Timestamp: base},
	)
	if err := h.p.ticks.Close(); err != nil {
		t.Fatalf("close tick writer: %v", err)
	}

	rows := readTickRows(t, filepath.Join(h.dir, "ticks"))
	if len(rows) != 1 {
		t.Fatalf("persisted %d tick rows, want 1", len(rows))
	}
	if rows[0].InstrumentToken != token {
		t.Fatalf("instrument_token = %d, want %d", rows[0].InstrumentToken, token)
	}
	if rows[0].Symbol != "RELIANCE" || rows[0].LastPrice != 100 || rows[0].Volume != 7 {
		t.Fatalf("tick row = %+v", rows[0])
	}
	if rows[0].Timestamp != base.UnixMilli() {
		t.Fatalf("timestamp = %d, want %d", rows[0].Timestamp, base.UnixMilli())
	}
}

// TestBackpressureDropsOldest is the overload policy test. The reader is
// stalled, so the queue fills; the newest tick must survive, the oldest must be
// evicted, and the drop must be counted.
func TestBackpressureDropsOldest(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	const size = 4
	queue := make(chan broker.Tick, size)

	for i := range size {
		h.p.deliver(tickAt(float64(i), base), queue)
	}
	if n := h.metric("mft_ingestion_ticks_dropped_total"); n != 0 {
		t.Fatalf("drops = %d before overload, want 0", n)
	}

	// Two more arrive with the reader still stalled. Drop-oldest evicts ticks
	// 0 and 1 and keeps 2, 3, 4, 5.
	h.p.deliver(tickAt(4, base), queue)
	h.p.deliver(tickAt(5, base), queue)

	if got := len(queue); got != size {
		t.Fatalf("queue depth = %d, want %d", got, size)
	}
	if n := h.metric("mft_ingestion_ticks_dropped_total"); n != 2 {
		t.Fatalf("drops = %d, want 2", n)
	}
	if n := h.metric("mft_ingestion_tick_queue_depth"); n != size {
		t.Fatalf("queue depth gauge = %d, want %d", n, size)
	}

	prices := make([]float64, 0, size)
	for len(queue) > 0 {
		prices = append(prices, (<-queue).Price)
	}
	want := []float64{2, 3, 4, 5}
	if len(prices) != len(want) {
		t.Fatalf("queue holds %v, want %v", prices, want)
	}
	for i := range want {
		if prices[i] != want[i] {
			t.Fatalf("queue holds %v, want %v (newest kept, oldest evicted)", prices, want)
		}
	}
}

// TestBackpressureNeverBlocksTheReader is the property that motivated the
// policy: deliver must return promptly even when nothing is draining, because
// a blocked sender stalls the broker's socket reader.
func TestBackpressureNeverBlocksTheReader(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	queue := make(chan broker.Tick, 2)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := range 1000 {
			h.p.deliver(tickAt(float64(i), base), queue)
		}
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("deliver blocked with a full queue and no reader")
	}
	if n := h.metric("mft_ingestion_ticks_dropped_total"); n != 998 {
		t.Fatalf("drops = %d, want 998", n)
	}
}

// TestRelayDrainsUpstreamAndClosesQueue asserts the relay moves every tick
// across and closes the queue, which is what lets the processor finish its
// range at shutdown.
func TestRelayDrainsUpstreamAndClosesQueue(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	const n = 50
	upstream := make(chan broker.Tick, n)
	for i := range n {
		upstream <- tickAt(float64(i), base)
	}
	close(upstream)

	queue := make(chan broker.Tick, n)
	h.p.relay(upstream, queue)

	var got []float64
	for tick := range queue {
		got = append(got, tick.Price)
	}
	if len(got) != n {
		t.Fatalf("relayed %d ticks, want %d", len(got), n)
	}
	for i, price := range got {
		if price != float64(i) {
			t.Fatalf("tick %d has price %v, want %d: the relay must preserve order", i, price, i)
		}
	}
}

// TestConcurrentTicksSingleWriter is the -race proof for the single-writer
// discipline. Many producers feed the queue while one processor folds; nothing
// else may touch the writers, the cache or the state map, so the run must be
// clean and every tick must be accounted for exactly once.
func TestConcurrentTicksSingleWriter(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	const (
		symbols = 4
		prods   = 8
		each    = 250
	)
	names := []string{"RELIANCE", "TCS", "INFY", "HDFC"}
	total := prods * each

	queue := make(chan broker.Tick, total)
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.p.process(queue)
	}()

	var produced atomic.Int64
	var wg sync.WaitGroup
	for prod := range prods {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range each {
				n := prod*each + i
				queue <- broker.Tick{
					Symbol:    names[n%symbols],
					Token:     token,
					Price:     float64(n%50 + 1),
					Volume:    1,
					Timestamp: base.Add(time.Duration(n/10) * time.Minute),
				}
				produced.Add(1)
			}
		}()
	}

	wg.Wait()
	close(queue)
	<-done

	processed := h.metric("mft_ingestion_ticks_processed_total")
	dropped := h.metric("mft_ingestion_ticks_dropped_total")
	if processed+dropped != int(produced.Load()) {
		t.Fatalf("processed %d + dropped %d = %d, want %d produced",
			processed, dropped, processed+dropped, produced.Load())
	}
	if dropped != 0 {
		t.Fatalf("dropped %d ticks into a %d-deep queue, want 0", dropped, total)
	}

	// The state map is internally consistent: a symbol's open bar is strictly
	// newer than its watermark. A concurrent fold that broke the single-writer
	// rule would be free to violate this.
	for symbol, st := range h.p.state {
		if st.closed.IsZero() || st.open.Symbol == "" {
			continue
		}
		if !st.open.Timestamp.After(st.closed) {
			t.Fatalf("symbol %s: open bar %s is not after watermark %s",
				symbol, st.open.Timestamp, st.closed)
		}
	}
}

// TestLastTickAgeTracksTheClock is the wedge detector. The gauge is sampled on
// its own goroutine and read through the injected clock, so a stream that goes
// quiet makes the age climb — which is the whole reason it is not updated from
// inside the processor that might be the thing that wedged.
func TestLastTickAgeTracksTheClock(t *testing.T) {
	h := newHarness(t, &scriptedStreamer{ticks: []broker.Tick{tickAt(100, base)}})

	h.start()
	defer h.stop()

	waitFor(t, "the first tick to be processed", func() bool {
		return h.metric("mft_ingestion_ticks_processed_total") == 1
	})
	if got := h.metric("mft_ingestion_last_tick_age_seconds"); got != 0 {
		t.Fatalf("last tick age = %d with a fresh tick, want 0", got)
	}

	// Five minutes of market time pass with no further tick.
	h.advance(5 * time.Minute)
	waitFor(t, "the last-tick age to reach five minutes", func() bool {
		return h.metric("mft_ingestion_last_tick_age_seconds") == 300
	})
}

// TestShutdownDropsAnIncompleteBar pins the deliberate choice at shutdown: a
// bar still inside its minute is not a complete bar, and publishing a short
// OHLCV the feature pipeline cannot distinguish from a full one would be worse
// than dropping it. The raw ticks it came from are already durable.
func TestShutdownDropsAnIncompleteBar(t *testing.T) {
	h := newHarness(t, &scriptedStreamer{ticks: []broker.Tick{tickAt(100, base)}})

	h.start()
	waitFor(t, "the tick to be processed", func() bool {
		return h.metric("mft_ingestion_ticks_processed_total") == 1
	})

	// The minute has not rolled yet, so stopping here must not write the bar.
	h.stop()
	if got := h.candles("RELIANCE"); len(got) != 0 {
		t.Fatalf("persisted %d candles for an incomplete bar, want 0", len(got))
	}
}

// TestShutdownPersistsABarThatJustCompleted is the same shutdown path one
// minute later: the bar is complete, so it is written.
func TestShutdownPersistsABarThatJustCompleted(t *testing.T) {
	h := newHarness(t, &scriptedStreamer{ticks: []broker.Tick{tickAt(100, base)}})

	h.start()
	waitFor(t, "the tick to be processed", func() bool {
		return h.metric("mft_ingestion_ticks_processed_total") == 1
	})

	h.advance(time.Minute)
	h.stop()

	got := h.candles("RELIANCE")
	if len(got) != 1 {
		t.Fatalf("persisted %d candles, want 1: a completed bar must not be lost to shutdown", len(got))
	}
	if !got[0].Timestamp.Equal(base) || got[0].Close != 100 {
		t.Fatalf("persisted bar = %+v, want it at %s with close 100", got[0], base)
	}
}

// TestAttachBrokerLoggerIsOptional proves the module still wires up with a
// Streamer that has no logger hook, which is what keeps the ingestion module
// decoupled from the concrete broker connector.
func TestAttachBrokerLoggerIsOptional(t *testing.T) {
	AttachBrokerLogger(idleStreamer{}, testutil.NewLogger())

	withHook := &loggerCaptureStreamer{}
	AttachBrokerLogger(withHook, testutil.NewLogger())
	if withHook.log == nil {
		t.Fatal("AttachBrokerLogger did not reach a Streamer that exposes SetLogger")
	}
}

// loggerCaptureStreamer records the logger handed to it.
type loggerCaptureStreamer struct {
	idleStreamer
	log *zap.Logger
}

func (l *loggerCaptureStreamer) SetLogger(log *zap.Logger) { l.log = log }

// TestHealthCheckDegradesOnAppendFailure proves /readyz reflects a broken store
// instead of serving a healthy process that has silently stopped persisting.
func TestHealthCheckDegradesOnAppendFailure(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	RegisterHealth(h.p)
	if err := h.p.Check(context.Background()); err != nil {
		t.Fatalf("Check on a healthy pipeline = %v, want nil", err)
	}

	if err := h.p.ticks.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	h.runProcessor(tickAt(1, base))

	if err := h.p.Check(context.Background()); err == nil {
		t.Fatal("Check after a refused append = nil, want an error")
	}
}

// TestBackoffSaturates covers the overflow guard: a stream that has been down
// for hours must not produce a zero or negative delay and turn into a hot loop.
func TestBackoffSaturates(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	for _, tc := range []struct {
		attempt int
		want    time.Duration
	}{
		{-1, time.Millisecond},
		{0, time.Millisecond},
		{1, 2 * time.Millisecond},
		{2, 4 * time.Millisecond},
		{5, 5 * time.Millisecond},
		{1000, 5 * time.Millisecond},
	} {
		if got := h.p.backoffFor(tc.attempt); got != tc.want {
			t.Fatalf("backoffFor(%d) = %v, want %v", tc.attempt, got, tc.want)
		}
	}
}

// TestCandleStoreRejectsNil guards the fx constructors against a nil config,
// which would otherwise panic deep inside the writer.
func TestCandleStoreRejectsNil(t *testing.T) {
	if _, err := NewCandleStore(nil); err == nil {
		t.Fatal("NewCandleStore(nil) = nil error, want an error")
	}
	if _, err := NewCandleStoreFromWriter(nil); err == nil {
		t.Fatal("NewCandleStoreFromWriter(nil) = nil error, want an error")
	}
}

// TestModuleProvidesBothWriters is the FX wiring check: the graph must resolve
// a tick writer and a candle writer without core/fx.go changing, which is the
// entire point of the local CandleStore type.
func TestModuleProvidesBothWriters(t *testing.T) {
	cfg := &config.Config{
		Broker:  config.BrokerConfig{Instruments: []string{"RELIANCE"}},
		Storage: config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: 1, FlushMaxRows: 100},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}

	var got CandleStore
	app := fx.New(
		fx.Supply(cfg),
		fx.Provide(
			func() *prometheus.Registry { return testutil.NewRegistry() },
			func() *zap.Logger { return testutil.NewLogger() },
			fluxkv.New,
			func() broker.Streamer { return idleStreamer{} },
			func() (*storage.Writer, error) { return storage.NewWriterFromConfig(cfg.Storage) },
			NewCandleStore,
			NewPipeline,
		),
		fx.Invoke(func(s CandleStore, p *Pipeline) { got = s; _ = p }),
	)

	if err := app.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := app.Stop(context.Background()); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if got.writer == nil {
		t.Fatal("fx graph did not resolve the candle writer")
	}
}

// readTickRows decodes every published tick parquet file under root.
func readTickRows(t *testing.T, root string) []storage.Tick {
	t.Helper()
	var rows []storage.Tick
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if d.IsDir() || filepath.Ext(path) != ".parquet" {
			return nil
		}
		got, rerr := parquet.ReadFile[storage.Tick](path)
		if rerr != nil {
			return rerr
		}
		rows = append(rows, got...)
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
	return rows
}
