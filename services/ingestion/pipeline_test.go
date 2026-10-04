package ingestion

import (
	"context"
	"errors"
	"fmt"
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
	opts.appendRetryEvery = time.Millisecond
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
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 100, Timestamp: base},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 105, Volume: 103, Timestamp: base.Add(20 * time.Second)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 95, Volume: 105, Timestamp: base.Add(40 * time.Second)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 101, Volume: 106, Timestamp: base.Add(50 * time.Second)},
	)

	c := h.p.cache.Candle("RELIANCE")
	if c == nil {
		t.Fatal("expected a live candle in the cache")
	}
	want := fluxkv.Candle{
		Symbol: "RELIANCE", Timestamp: base,
		Open: 100, High: 105, Low: 95, Close: 101, Volume: 6,
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

func TestVolumeDeltaHandlesCumulativeReplayAndSessionReset(t *testing.T) {
	p := &Pipeline{}
	var st symbolState
	open := time.Date(2026, 8, 3, 9, 15, 0, 0, indiaLocation)
	feed := func(at time.Time, volume int64) int64 {
		return p.volumeDelta(&st, broker.Tick{Timestamp: at, Volume: volume})
	}
	if got := feed(open, 100); got != 100 {
		t.Fatalf("opening volume delta = %d, want 100", got)
	}
	if got := feed(open.Add(30*time.Second), 110); got != 10 {
		t.Fatalf("volume delta = %d, want 10", got)
	}
	if got := feed(open.Add(30*time.Second), 112); got != 2 {
		t.Fatalf("same-second volume delta = %d, want 2", got)
	}
	if got := feed(open.Add(30*time.Second), 112); got != 0 {
		t.Fatalf("exact replay volume delta = %d, want 0", got)
	}
	if got := feed(open.Add(30*time.Second), 110); got != 0 {
		t.Fatalf("same-second replay regression delta = %d, want 0", got)
	}
	if got := feed(open.Add(20*time.Second), 105); got != 0 {
		t.Fatalf("replayed volume delta = %d, want 0", got)
	}
	if got := feed(open.Add(time.Minute), 120); got != 8 {
		t.Fatalf("post-replay volume delta = %d, want 8", got)
	}
	if got := feed(open.Add(2*time.Minute), 40); got != 0 {
		t.Fatalf("volume reset delta = %d, want 0", got)
	}
	if got := feed(open.Add(3*time.Minute), 48); got != 8 {
		t.Fatalf("post-reset volume delta = %d, want 8", got)
	}
	nextDay := open.AddDate(0, 0, 1)
	if got := feed(nextDay, 7); got != 7 {
		t.Fatalf("new-session opening delta = %d, want 7", got)
	}
}

func TestFoldRejectsMinuteRegressionAndSameMinuteTimestampRegression(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	start := base.Add(9*time.Hour + 15*time.Minute)
	h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 100, Timestamp: start})
	h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 102, Volume: 120, Timestamp: start.Add(2 * time.Minute)})
	if st := h.p.state["RELIANCE"]; !st.open.Timestamp.Equal(start.Add(2 * time.Minute)) {
		t.Fatalf("open minute regressed to %s", st.open.Timestamp)
	}
	before := h.p.state["RELIANCE"].open
	h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 90, Volume: 110, Timestamp: start.Add(time.Minute)})
	if got := h.p.state["RELIANCE"].open; got != before {
		t.Fatalf("older-minute tick changed open candle: got %+v want %+v", got, before)
	}
	h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 104, Volume: 121, Timestamp: start.Add(2*time.Minute + 20*time.Second)})
	h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 80, Volume: 119, Timestamp: start.Add(2*time.Minute + 10*time.Second)})
	if got := h.p.state["RELIANCE"].open.Close; got != 104 {
		t.Fatalf("same-minute reordered tick regressed close to %v", got)
	}
}

func TestRolloverBackpressuresThroughMoreThanQueueCapacity(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	store := h.p.candles
	remaining := tickBufferSize + 3
	rejected := make(chan struct{})
	resume := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(resume)
		}
	}()
	store.appendWithStatus = func(row storage.Candle) (bool, error) {
		if remaining > 0 {
			if remaining == tickBufferSize+2 {
				close(rejected)
				<-resume
			}
			remaining--
			return false, storage.ErrClosed
		}
		return store.writer.AppendWithStatus(row)
	}
	h.p.candles = store
	open := base.Add(9*time.Hour + 15*time.Minute)
	done := make(chan struct{})
	go func() {
		h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 100, Timestamp: open})
		h.p.fold(broker.Tick{Symbol: "RELIANCE", Token: token, Price: 101, Volume: 110, Timestamp: open.Add(time.Minute)})
		close(done)
	}()
	<-rejected
	if err := h.p.Check(context.Background()); !errors.Is(err, storage.ErrClosed) {
		t.Fatalf("readiness during writer rejection = %v, want storage.ErrClosed", err)
	}
	close(resume)
	released = true
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("rollover did not recover after transient writer rejection")
	}
	if remaining != 0 {
		t.Fatalf("writer rejection attempts remaining = %d", remaining)
	}
	if err := h.p.Check(context.Background()); err != nil {
		t.Fatalf("readiness after successful recovery = %v, want nil", err)
	}
	if got := h.p.state["RELIANCE"].open; !got.Timestamp.Equal(open.Add(time.Minute)) {
		t.Fatalf("rollover tick was not folded after recovery: %+v", got)
	}
	if err := h.p.candles.Flush(context.Background()); err != nil {
		t.Fatalf("flush recovered candle: %v", err)
	}
	got := h.candles("RELIANCE")
	if len(got) != 1 || got[0].Timestamp != open {
		t.Fatalf("persisted candles after recovery = %+v, want first bar at %s", got, open)
	}
}

func TestBrokerTicksReachParquetWithSessionVolume(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	open := time.Date(2026, 8, 3, 9, 15, 0, 0, indiaLocation)
	h.runProcessor(
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 100, Timestamp: open},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 101, Volume: 110, Timestamp: open.Add(30 * time.Second)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 102, Volume: 120, Timestamp: open.Add(time.Minute)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 103, Volume: 130, Timestamp: open.Add(90 * time.Second)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 104, Volume: 140, Timestamp: open.Add(2 * time.Minute)},
	)
	if err := h.p.ticks.Flush(context.Background()); err != nil {
		t.Fatalf("flush ticks: %v", err)
	}
	got := h.candles("RELIANCE")
	if len(got) != 2 || got[0].Volume != 110 || got[1].Volume != 20 {
		t.Fatalf("persisted candles = %+v, want session deltas [110,20]", got)
	}
	files, err := filepath.Glob(filepath.Join(h.dir, "ticks", "symbol=RELIANCE", "date=2026-08-03", "part-*.parquet"))
	if err != nil || len(files) == 0 {
		t.Fatalf("tick parquet files = %v, %v", files, err)
	}
	rows, err := parquet.ReadFile[storage.Tick](files[0])
	if err != nil {
		t.Fatalf("read raw tick parquet: %v", err)
	}
	if len(rows) != 5 || rows[1].Volume != 110 {
		t.Fatalf("raw tick rows = %+v, want original cumulative volumes", rows)
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
	if h.p.cache.Candle("RELIANCE") != nil {
		t.Fatal("rejected raw tick must not advance the candle or its volume watermark")
	}
}

func TestRawTickCapacityBackpressuresUntilRecovery(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	var refusals atomic.Int64
	refused := make(chan struct{})
	recover := make(chan struct{})
	h.p.appendTick = func(row storage.Tick) (bool, error) {
		attempt := refusals.Add(1)
		if attempt == 1 {
			return h.p.ticks.AppendWithStatus(row)
		}
		if attempt <= 3 {
			if attempt == 3 {
				close(refused)
				<-recover
			}
			return false, fmt.Errorf("injected capacity refusal: %w", storage.ErrCapacity)
		}
		return h.p.ticks.AppendWithStatus(row)
	}
	first := broker.Tick{Symbol: "RELIANCE", Token: token, Price: 100, Volume: 100, Timestamp: base}
	if !h.p.handle(first) {
		t.Fatal("first tick was rejected")
	}
	before := h.p.state["RELIANCE"]
	second := broker.Tick{Symbol: "RELIANCE", Token: token, Price: 101, Volume: 115, Timestamp: base.Add(time.Second)}
	done := make(chan bool, 1)
	go func() { done <- h.p.handle(second) }()
	select {
	case <-refused:
	case <-time.After(time.Second):
		t.Fatal("raw tick did not reach capacity refusal")
	}
	if err := h.p.Check(context.Background()); !errors.Is(err, storage.ErrCapacity) {
		t.Fatalf("readiness during raw capacity refusal = %v, want storage.ErrCapacity", err)
	}
	if got := h.p.state["RELIANCE"]; got != before {
		t.Fatalf("state advanced while raw row was rejected: got %+v want %+v", got, before)
	}
	candleErr := errors.New("unresolved candle write")
	h.p.setStorageError(candleErr)
	pendingCandle := contracts.Candle{Symbol: "RELIANCE", Timestamp: base, Open: 100, High: 100, Low: 100, Close: 100}
	if err := h.p.candles.Append(storage.NewCandle(pendingCandle)); err != nil {
		t.Fatalf("append pending candle: %v", err)
	}
	close(recover)
	select {
	case accepted := <-done:
		if !accepted {
			t.Fatal("raw tick remained rejected after flush recovery")
		}
	case <-time.After(time.Second):
		t.Fatal("raw tick did not resume after flush recovery")
	}
	if got := h.p.state["RELIANCE"]; got.lastVolume != 115 || !got.lastVolumeAt.Equal(second.Timestamp) || got.open.Volume != 15 {
		t.Fatalf("recovered tick state = %+v, want volume 115 / delta 15", got)
	}
	h.p.errMu.Lock()
	tickErr := h.p.tickStorageErr
	h.p.errMu.Unlock()
	if tickErr != nil {
		t.Fatalf("raw storage error after recovery = %v, want nil", tickErr)
	}
	if err := h.p.Check(context.Background()); !errors.Is(err, candleErr) {
		t.Fatalf("readiness after raw recovery = %v, want unresolved candle error", err)
	}
	if err := h.p.ticks.Flush(context.Background()); err != nil {
		t.Fatalf("flush recovered ticks: %v", err)
	}
	files, err := filepath.Glob(filepath.Join(h.dir, "ticks", "symbol=RELIANCE", "date=2026-08-03", "part-*.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []storage.Tick
	for _, file := range files {
		part, err := parquet.ReadFile[storage.Tick](file)
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, part...)
	}
	if len(rows) != 2 || rows[0].Volume != 100 || rows[1].Volume != 115 {
		t.Fatalf("raw rows = %+v, want original ticks exactly once", rows)
	}
}

func TestAcceptedRawFlushErrorDegradesReadinessUntilRecovery(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	root := filepath.Join(h.dir, "raw-ticks")
	h.p.ticks = storage.NewWriter(root, 1)
	h.p.appendTick = h.p.ticks.AppendWithStatus
	blocker := filepath.Join(root, "symbol=RELIANCE")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blocker, []byte("blocked"), 0o644); err != nil {
		t.Fatal(err)
	}
	tick := tickAt(10, base)
	if !h.p.handle(tick) {
		t.Fatal("accepted raw tick was rejected")
	}
	if err := h.p.Check(context.Background()); err == nil {
		t.Fatalf("readiness after accepted flush error = %v, want flush error", err)
	}
	if err := os.Remove(blocker); err != nil {
		t.Fatal(err)
	}
	if !h.p.handle(tickAt(11, base.Add(time.Second))) {
		t.Fatal("raw tick did not recover")
	}
	if err := h.p.Check(context.Background()); err != nil {
		t.Fatalf("readiness after raw recovery = %v, want nil", err)
	}
}

func TestCheckSurfacesUnlatchedTickWriterFailure(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	root := filepath.Join(h.dir, "unlatched-ticks")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	blocker := filepath.Join(root, "symbol=RELIANCE")
	if err := os.WriteFile(blocker, []byte("blocked"), 0o644); err != nil {
		t.Fatal(err)
	}
	h.p.ticks = storage.NewWriter(root, 1)
	accepted, appendErr := h.p.ticks.AppendWithStatus(storage.NewTick(tickAt(10, base)))
	if !accepted || appendErr == nil {
		t.Fatalf("direct writer append = %v, %v; want accepted with unlatched flush error", accepted, appendErr)
	}
	h.p.errMu.Lock()
	tickStorageErr := h.p.tickStorageErr
	h.p.errMu.Unlock()
	if tickStorageErr != nil {
		t.Fatal("direct writer failure unexpectedly populated the pipeline latch")
	}
	if err := h.p.Check(context.Background()); err == nil {
		t.Fatal("Check ignored an unlatched tick writer error")
	}
	if err := os.Remove(blocker); err != nil {
		t.Fatal(err)
	}
	if err := h.p.ticks.Flush(context.Background()); err != nil {
		t.Fatalf("flush after recovery: %v", err)
	}
	if err := h.p.Check(context.Background()); err != nil {
		t.Fatalf("Check after writer recovery = %v, want nil", err)
	}
}

func TestRawTickCapacityCancellationCountsPendingTick(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	var refusals atomic.Int64
	h.p.appendTick = func(storage.Tick) (bool, error) {
		refusals.Add(1)
		return false, fmt.Errorf("injected capacity refusal: %w", storage.ErrCapacity)
	}
	ctx, cancel := context.WithCancel(context.Background())
	h.p.runCtx = ctx
	second := broker.Tick{Symbol: "RELIANCE", Token: token, Price: 101, Volume: 115, Timestamp: base.Add(time.Second)}
	queue := make(chan broker.Tick, 1)
	queue <- second
	close(queue)
	done := make(chan struct{})
	go func() { h.p.process(queue); close(done) }()
	waitFor(t, "capacity retry", func() bool { return refusals.Load() > 1 })
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("processor did not stop after capacity retry cancellation")
	}
	if h.p.shutdownPendingTicks != 1 {
		t.Fatalf("shutdown pending ticks = %d, want exactly the rejected current tick", h.p.shutdownPendingTicks)
	}
	if got := h.metric("mft_ingestion_ticks_dropped_total"); got != 1 {
		t.Fatalf("dropped ticks = %d, want 1", got)
	}
	if got := h.p.state["RELIANCE"]; got.open.Symbol != "" || got.lastVolume != 0 {
		t.Fatalf("state advanced on cancelled raw tick: %+v", got)
	}
}

// TestCandleAppendErrorIsSurfaced is the same regression for the candle path.
func TestCandleAppendErrorIsSurfaced(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	if err := h.p.candles.Close(); err != nil {
		t.Fatalf("close candle writer: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	h.p.runCtx = ctx
	h.p.fold(tickAt(100, base))
	done := make(chan struct{})
	go func() {
		h.p.fold(tickAt(101, base.Add(time.Minute)))
		close(done)
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("rollover did not stop on cancellation")
	}

	if err := h.p.Check(context.Background()); !errors.Is(err, storage.ErrClosed) {
		t.Fatalf("readiness = %v, want storage.ErrClosed", err)
	}
	if err := h.p.Err(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Err = %v, want stopped rollover context", err)
	}
	if got := h.metric("mft_ingestion_candles_completed_total"); got != 0 {
		t.Fatalf("candles completed = %d, want 0 when the append failed", got)
	}
	if got := h.metric("mft_ingestion_storage_append_errors_total"); got == 0 {
		t.Fatal("append errors = 0, want rejected candle attempts")
	}
	if got := h.p.unpersistedCompletedBars(base.Add(2 * time.Minute)); got != 1 {
		t.Fatalf("unpersisted completed bars = %d, want 1 retained bar", got)
	}
	if h.p.pendingRolloverTick == nil || h.p.pendingRolloverTick.Timestamp != base.Add(time.Minute) {
		t.Fatalf("pending rollover tick = %+v, want timestamp %s", h.p.pendingRolloverTick, base.Add(time.Minute))
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
				Volume:    int64(100 + minute*8 + 2*int(offset/(15*time.Second))),
				Timestamp: base.Add(time.Duration(minute)*time.Minute + offset),
			})
		}
	}
	// A tick in the sixth minute is what closes the fifth bar.
	ticks = append(ticks, tickAt(105, base.Add(time.Duration(minutes)*time.Minute)))

	h.runProcessor(ticks...)

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
		wantVolume := int64(6)
		if i > 0 {
			wantVolume = 8
		}
		if candle.Volume != wantVolume {
			t.Fatalf("candle %d volume = %d, want %d", i, candle.Volume, wantVolume)
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
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 20, Volume: 5,
			Timestamp: base.Add(time.Minute)},
		broker.Tick{Symbol: "RELIANCE", Token: token, Price: 30, Volume: 6,
			Timestamp: base.Add(2 * time.Minute)},
	)
	h.flushCandles()

	got := h.candles("RELIANCE")
	if len(got) != 2 {
		t.Fatalf("persisted %d candles, want 2: %+v", len(got), got)
	}
	first, second := got[0], got[1]
	if !first.Timestamp.Equal(base) || first.Open != 10 || first.Close != 10 || first.Volume != 0 {
		t.Fatalf("first bar = %+v, want {O10 C10 V0} at %s", first, base)
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
	if got[0].Volume != 0 || got[0].Close != 10 {
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
				Volume:    int64(100 + minute*perMin + i),
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
		wantVolume := int64(perMin)
		if i == 0 {
			wantVolume--
		}
		if candle.Volume != wantVolume {
			t.Fatalf("candle %d volume = %d, want %d", i, candle.Volume, wantVolume)
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

// TestBackpressurePreservesTicks is the overload policy test.
func TestBackpressurePreservesTicks(t *testing.T) {
	h := newHarness(t, idleStreamer{})

	const size = 4
	queue := make(chan broker.Tick, size)

	for i := range size {
		h.p.deliver(tickAt(float64(i), base), queue)
	}
	if n := h.metric("mft_ingestion_ticks_dropped_total"); n != 0 {
		t.Fatalf("drops = %d before overload, want 0", n)
	}

	done := make(chan struct{})
	go func() {
		h.p.deliver(tickAt(4, base), queue)
		h.p.deliver(tickAt(5, base), queue)
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("deliver passed a full bounded queue")
	case <-time.After(10 * time.Millisecond):
	}
	<-queue
	<-queue
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("deliver did not resume after queue capacity returned")
	}

	if got := len(queue); got != size {
		t.Fatalf("queue depth = %d, want %d", got, size)
	}
	if n := h.metric("mft_ingestion_ticks_dropped_total"); n != 0 {
		t.Fatalf("drops = %d, want 0", n)
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

// TestBackpressureUsesBoundedCapacity checks blocking and recovery.
func TestBackpressureUsesBoundedCapacity(t *testing.T) {
	h := newHarness(t, idleStreamer{})
	queue := make(chan broker.Tick, 2)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := range 1000 {
			h.p.deliver(tickAt(float64(i), base), queue)
		}
	}()

	time.Sleep(10 * time.Millisecond)
	select {
	case <-done:
		t.Fatal("deliver bypassed bounded queue capacity")
	default:
	}
	for range 1000 {
		<-queue
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("producer did not resume after draining")
	}
	if n := h.metric("mft_ingestion_ticks_dropped_total"); n != 0 {
		t.Fatalf("drops = %d, want 0", n)
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
		Broker:    config.BrokerConfig{Instruments: []string{"RELIANCE"}},
		Storage:   config.StorageConfig{DataDir: t.TempDir(), FlushIntervalSecs: 1, FlushMaxRows: 100},
		Execution: config.ExecutionConfig{PaperTrading: true},
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
