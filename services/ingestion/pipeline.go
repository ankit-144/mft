// Package ingestion implements Service 1 of the MFT platform: the tick
// ingestion pipeline. It is the head of the data flow (Plan.md §3). Every
// candle the inference service ever scores is aggregated here, and every raw
// tick the platform keeps was written here first.
//
// # One writer, no locks
//
// A single goroutine owns the tick path. It folds each tick into the fluxKV
// one-minute candle, appends the raw tick to Parquet, and decides when a bar
// is complete. Nothing else touches the writers, the candle map, or the open
// bars, so the hot path carries no lock at all and candle rollover is
// race-free by construction rather than by discipline.
// TestConcurrentTicksSingleWriter runs that claim under -race.
//
// # Rollover
//
// A candle is complete when a tick for a later minute arrives, or when the
// clock passes the bar's minute for a symbol that has gone quiet — an illiquid
// scrip, or the last minute of a session, produces no further tick and would
// otherwise never be written. Either way the bar is appended once and removed
// from the open set in the same step, so it can never land twice, and a
// per-symbol watermark on the last persisted minute rejects a late tick that
// would otherwise resurrect a closed bar.
//
// # Overload: drop oldest
//
// The broker-facing queues are bounded, and the policy when they fill — which
// means storage is blocking — is to drop the *oldest* queued tick and admit
// the newest. Two reasons:
//
//   - The freshest tick is the one that matters. It sets the candle close the
//     next inference pull will read, and a queued tick cannot become more
//     valuable with age.
//   - Blocking instead stalls the socket reader. A Kite reader that stops
//     consuming stops answering the server's ping control frames, and the
//     broker drops the session — so a storage slowdown would escalate into a
//     reconnect storm, which is strictly worse than losing a few ticks of a
//     bar that is already stale.
//
// Every drop is counted in mft_ingestion_ticks_dropped_total, so a sustained
// non-zero rate is a visible symptom rather than a silent one.
//
// # Supervision
//
// core/broker reconnects internally with backoff, but Stream can still return:
// a bad credential, an instrument missing from the dump, a closed ctx. The
// pipeline supervises it and re-invokes with its own backoff, because a stream
// that quietly returns leaves a process that looks healthy and stores nothing.
package ingestion

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/metrics"
	"github.com/mft/core/storage"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Tunables with no key in the frozen config schema. See docs/contracts.md §8:
// no component may add one, so anything the pipeline needs beyond the declared
// keys lives here.
const (
	// tickBufferSize is the depth of both tick queues: the one the broker
	// writes to and the one the processor reads. A 60-second decision cadence
	// at NSE retail tick rates is tens of ticks a second, so a thousand slots
	// is tens of seconds of slack — long enough to ride out a flush, short
	// enough that a genuinely wedged consumer is caught by the drop counter
	// rather than by the process hanging.
	tickBufferSize = 1024

	// sweepInterval is how often the processor looks for bars whose minute has
	// passed. It bounds how long a bar for a symbol that stopped trading waits
	// before it is persisted.
	sweepInterval = 5 * time.Second

	// ageInterval is the cadence of the last-tick-age gauge. It runs on its
	// own goroutine, deliberately independent of the processor: if the
	// processor wedged, an age updated from inside it would freeze at a
	// healthy-looking value, which is the one thing a liveness signal must
	// never do.
	ageInterval = time.Second

	// defaultReconnectBase is the first supervisor backoff delay. It is short
	// because core/broker has already applied its own backoff before Stream
	// returns; this only paces the re-invocation.
	defaultReconnectBase = time.Second

	// defaultReconnectMax caps the supervisor backoff and matches the
	// broker.reconnect_max_backoff_seconds default.
	defaultReconnectMax = 60 * time.Second
)

// Module is the FX module for the ingestion service.
var Module = fx.Module("ingestion",
	fx.Provide(
		NewCandleStore,
		NewPipeline,
	),
	fx.Invoke(
		AttachBrokerLogger,
		RegisterHealth,
		(*Pipeline).Run,
	),
)

// loggableStreamer is the optional logger hook the broker connector exposes.
type loggableStreamer interface {
	SetLogger(*zap.Logger)
}

// AttachBrokerLogger hands the service logger to the broker connector.
//
// core/broker logs every reconnect and its backoff, and without this those go
// to a no-op logger: the single most useful line during an outage is lost. The
// type assertion keeps this module decoupled from the concrete connector — a
// Streamer without the hook is still perfectly usable, it just does not log
// its own reconnect attempts.
func AttachBrokerLogger(streamer broker.Streamer, log *zap.Logger) {
	if s, ok := streamer.(loggableStreamer); ok {
		s.SetLogger(log)
	}
}

// Clock reports the current time. Candle boundaries depend on wall-clock time,
// so the pipeline reads it through a Clock rather than calling time.Now: a
// rollover test sets the instant under test instead of waiting for the real
// minute boundary.
type Clock func() time.Time

// SystemClock returns the current instant in UTC, the timezone every timestamp
// in the platform is normalised to (docs/contracts.md §1).
func SystemClock() time.Time { return time.Now().UTC() }

// metrics is the Prometheus surface of the pipeline. Every name carries the
// mandatory mft_ prefix; core/metrics validates it at registration, so a typo
// here stops the process at startup rather than shipping a series nobody is
// watching.
type ingestionMetrics struct {
	ticksProcessed prometheus.Counter
	ticksDropped   prometheus.Counter
	ticksLate      prometheus.Counter
	candlesDone    prometheus.Counter
	appendErrors   *prometheus.CounterVec
	lastTickAge    prometheus.Gauge
	queueDepth     prometheus.Gauge
	streamRestarts prometheus.Counter
}

// newMetrics registers the pipeline's collectors on reg.
func newMetrics(reg *prometheus.Registry) *ingestionMetrics {
	return &ingestionMetrics{
		ticksProcessed: metrics.Counter(reg, "mft_ingestion_ticks_processed_total",
			"Broker ticks folded into a candle and appended to the tick store."),
		ticksDropped: metrics.Counter(reg, "mft_ingestion_ticks_dropped_total",
			"Ticks discarded because the processor was behind and a queue was full. The oldest is dropped."),
		ticksLate: metrics.Counter(reg, "mft_ingestion_ticks_late_total",
			"Ticks whose minute was already persisted, discarded rather than reopening a closed candle."),
		candlesDone: metrics.Counter(reg, "mft_ingestion_candles_completed_total",
			"One-minute candles completed and appended to the candle store."),
		appendErrors: metrics.CounterVec(reg, "mft_ingestion_storage_append_errors_total",
			"Rows the Parquet writers refused, by dataset.", "dataset"),
		lastTickAge: metrics.Gauge(reg, "mft_ingestion_last_tick_age_seconds",
			"Age of the most recent broker tick. Growth means the pipeline is wedged."),
		queueDepth: metrics.Gauge(reg, "mft_ingestion_tick_queue_depth",
			"Ticks currently buffered between the broker stream and the processor."),
		streamRestarts: metrics.Counter(reg, "mft_ingestion_stream_restarts_total",
			"Times the broker stream returned and the pipeline re-invoked it."),
	}
}

// Pipeline folds broker ticks into one-minute candles and persists both the
// raw ticks and the completed candles to Parquet.
type Pipeline struct {
	streamer broker.Streamer
	cache    *fluxkv.KV
	ticks    *storage.Writer
	candles  CandleStore
	symbols  []string
	log      *zap.Logger
	opts     options

	// state is the processor's private, single-writer view of one symbol per
	// entry. Only the processor goroutine reads or writes it.
	state map[string]symbolState

	// lastTickNS is the Unix nanosecond timestamp of the most recent processed
	// tick, read by the age goroutine.
	lastTickNS atomic.Int64

	// errMu guards firstErr, the earliest pipeline failure seen on any
	// goroutine. It is returned by Err and by the fx stop hook.
	errMu    sync.Mutex
	firstErr error

	cancel context.CancelFunc
	wg     sync.WaitGroup

	metrics *ingestionMetrics
}

// symbolState is what the processor remembers about one symbol between ticks.
//
// closed is the watermark that makes "exactly once" hold: a tick for a minute
// at or before it is late and is dropped rather than reopening a bar that has
// already been written.
type symbolState struct {
	// open is the bar still accepting ticks, or the zero Candle when the
	// symbol has no bar open.
	open contracts.Candle
	// closed is the minute of the last bar persisted for this symbol, or the
	// zero time when nothing has been persisted yet.
	closed time.Time
}

// options are the seams tests drive. Zero values are not usable; see
// defaultOptions.
type options struct {
	// clock supplies the current time. Defaults to SystemClock.
	clock Clock
	// tickBuffer is the depth of each tick queue. Defaults to tickBufferSize.
	tickBuffer int
	// sweepEvery is the bar-sweep cadence. Defaults to sweepInterval.
	sweepEvery time.Duration
	// ageEvery is the last-tick-age sampling cadence. Defaults to ageInterval.
	ageEvery time.Duration
	// reconnectBase and reconnectMax bound the supervisor backoff.
	reconnectBase time.Duration
	reconnectMax  time.Duration
}

// defaultOptions fills the seams from the frozen config, falling back to the
// package defaults for every value a zero-valued config.Config leaves unset.
func defaultOptions(cfg *config.Config) options {
	opts := options{
		clock:         SystemClock,
		tickBuffer:    tickBufferSize,
		sweepEvery:    sweepInterval,
		ageEvery:      ageInterval,
		reconnectBase: defaultReconnectBase,
		reconnectMax:  defaultReconnectMax,
	}
	if max := time.Duration(cfg.Broker.ReconnectMaxBackoffSecs) * time.Second; max > 0 {
		opts.reconnectMax = max
	}
	return opts
}

// NewPipeline builds the ingestion pipeline from the FX graph.
//
// The subscription list comes from cfg.Broker.Instruments: passing nil would
// leave core/broker to fall back on its own watchlist, which is the same list
// but by coincidence rather than by wiring, and the two can drift.
func NewPipeline(
	streamer broker.Streamer,
	cache *fluxkv.KV,
	ticks *storage.Writer,
	candles CandleStore,
	cfg *config.Config,
	reg *prometheus.Registry,
	log *zap.Logger,
) *Pipeline {
	return newPipeline(streamer, cache, ticks, candles, cfg, reg, log, defaultOptions(cfg))
}

// newPipeline is the injectable constructor. Tests supply the clock and the
// intervals through opts instead of waiting on real time.
func newPipeline(
	streamer broker.Streamer,
	cache *fluxkv.KV,
	ticks *storage.Writer,
	candles CandleStore,
	cfg *config.Config,
	reg *prometheus.Registry,
	log *zap.Logger,
	opts options,
) *Pipeline {
	if log == nil {
		log = zap.NewNop()
	}
	if opts.clock == nil {
		opts.clock = SystemClock
	}
	if opts.tickBuffer <= 0 {
		opts.tickBuffer = tickBufferSize
	}
	if opts.sweepEvery <= 0 {
		opts.sweepEvery = sweepInterval
	}
	if opts.ageEvery <= 0 {
		opts.ageEvery = ageInterval
	}
	if opts.reconnectBase <= 0 {
		opts.reconnectBase = defaultReconnectBase
	}
	if opts.reconnectMax <= 0 {
		opts.reconnectMax = defaultReconnectMax
	}
	return &Pipeline{
		streamer: streamer,
		cache:    cache,
		ticks:    ticks,
		candles:  candles,
		symbols:  append([]string(nil), cfg.Broker.Instruments...),
		log:      log,
		opts:     opts,
		state:    make(map[string]symbolState),
		metrics:  newMetrics(reg),
	}
}

// RegisterHealth adds the pipeline's storage check to the process readiness
// probe set, so /readyz reports "degraded" when Parquet writes are failing
// instead of serving a healthy process that has silently stopped storing.
func RegisterHealth(p *Pipeline) {
	metrics.DefaultChecks().Add(metrics.Checker{
		Name:  "ingestion-storage",
		Check: p.Check,
	})
}

// Check reports the earliest pipeline failure, or nil while ingestion is
// storing everything it is given. It is the /readyz probe.
func (p *Pipeline) Check(context.Context) error { return p.Err() }

// Err returns the earliest error the pipeline recorded: a refused Parquet
// append, or a writer that failed on its background flusher. It is nil until
// something has actually gone wrong.
func (p *Pipeline) Err() error {
	p.errMu.Lock()
	defer p.errMu.Unlock()
	return p.firstErr
}

// recordError keeps the first failure. Later errors are already visible in
// the append-error counter and in the logs, and the first one is the one that
// explains the rest.
func (p *Pipeline) recordError(err error) {
	if err == nil {
		return
	}
	p.errMu.Lock()
	defer p.errMu.Unlock()
	if p.firstErr == nil {
		p.firstErr = err
	}
}

// Run registers the pipeline lifecycle with fx: start the writers and the
// worker goroutines on start, drain and close everything on stop.
func (p *Pipeline) Run(lc fx.Lifecycle) {
	lc.Append(fx.Hook{
		OnStart: p.Start,
		OnStop:  p.Stop,
	})
}

// Start brings the writers up and launches the supervisor, the relay, the
// processor and the age sampler.
//
// The long-lived context is derived with context.WithoutCancel from the fx
// start context: that context carries a start deadline and is cancelled the
// moment OnStart returns, and both the writers' background flusher and the
// broker stream must outlive it. Values are kept; only the deadline is not.
func (p *Pipeline) Start(ctx context.Context) error {
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))

	if err := p.ticks.Start(runCtx); err != nil {
		cancel()
		return fmt.Errorf("ingestion: start tick writer: %w", err)
	}
	if err := p.candles.Start(runCtx); err != nil {
		cancel()
		return fmt.Errorf("ingestion: start candle writer: %w", err)
	}
	// p.cancel is assigned only once both writers are up. A failed Start
	// leaves the previous cancel in place, so a caller that starts twice by
	// mistake still stops the pipeline that is actually running.
	p.cancel = cancel

	// upstream is written by Stream and closed by supervise once Stream has
	// returned, so a send can never race the close. queue is written by relay
	// and closed by relay once upstream is closed and drained.
	upstream := make(chan broker.Tick, p.opts.tickBuffer)
	queue := make(chan broker.Tick, p.opts.tickBuffer)

	p.wg.Add(4)
	go func() {
		defer p.wg.Done()
		p.supervise(runCtx, upstream)
	}()
	go func() {
		defer p.wg.Done()
		p.relay(upstream, queue)
	}()
	go func() {
		defer p.wg.Done()
		p.process(queue)
	}()
	go func() {
		defer p.wg.Done()
		p.sampleAge(runCtx)
	}()

	p.log.Info("ingestion pipeline started",
		zap.Strings("instruments", p.symbols),
		zap.Int("tick_buffer", p.opts.tickBuffer),
	)
	return nil
}

// Stop drains the pipeline and closes the writers. It is the fx OnStop hook.
//
// The order is fixed and each step depends on the previous one:
//
//  1. cancel, so core/broker's cancellable send unblocks and Stream returns;
//  2. wait for the supervisor, which is what closes upstream — closing a
//     channel the streamer might still be sending on would panic;
//  3. the relay drains upstream into queue and closes queue;
//  4. the processor drains queue and returns, having folded every tick;
//  5. sweep once more, so a bar whose minute rolled over in the last few
//     seconds is not lost to the sweep interval;
//  6. flush and close both writers, then surface Err from each.
//
// The wait is bounded by the fx stop deadline. A streamer that ignores
// cancellation cannot be closed safely, so the pipeline reports a timeout
// rather than panicking on a send to a closed channel.
func (p *Pipeline) Stop(ctx context.Context) error {
	p.log.Info("ingestion pipeline stopping")
	if p.cancel != nil {
		p.cancel()
	}

	drained := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(drained)
	}()

	select {
	case <-drained:
	case <-ctx.Done():
		err := fmt.Errorf("ingestion: timed out draining %d tick(s) after shutdown: %w",
			p.ticks.Pending()+p.candles.Pending(), ctx.Err())
		p.recordError(err)
		p.log.Error("ingestion pipeline drain timed out", zap.Error(err))
		return err
	}

	// The wait group has been observed, so the processor has stopped and the
	// state map is no longer owned by another goroutine. Sweeping here closes
	// the window in which a bar that completed just before shutdown would sit
	// unwritten until the next sweep tick.
	p.sweep(p.opts.clock())
	if open := p.openBars(); open > 0 {
		// A bar still inside its minute is not a complete bar. Persisting it
		// would publish a short OHLCV that the feature pipeline cannot tell
		// apart from a full one, so it is dropped and the raw ticks it came
		// from are already durable in the tick store.
		p.log.Info("dropping incomplete candles at shutdown", zap.Int("symbols", open))
	}

	// Only flush while the stop deadline still has time; Close flushes on a
	// detached context regardless, so an expired deadline costs a log line
	// rather than an unflushed buffer.
	if ctx.Err() == nil {
		if err := p.ticks.Flush(ctx); err != nil {
			p.recordError(fmt.Errorf("ingestion: flush ticks: %w", err))
		}
		if err := p.candles.Flush(ctx); err != nil {
			p.recordError(fmt.Errorf("ingestion: flush candles: %w", err))
		}
	}
	if err := p.ticks.Close(); err != nil {
		p.recordError(fmt.Errorf("ingestion: close tick writer: %w", err))
	}
	if err := p.candles.Close(); err != nil {
		p.recordError(fmt.Errorf("ingestion: close candle writer: %w", err))
	}
	// A background flush can fail after the last explicit one, so Err is the
	// only way to see it.
	if err := p.ticks.Err(); err != nil {
		p.recordError(fmt.Errorf("ingestion: tick writer: %w", err))
	}
	if err := p.candles.Err(); err != nil {
		p.recordError(fmt.Errorf("ingestion: candle writer: %w", err))
	}

	err := p.Err()
	p.log.Info("ingestion pipeline stopped", zap.Error(err))
	return err
}

// supervise runs the broker stream in a loop.
//
// core/broker reconnects internally with backoff and normally blocks until its
// context ends. It can still return — a credential rejected at start, an
// instrument missing from the dump, a dial that never succeeds. Left alone,
// that leaves a process that reports healthy and stores nothing, so the
// pipeline logs the return and re-invokes on its own backoff.
func (p *Pipeline) supervise(ctx context.Context, upstream chan broker.Tick) {
	// Closing upstream here is what makes the drain safe: after this point no
	// goroutine can send on it.
	defer close(upstream)

	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			p.metrics.streamRestarts.Inc()
		}
		err := p.streamer.Stream(ctx, p.symbols, upstream)
		if ctx.Err() != nil {
			return
		}
		delay := p.backoffFor(attempt)
		p.log.Error("broker stream returned, restarting",
			zap.Int("attempt", attempt),
			zap.Duration("backoff", delay),
			zap.Error(err),
		)
		if !p.wait(ctx, delay) {
			return
		}
	}
}

// backoffFor doubles reconnectBase per attempt, saturating at reconnectMax.
// The doubling is iterative rather than a shift, so a long-lived stream that
// has been down for hours cannot overflow the duration into a negative wait
// and turn into a hot loop.
func (p *Pipeline) backoffFor(attempt int) time.Duration {
	if attempt < 0 {
		attempt = 0
	}
	delay := p.opts.reconnectBase
	for i := 0; i < attempt; i++ {
		if delay >= p.opts.reconnectMax {
			return p.opts.reconnectMax
		}
		delay *= 2
	}
	if delay > p.opts.reconnectMax || delay <= 0 {
		return p.opts.reconnectMax
	}
	return delay
}

// wait sleeps for d and reports whether it slept to completion rather than
// being cancelled.
func (p *Pipeline) wait(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// relay moves ticks from the broker-facing channel to the processor's queue,
// dropping the oldest entry when the queue is full.
//
// It takes no context on purpose: the shutdown path cancels, waits for the
// supervisor, and only then closes upstream, so a context-aware relay could
// abandon ticks the streamer had already handed over. Running until upstream
// closes makes the drain complete by construction.
func (p *Pipeline) relay(upstream <-chan broker.Tick, queue chan broker.Tick) {
	defer close(queue)
	for tick := range upstream {
		p.deliver(tick, queue)
	}
}

// deliver hands one tick to the processor, evicting the oldest queued tick if
// there is no room. It never blocks, which is the whole point: a stalled
// consumer must not stall the socket reader.
//
// queue is bidirectional because dropping the oldest tick is a receive. The
// relay is its only writer, so an eviction cannot race another producer.
func (p *Pipeline) deliver(tick broker.Tick, queue chan broker.Tick) {
	select {
	case queue <- tick:
		p.metrics.queueDepth.Set(float64(len(queue)))
		return
	default:
	}

	select {
	case <-queue:
		p.metrics.ticksDropped.Inc()
	default:
	}
	select {
	case queue <- tick:
		p.metrics.queueDepth.Set(float64(len(queue)))
	default:
		// The single reader freed and re-filled the slot between the two
		// attempts. The tick is lost, but the counter says so.
		p.metrics.ticksDropped.Inc()
	}
}

// process is the single writer. It folds every tick, appends it to the tick
// store, completes bars, and sweeps bars whose minute has passed.
func (p *Pipeline) process(queue <-chan broker.Tick) {
	sweep := time.NewTicker(p.opts.sweepEvery)
	defer sweep.Stop()

	for {
		select {
		case tick, ok := <-queue:
			if !ok {
				return
			}
			p.handle(tick)
		case <-sweep.C:
			p.sweep(p.opts.clock())
		}
	}
}

// handle folds one tick. It runs only on the processor goroutine.
func (p *Pipeline) handle(tick broker.Tick) {
	if err := p.ticks.Append(storage.NewTick(tick)); err != nil {
		p.metrics.appendErrors.WithLabelValues("ticks").Inc()
		err = fmt.Errorf("ingestion: append tick %s at %s: %w",
			tick.Symbol, tick.Timestamp.UTC().Format(time.RFC3339), err)
		p.recordError(err)
		p.log.Error("tick append failed", zap.String("symbol", tick.Symbol), zap.Error(err))
	}

	p.fold(tick)

	p.metrics.ticksProcessed.Inc()
	p.lastTickNS.Store(tick.Timestamp.UnixNano())
	p.metrics.lastTickAge.Set(p.opts.clock().Sub(tick.Timestamp).Seconds())
}

// fold updates the live candle and persists the bar the tick just closed.
//
// A tick whose minute is at or before the symbol's last persisted minute is
// late — a clock skew, a reordered frame, a reconnect replaying the tail of
// the session. Folding it would rebuild a bar that is already on disk and
// write it twice, so it is counted and dropped. The raw tick is still
// appended, so nothing is lost from the tick store.
func (p *Pipeline) fold(tick broker.Tick) {
	minute := tick.Timestamp.Truncate(time.Minute)
	st := p.state[tick.Symbol]
	if !st.closed.IsZero() && !minute.After(st.closed) {
		p.metrics.ticksLate.Inc()
		p.log.Debug("late tick dropped",
			zap.String("symbol", tick.Symbol),
			zap.Time("tick_minute", minute),
			zap.Time("last_closed", st.closed),
		)
		return
	}

	if st.open.Symbol != "" && !st.open.Timestamp.Equal(minute) {
		p.complete(st.open)
	}

	// complete may have advanced the watermark, so the state is re-read
	// rather than reused.
	st = p.state[tick.Symbol]
	st.open = toContract(p.cache.UpdateCandle(tick.Symbol, tick.Timestamp, tick.Price, tick.Volume))
	p.state[tick.Symbol] = st
}

// sweep persists bars whose minute has passed, for symbols that have stopped
// trading. Without it the last bar of a quiet scrip, or of a session, would
// never be written — there is no later tick to trigger the rollover.
//
// It runs on the processor goroutine, so the same goroutine owns the state a
// rollover-driven complete would touch. It is also called from Stop, which is
// safe because Stop observes the wait group first, so the processor has
// already stopped and no concurrent writer is left.
func (p *Pipeline) sweep(now time.Time) {
	minute := now.UTC().Truncate(time.Minute)
	for _, st := range p.state {
		if st.open.Symbol == "" || !st.open.Timestamp.Before(minute) {
			continue
		}
		p.complete(st.open)
	}
}

// openBars returns how many symbols still have an incomplete bar open. It is
// a shutdown diagnostic.
func (p *Pipeline) openBars() int {
	n := 0
	for _, st := range p.state {
		if st.open.Symbol != "" {
			n++
		}
	}
	return n
}

// complete appends one finished bar to the candle store and retires it: the
// open bar is cleared and the watermark advances to its minute, in the same
// step, on the single goroutine that owns the state. That is what makes
// "written exactly once" hold rather than merely being likely.
func (p *Pipeline) complete(candle contracts.Candle) {
	st := p.state[candle.Symbol]
	st.open = contracts.Candle{}
	st.closed = candle.Timestamp
	p.state[candle.Symbol] = st

	if err := p.candles.Append(storage.NewCandle(candle)); err != nil {
		p.metrics.appendErrors.WithLabelValues("candles").Inc()
		err = fmt.Errorf("ingestion: append candle %s %s: %w",
			candle.Symbol, candle.Timestamp.UTC().Format(time.RFC3339), err)
		p.recordError(err)
		p.log.Error("candle append failed", zap.String("symbol", candle.Symbol), zap.Error(err))
		return
	}
	p.metrics.candlesDone.Inc()
	p.log.Debug("candle completed",
		zap.String("symbol", candle.Symbol),
		zap.Time("minute", candle.Timestamp),
		zap.Float64("close", candle.Close),
	)
}

// sampleAge keeps the last-tick-age gauge current.
//
// It runs on its own goroutine, and that is deliberate: an age gauge updated
// from the processor would freeze at whatever it last managed to record if the
// processor wedged, which is precisely the case the gauge exists to reveal.
func (p *Pipeline) sampleAge(ctx context.Context) {
	ticker := time.NewTicker(p.opts.ageEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			last := p.lastTickNS.Load()
			if last == 0 {
				// No tick has arrived yet. Zero is the honest reading: not
				// stale, just empty.
				p.metrics.lastTickAge.Set(0)
				continue
			}
			p.metrics.lastTickAge.Set(p.opts.clock().Sub(time.Unix(0, last)).Seconds())
		}
	}
}

// toContract converts a fluxKV candle snapshot into the frozen domain type.
// C6 hands back a copy, so the result shares no state with the cache.
func toContract(c *fluxkv.Candle) contracts.Candle {
	if c == nil {
		return contracts.Candle{}
	}
	return contracts.Candle{
		Symbol:    c.Symbol,
		Timestamp: c.Timestamp.UTC(),
		Open:      c.Open,
		High:      c.High,
		Low:       c.Low,
		Close:     c.Close,
		Volume:    c.Volume,
	}
}
