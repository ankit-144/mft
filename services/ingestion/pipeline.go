// Package ingestion stores broker ticks and rolls them into completed one-minute
// candles.
package ingestion

import (
	"context"
	"errors"
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

// Internal pipeline limits and retry intervals.
const (
	tickBufferSize = 1024

	sweepInterval       = 5 * time.Second
	appendRetryInterval = 100 * time.Millisecond

	ageInterval = time.Second

	defaultReconnectBase = time.Second

	defaultReconnectMax = 60 * time.Second
)

var indiaLocation = func() *time.Location {
	loc, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		return time.FixedZone("IST", 5*60*60+30*60)
	}
	return loc
}()

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
func AttachBrokerLogger(streamer broker.Streamer, log *zap.Logger) {
	if s, ok := streamer.(loggableStreamer); ok {
		s.SetLogger(log)
	}
}

// Clock reports the current time.
type Clock func() time.Time

// SystemClock returns the current instant in UTC, the timezone every timestamp in the
// platform is normalised to (docs/contracts.md §1).
func SystemClock() time.Time { return time.Now().UTC() }

// metrics is the Prometheus surface of the pipeline.
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
			"Ticks left unprocessed when shutdown interrupts storage backpressure."),
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

// Pipeline folds broker ticks into one-minute candles and persists both the raw ticks
// and the completed candles to Parquet.
type Pipeline struct {
	streamer   broker.Streamer
	cache      *fluxkv.KV
	ticks      *storage.Writer
	appendTick func(storage.Tick) (bool, error)
	candles    CandleStore
	symbols    []string
	log        *zap.Logger
	opts       options

	state                map[string]symbolState
	runCtx               context.Context
	pendingRolloverTick  *broker.Tick
	shutdownPendingTicks int

	lastTickNS atomic.Int64

	errMu            sync.Mutex
	firstErr         error
	storageErr       error
	tickStorageErr   error
	tickBackpressure bool

	cancel context.CancelFunc
	wg     sync.WaitGroup

	metrics *ingestionMetrics
}

// symbolState is what the processor remembers about one symbol between ticks.
type symbolState struct {
	open contracts.Candle

	closed       time.Time
	volumeDate   time.Time
	lastVolume   int64
	lastVolumeAt time.Time
	lastPrice    float64
}

// options are the seams tests drive.
type options struct {
	clock Clock

	tickBuffer int

	sweepEvery       time.Duration
	appendRetryEvery time.Duration

	ageEvery time.Duration

	reconnectBase time.Duration
	reconnectMax  time.Duration
}

// defaultOptions fills the seams from the shared config, falling back to the package
// defaults for every value a zero-valued config.Config leaves unset.
func defaultOptions(cfg *config.Config) options {
	opts := options{
		clock:            SystemClock,
		tickBuffer:       tickBufferSize,
		sweepEvery:       sweepInterval,
		appendRetryEvery: appendRetryInterval,
		ageEvery:         ageInterval,
		reconnectBase:    defaultReconnectBase,
		reconnectMax:     defaultReconnectMax,
	}
	if max := time.Duration(cfg.Broker.ReconnectMaxBackoffSecs) * time.Second; max > 0 {
		opts.reconnectMax = max
	}
	return opts
}

// NewPipeline builds the ingestion pipeline from the FX graph.
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

// newPipeline is the injectable constructor.
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
		streamer:   streamer,
		cache:      cache,
		ticks:      ticks,
		appendTick: ticks.AppendWithStatus,
		candles:    candles,
		symbols:    append([]string(nil), cfg.Broker.Instruments...),
		log:        log,
		opts:       opts,
		state:      make(map[string]symbolState),
		metrics:    newMetrics(reg),
	}
}

// RegisterHealth adds the pipeline's storage check to the process readiness probe set,
// so /readyz reports "degraded" when Parquet writes are failing instead of serving a
// healthy process that has silently stopped storing.
func RegisterHealth(p *Pipeline) {
	metrics.DefaultChecks().Add(metrics.Checker{
		Name:  "ingestion-storage",
		Check: p.Check,
	})
}

// Check reports latched storage failures, writer failures, or the earliest pipeline
// error while ingestion is storing everything it is given.
func (p *Pipeline) Check(context.Context) error {
	p.errMu.Lock()
	defer p.errMu.Unlock()
	if p.storageErr != nil {
		if !errors.Is(p.storageErr, storage.ErrClosed) && p.candles.Err() == nil && p.candles.Pending() == 0 {
			p.storageErr = nil
		} else {
			return p.storageErr
		}
	}
	if p.tickStorageErr != nil {
		if !p.tickBackpressure && !errors.Is(p.tickStorageErr, storage.ErrClosed) && p.ticks.Err() == nil {
			p.tickStorageErr = nil
		} else {
			return p.tickStorageErr
		}
	}
	if err := p.candles.Err(); err != nil {
		return fmt.Errorf("ingestion: candle writer: %w", err)
	}
	if err := p.ticks.Err(); err != nil {
		return fmt.Errorf("ingestion: tick writer: %w", err)
	}
	return p.firstErr
}

// Err returns the earliest permanent pipeline error; Check also reports recoverable
// storage failures.
func (p *Pipeline) Err() error {
	p.errMu.Lock()
	defer p.errMu.Unlock()
	return p.firstErr
}

// recordError keeps the first failure.
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

func (p *Pipeline) setStorageError(err error) {
	p.errMu.Lock()
	p.storageErr = err
	p.errMu.Unlock()
}

func (p *Pipeline) clearStorageError() {
	p.errMu.Lock()
	p.storageErr = nil
	p.errMu.Unlock()
}

func (p *Pipeline) setTickStorageError(err error) {
	p.errMu.Lock()
	p.tickStorageErr = err
	p.tickBackpressure = false
	p.errMu.Unlock()
}

func (p *Pipeline) setTickBackpressureError(err error) {
	p.errMu.Lock()
	p.tickStorageErr = err
	p.tickBackpressure = true
	p.errMu.Unlock()
}

func (p *Pipeline) clearTickStorageError() {
	p.errMu.Lock()
	p.tickStorageErr = nil
	p.tickBackpressure = false
	p.errMu.Unlock()
}

// Run registers the pipeline lifecycle with fx: start the writers and the worker
// goroutines on start, drain and close everything on stop.
func (p *Pipeline) Run(lc fx.Lifecycle) {
	lc.Append(fx.Hook{
		OnStart: p.Start,
		OnStop:  p.Stop,
	})
}

// Start brings the writers up and launches the supervisor, the relay, the processor and
// the age sampler.
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

	p.cancel = cancel
	p.runCtx = runCtx

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

// Stop cancels the stream, drains or accounts for queued ticks, and closes writers.
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

	p.sweep(p.opts.clock())
	if n := p.unpersistedCompletedBars(p.opts.clock()); n > 0 {
		p.recordError(fmt.Errorf("ingestion: %d completed candle(s) remain unpersisted at shutdown", n))
	}
	if open := p.openBars(); open > 0 {

		p.log.Info("dropping incomplete candles at shutdown", zap.Int("symbols", open))
	}

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
func (p *Pipeline) supervise(ctx context.Context, upstream chan broker.Tick) {

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

// wait sleeps for d and reports whether it slept to completion rather than being
// cancelled.
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

// relay moves broker ticks into the bounded processor queue.
func (p *Pipeline) relay(upstream <-chan broker.Tick, queue chan broker.Tick) {
	defer close(queue)
	for tick := range upstream {
		p.deliver(tick, queue)
	}
}

// deliver applies backpressure when the processor queue is full.
func (p *Pipeline) deliver(tick broker.Tick, queue chan broker.Tick) {
	queue <- tick
	p.metrics.queueDepth.Set(float64(len(queue)))
}

// process is the single writer.
func (p *Pipeline) process(queue <-chan broker.Tick) {
	sweep := time.NewTicker(p.opts.sweepEvery)
	defer sweep.Stop()

	for {
		select {
		case tick, ok := <-queue:
			if !ok {
				return
			}
			if !p.handle(tick) {
				if p.cancel != nil {
					p.cancel()
				}
				pending := 1
				for range queue {
					pending++
				}
				p.shutdownPendingTicks += pending
				p.metrics.ticksDropped.Add(float64(pending))
				return
			}
			if p.runCtx != nil && p.runCtx.Err() != nil {
				pending := 0
				if p.pendingRolloverTick != nil {
					pending++
				}
				for range queue {
					pending++
				}
				p.shutdownPendingTicks += pending
				if pending > 0 {
					p.metrics.ticksDropped.Add(float64(pending))
					p.recordError(fmt.Errorf("ingestion: shutdown left %d tick(s) unprocessed after candle storage failure", pending))
				}
				return
			}
		case <-sweep.C:
			p.sweep(p.opts.clock())
		}
	}
}

// handle folds one tick.
func (p *Pipeline) handle(tick broker.Tick) bool {
	row := storage.NewTick(tick)
	ctx := p.runCtx
	if ctx == nil {
		ctx = context.Background()
	}
	interval := p.opts.appendRetryEvery
	if interval <= 0 {
		interval = appendRetryInterval
	}
	var ticker *time.Ticker
	var retry <-chan time.Time
	defer func() {
		if ticker != nil {
			ticker.Stop()
		}
	}()
	for {
		accepted, err := p.appendTick(row)
		if accepted {
			if err != nil {
				p.metrics.appendErrors.WithLabelValues("ticks").Inc()
				err = fmt.Errorf("ingestion: append tick %s at %s: %w",
					tick.Symbol, tick.Timestamp.UTC().Format(time.RFC3339), err)
				p.setTickStorageError(err)
				p.log.Error("tick append failed", zap.String("symbol", tick.Symbol), zap.Error(err))
			} else if p.ticks.Err() == nil {
				p.clearTickStorageError()
			}
			break
		}
		if err != nil && errors.Is(err, storage.ErrCapacity) {
			p.setTickBackpressureError(fmt.Errorf("ingestion: raw tick buffer full: %w", err))
			if ticker == nil {
				ticker = time.NewTicker(interval)
				retry = ticker.C
			}
			select {
			case <-ctx.Done():
				p.metrics.appendErrors.WithLabelValues("ticks").Inc()
				p.recordError(fmt.Errorf("ingestion: raw tick %s at %s remained pending during storage backpressure: %w",
					tick.Symbol, tick.Timestamp.UTC().Format(time.RFC3339), ctx.Err()))
				return false
			case <-retry:
				continue
			}
		}
		if err == nil {
			err = errors.New("writer rejected tick without an error")
		}
		p.metrics.appendErrors.WithLabelValues("ticks").Inc()
		err = fmt.Errorf("ingestion: reject tick %s at %s: %w",
			tick.Symbol, tick.Timestamp.UTC().Format(time.RFC3339), err)
		p.setTickStorageError(err)
		p.recordError(err)
		p.log.Error("tick append failed", zap.String("symbol", tick.Symbol), zap.Error(err))
		return false
	}

	p.fold(tick)

	p.metrics.ticksProcessed.Inc()
	p.lastTickNS.Store(tick.Timestamp.UnixNano())
	p.metrics.lastTickAge.Set(p.opts.clock().Sub(tick.Timestamp).Seconds())
	return true
}

// fold rejects stale ticks, updates the open candle, and closes prior minutes.
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
	if st.open.Symbol != "" {
		openMinute := st.open.Timestamp
		if minute.Before(openMinute) || tick.Timestamp.Before(st.lastVolumeAt) {
			p.metrics.ticksLate.Inc()
			return
		}
		if tick.Timestamp.Equal(st.lastVolumeAt) && tick.Price == st.lastPrice && tick.Volume == st.lastVolume {
			return
		}
		if tick.Timestamp.Equal(st.lastVolumeAt) && tick.Volume < st.lastVolume {
			p.metrics.ticksLate.Inc()
			return
		}
	}

	if st.open.Symbol != "" && !st.open.Timestamp.Equal(minute) {
		if !p.completeUntilAccepted(st.open, tick) {
			return
		}
	}

	st = p.state[tick.Symbol]
	delta := p.volumeDelta(&st, tick)
	st.open = toContract(p.cache.UpdateCandle(tick.Symbol, tick.Timestamp, tick.Price, delta))
	st.lastPrice = tick.Price
	p.state[tick.Symbol] = st
}

func (p *Pipeline) completeUntilAccepted(candle contracts.Candle, rollover broker.Tick) bool {
	ctx := p.runCtx
	if ctx == nil {
		ctx = context.Background()
	}
	interval := p.opts.appendRetryEvery
	if interval <= 0 {
		interval = appendRetryInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		if p.complete(candle) {
			p.pendingRolloverTick = nil
			return true
		}
		p.pendingRolloverTick = &rollover
		select {
		case <-ctx.Done():
			p.recordError(fmt.Errorf("ingestion: rollover stopped with %s tick at %s pending behind candle %s %s: %w",
				rollover.Symbol, rollover.Timestamp.UTC().Format(time.RFC3339Nano), candle.Symbol,
				candle.Timestamp.UTC().Format(time.RFC3339), ctx.Err()))
			return false
		case <-ticker.C:
		}
	}
}

func (p *Pipeline) volumeDelta(st *symbolState, tick broker.Tick) int64 {
	if !st.lastVolumeAt.IsZero() && tick.Timestamp.Before(st.lastVolumeAt) {
		return 0
	}
	if tick.Timestamp.Equal(st.lastVolumeAt) && tick.Volume < st.lastVolume {
		return 0
	}
	local := tick.Timestamp.In(indiaLocation)
	day := time.Date(local.Year(), local.Month(), local.Day(), 0, 0, 0, 0, indiaLocation)
	if !st.volumeDate.Equal(day) {
		st.volumeDate = day
		st.lastVolume = tick.Volume
		st.lastVolumeAt = tick.Timestamp
		if local.Hour() == 9 && local.Minute() == 15 {
			return max(tick.Volume, 0)
		}
		return 0
	}
	delta := tick.Volume - st.lastVolume
	if delta < 0 {
		st.lastVolume = tick.Volume
		st.lastVolumeAt = tick.Timestamp
		return 0
	}
	st.lastVolume = tick.Volume
	st.lastVolumeAt = tick.Timestamp
	return delta
}

// sweep persists bars whose minute has passed, for symbols that have stopped trading.
func (p *Pipeline) sweep(now time.Time) {
	minute := now.UTC().Truncate(time.Minute)
	for _, st := range p.state {
		if st.open.Symbol == "" || !st.open.Timestamp.Before(minute) {
			continue
		}
		p.complete(st.open)
	}
}

// openBars returns how many symbols still have an incomplete bar open.
func (p *Pipeline) openBars() int {
	n := 0
	for _, st := range p.state {
		if st.open.Symbol != "" {
			n++
		}
	}
	return n
}

func (p *Pipeline) unpersistedCompletedBars(now time.Time) int {
	minute := now.UTC().Truncate(time.Minute)
	n := 0
	for _, st := range p.state {
		if st.open.Symbol != "" && st.open.Timestamp.Before(minute) {
			n++
		}
	}
	return n
}

// complete appends a finished bar and advances its watermark after acceptance.
func (p *Pipeline) complete(candle contracts.Candle) bool {
	row := storage.NewCandle(candle)
	accepted, err := p.candles.AppendWithStatus(row)
	if !accepted {
		if err != nil {
			p.metrics.appendErrors.WithLabelValues("candles").Inc()
			p.setStorageError(fmt.Errorf("ingestion: reject completed candle %s %s: %w", candle.Symbol, candle.Timestamp, err))
		}
		return false
	}
	st := p.state[candle.Symbol]
	st.open = contracts.Candle{}
	st.closed = candle.Timestamp
	p.state[candle.Symbol] = st
	if err != nil {
		p.metrics.appendErrors.WithLabelValues("candles").Inc()
		p.setStorageError(fmt.Errorf("ingestion: append completed candle %s %s: %w", candle.Symbol, candle.Timestamp, err))
		p.log.Warn("candle retained after append flush failure", zap.String("symbol", candle.Symbol), zap.Error(err))
	}
	flushCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	flushErr := p.candles.Flush(flushCtx)
	cancel()
	if flushErr != nil {
		p.metrics.appendErrors.WithLabelValues("candles").Inc()
		p.setStorageError(fmt.Errorf("ingestion: flush completed candle %s %s: %w", candle.Symbol, candle.Timestamp, flushErr))
		p.log.Error("completed candle flush failed", zap.String("symbol", candle.Symbol), zap.Error(flushErr))
	} else {
		p.clearStorageError()
	}
	p.metrics.candlesDone.Inc()
	p.log.Debug("candle completed",
		zap.String("symbol", candle.Symbol),
		zap.Time("minute", candle.Timestamp),
		zap.Float64("close", candle.Close),
	)
	return true
}

// sampleAge keeps the last-tick-age gauge current.
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

				p.metrics.lastTickAge.Set(0)
				continue
			}
			p.metrics.lastTickAge.Set(p.opts.clock().Sub(time.Unix(0, last)).Seconds())
		}
	}
}

// toContract converts a fluxKV candle snapshot into the shared domain type.
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
