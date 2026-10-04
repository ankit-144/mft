package jobs

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/metrics"
	cron "github.com/netresearch/go-cron"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module is the FX module for the jobs service.
var Module = fx.Module("jobs",
	fx.Provide(
		NewLimiterFromConfig,
		NewParquetStoreFromConfig,
		NewScheduler,
		NewBackfillWorker,

		func(k *broker.Kite) InstrumentResolver { return k },

		func(h *KiteHistory) HistoricalClient { return h },
		func(s *ParquetStore) SegmentStore { return s },
		NewKiteHistory,
	),
	fx.Invoke(StartScheduler),
)

// Defaults for the shared jobs config block.
const (
	defaultRateLimitPerSecond = 3
	defaultLookbackDays       = 730
	defaultInterval           = "minute"

	segmentDays = 60

	maxAttempts = 4
)

// day is a calendar day, the granularity the historical API accepts.
const day = 24 * time.Hour

var indiaTimeZone = time.FixedZone("IST", 5*60*60+30*60)

// backfillMetrics is the Prometheus surface of the job.
type backfillMetrics struct {
	runs        prometheus.Counter
	runsFailed  prometheus.Counter
	candles     prometheus.Counter
	requests    prometheus.Counter
	throttles   prometheus.Counter
	segments    prometheus.Counter
	skipped     prometheus.Counter
	lastSuccess prometheus.Gauge
	duration    prometheus.Histogram
}

// newBackfillMetrics registers the job's collectors on reg.
func newBackfillMetrics(reg *prometheus.Registry) *backfillMetrics {
	return &backfillMetrics{
		runs: metrics.Counter(reg, "mft_jobs_backfill_runs_total",
			"Historical backfill runs started."),
		runsFailed: metrics.Counter(reg, "mft_jobs_backfill_runs_failed_total",
			"Historical backfill runs that ended without landing a single candle."),
		candles: metrics.Counter(reg, "mft_jobs_backfill_candles_total",
			"Candles written to the historical Parquet store."),
		requests: metrics.Counter(reg, "mft_jobs_backfill_requests_total",
			"Requests made to the broker historical API, including retried ones."),
		throttles: metrics.Counter(reg, "mft_jobs_backfill_throttles_total",
			"Times the broker historical API throttled a request with 429."),
		segments: metrics.Counter(reg, "mft_jobs_backfill_segments_total",
			"Historical segments written to the Parquet store."),
		skipped: metrics.Counter(reg, "mft_jobs_backfill_segments_skipped_total",
			"Segments skipped because a previous run already landed them."),
		lastSuccess: metrics.Gauge(reg, "mft_jobs_backfill_last_success_timestamp_seconds",
			"Unix timestamp of the last run that landed at least one candle; zero if never."),
		duration: metrics.Histogram(reg, "mft_jobs_backfill_duration_seconds",
			"Wall-clock duration of a historical backfill run."),
	}
}

// BackfillWorker fetches historical candles from the broker, respecting a token-bucket
// rate limit and applying exponential backoff on throttling.
type BackfillWorker struct {
	log *zap.Logger

	history HistoricalClient
	store   SegmentStore
	limiter *Limiter
	backoff Backoff

	interval   string
	lookback   time.Duration
	segmentLen time.Duration
	watchlist  []string

	sleep func(ctx context.Context, d time.Duration) error
	now   func() time.Time

	metrics *backfillMetrics

	running sync.Mutex
}

// NewBackfillWorker constructs the backfill worker from the shared jobs config block,
// wiring the Kite history client, the Parquet segment store and the rate limiter.
func NewBackfillWorker(
	cfg *config.Config,
	hist HistoricalClient,
	store SegmentStore,
	limiter *Limiter,
	log *zap.Logger,
	reg *prometheus.Registry,
) *BackfillWorker {
	if log == nil {
		log = zap.NewNop()
	}
	rate := cfg.Jobs.RateLimitPerSecond
	if rate <= 0 {
		rate = defaultRateLimitPerSecond
	}
	days := cfg.Jobs.BackfillLookbackDays
	if days <= 0 {
		days = defaultLookbackDays
	}
	dir := cfg.Jobs.HistoricalDir
	if dir == "" {
		dir = cfg.Storage.DataDir + "/historical"
	}
	if store == nil {
		store = NewParquetStore(dir)
	}
	if limiter == nil {
		limiter = NewLimiter(float64(rate))
	}
	return &BackfillWorker{
		log:        log,
		history:    hist,
		store:      store,
		limiter:    limiter,
		backoff:    DefaultBackoff(),
		interval:   defaultInterval,
		lookback:   time.Duration(days) * day,
		segmentLen: segmentDays * day,
		watchlist:  append([]string(nil), cfg.Broker.Instruments...),
		sleep:      sleepCtx,
		now:        time.Now,
		metrics:    newBackfillMetrics(reg),
	}
}

// NewLimiterFromConfig builds the token bucket that paces the historical API, from
// jobs.rate_limit_per_second.
func NewLimiterFromConfig(cfg *config.Config) *Limiter {
	rate := cfg.Jobs.RateLimitPerSecond
	if rate <= 0 {
		rate = defaultRateLimitPerSecond
	}
	return NewLimiter(float64(rate))
}

// RunBackfill fetches every configured segment for every configured instrument and
// lands it in the Parquet store.
func (w *BackfillWorker) RunBackfill(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("backfill: %w", err)
	}

	w.running.Lock()
	defer w.running.Unlock()

	symbols := w.symbols()
	w.metrics.runs.Inc()
	started := w.now()
	w.log.Info("backfill run started",
		zap.Strings("symbols", symbols),
		zap.Int("lookback_days", int(w.lookback/day)),
		zap.Duration("segment", w.segmentLen),
	)

	var (
		candles  int
		failures []error
	)
	for _, symbol := range symbols {
		n, err := w.runSymbol(ctx, symbol)
		candles += n
		if err != nil {
			failures = append(failures, err)
			if ctx.Err() != nil {
				break
			}
		}
	}

	w.metrics.duration.Observe(w.now().Sub(started).Seconds())
	w.metrics.candles.Add(float64(candles))

	if candles == 0 && len(failures) > 0 {
		w.metrics.runsFailed.Inc()
		return fmt.Errorf("backfill: %d of %d symbols failed, 0 candles written: %w: %w",
			len(failures), len(symbols), ErrNoHistory, errors.Join(failures...))
	}
	if candles == 0 {
		w.metrics.lastSuccess.Set(float64(w.now().Unix()))
		w.log.Info("backfill run already complete", zap.Int("symbols", len(symbols)))
		return nil
	}

	w.metrics.lastSuccess.Set(float64(w.now().Unix()))
	w.log.Info("backfill run finished",
		zap.Int("candles", candles),
		zap.Int("symbols", len(symbols)),
		zap.Int("failures", len(failures)),
	)
	if len(failures) > 0 {

		w.log.Warn("backfill run finished with failures", zap.Error(errors.Join(failures...)))
	}
	return nil
}

// runSymbol backfills one symbol and reports how many candles it wrote.
func (w *BackfillWorker) runSymbol(ctx context.Context, symbol string) (int, error) {
	now := w.now().In(indiaTimeZone)
	to := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, indiaTimeZone)
	days := int(w.lookback / day)
	if days <= 0 {
		days = defaultLookbackDays
	}
	from := to.AddDate(0, 0, -days)
	segments := w.segmentsFor(symbol, from, to)

	var (
		written  int
		failures []error
	)
	for _, seg := range segments {
		if err := ctx.Err(); err != nil {
			return written, fmt.Errorf("backfill %s: %w", seg.String(), err)
		}

		present, err := w.store.HasSegment(ctx, seg)
		if err != nil {
			failures = append(failures, fmt.Errorf("check %s: %w", seg.String(), err))
			if isFatal(err) {
				break
			}
			continue
		}
		if present {
			w.metrics.skipped.Inc()
			w.log.Debug("segment already on disk", zap.String("segment", seg.String()))
			continue
		}

		candles, err := w.fetchSegment(ctx, seg)
		if err != nil {
			failures = append(failures, err)
			if isFatal(err) {

				break
			}
			continue
		}

		landed := seg
		landed.Candles = candles
		if err := w.store.WriteSegment(ctx, landed); err != nil {
			failures = append(failures, fmt.Errorf("land %s: %w", seg.String(), err))
			continue
		}

		w.metrics.segments.Inc()
		written += len(candles)
		w.log.Info("segment landed",
			zap.String("segment", seg.String()),
			zap.Int("candles", len(candles)),
		)
	}

	if len(failures) > 0 {
		return written, fmt.Errorf("backfill %s: %w", symbol, errors.Join(failures...))
	}
	return written, nil
}

// fetchSegment fetches one segment, pacing itself on the token bucket and backing off
// exponentially on a 429.
func (w *BackfillWorker) fetchSegment(ctx context.Context, seg Segment) ([]contracts.Candle, error) {
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 && lastErr != nil {
			if err := w.sleep(ctx, w.backoff.Delay(attempt-1)); err != nil {
				return nil, fmt.Errorf("backfill %s: backoff: %w", seg.String(), err)
			}
		}

		if err := w.limiter.Take(ctx); err != nil {
			return nil, fmt.Errorf("backfill %s: rate limit: %w", seg.String(), err)
		}
		w.metrics.requests.Inc()

		candles, err := w.history.HistoricalCandles(ctx, seg.Symbol, seg.From, seg.To, w.interval)
		if err == nil {
			return candles, nil
		}
		lastErr = fmt.Errorf("fetch %s: %w", seg.String(), err)

		if errors.Is(err, broker.ErrRateLimit) {
			w.metrics.throttles.Inc()
			w.log.Warn("broker throttled historical request",
				zap.String("segment", seg.String()),
				zap.Int("attempt", attempt+1),
				zap.Error(err),
			)
			continue
		}
		if isFatal(err) {
			return nil, lastErr
		}
		w.log.Warn("historical request failed",
			zap.String("segment", seg.String()),
			zap.Int("attempt", attempt+1),
			zap.Error(err),
		)
	}
	return nil, fmt.Errorf("backfill %s: %d attempts exhausted: %w", seg.String(), maxAttempts, lastErr)
}

// segments splits [from, to) into fixed-width chunks, independent of symbol.
func (w *BackfillWorker) segments(from, to time.Time) []Segment {
	width := w.segmentLen
	if width <= 0 {
		width = segmentDays * day
	}
	from = from.In(indiaTimeZone)
	to = to.In(indiaTimeZone)
	if from.IsZero() || !from.Before(to) {
		return nil
	}
	widthDays := int(width / day)
	if widthDays <= 0 {
		widthDays = segmentDays
	}
	epoch := time.Date(1970, 1, 1, 0, 0, 0, 0, indiaTimeZone)
	var out []Segment
	for cursor := from; cursor.Before(to); {
		days := int(cursor.Sub(epoch) / day)
		boundaryDay := (days/widthDays + 1) * widthDays
		end := epoch.AddDate(0, 0, boundaryDay)
		if end.After(to) {
			end = to
		}
		out = append(out, Segment{From: cursor, To: end})
		cursor = end
	}
	return out
}

// segmentsFor returns the segments a symbol still needs, with the symbol stamped on
// each one.
func (w *BackfillWorker) segmentsFor(symbol string, from, to time.Time) []Segment {
	chunks := w.segments(from, to)
	out := make([]Segment, 0, len(chunks))
	for _, c := range chunks {
		c.Symbol = symbol
		out = append(out, c)
	}
	return out
}

// symbols returns the deduplicated, upper-cased configured instrument list.
func (w *BackfillWorker) symbols() []string {
	return normaliseSymbols(w.watchlist)
}

// normaliseSymbols trims, upper-cases and deduplicates a symbol list, preserving
// first-seen order.
func normaliseSymbols(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		sym := strings.ToUpper(strings.TrimSpace(s))
		if sym == "" {
			continue
		}
		if _, dup := seen[sym]; dup {
			continue
		}
		seen[sym] = struct{}{}
		out = append(out, sym)
	}
	return out
}

// isFatal reports whether an error should stop the run for a symbol rather than costing
// one attempt.
func isFatal(err error) bool {
	return errors.Is(err, broker.ErrAuth) || errors.Is(err, broker.ErrInstrumentNotFound)
}

// Scheduler wraps the go-cron instance.
type Scheduler struct {
	cron *cron.Cron
	log  *zap.Logger
}

// NewScheduler creates a go-cron scheduler with a recover chain.
func NewScheduler(log *zap.Logger) *Scheduler {
	if log == nil {
		log = zap.NewNop()
	}
	c := cron.New(cron.WithChain(cron.Recover(cron.VerbosePrintfLogger(cronLogWriter{log}))))
	return &Scheduler{cron: c, log: log}
}

// cronLogWriter adapts zap to go-cron's Printf-based logger interface.
type cronLogWriter struct{ log *zap.Logger }

// Printf logs a formatted message at info level.
func (w cronLogWriter) Printf(format string, args ...any) {
	w.log.Info(fmt.Sprintf(format, args...))
}

// StartScheduler registers the weekly backfill job and starts the scheduler.
func StartScheduler(lc fx.Lifecycle, s *Scheduler, w *BackfillWorker, cfg *config.Config) {
	schedule := cfg.Jobs.Schedule
	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			_, err := s.cron.AddFunc(schedule, func() {
				if err := w.RunBackfill(ctx); err != nil {
					s.log.Error("backfill job failed", zap.Error(err))
				}
			}, cron.WithName("weekly-backfill"))
			if err != nil {
				return fmt.Errorf("register backfill job: %w", err)
			}
			s.cron.Start()
			s.log.Info("jobs scheduler started", zap.String("schedule", schedule))
			return nil
		},
		OnStop: func(ctx context.Context) error {
			s.cron.Stop()
			return nil
		},
	})
}
