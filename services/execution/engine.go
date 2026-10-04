// Package execution implements the execution service's risk gate and order lifecycle.
package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/metrics"
	"github.com/mft/services/execution/risk"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// Module wires the execution engine and its HTTP/reconciliation lifecycles.
var Module = fx.Module("execution",
	fx.Provide(NewEngine),
	fx.Invoke(StartExecutionEngine, StartHTTPServer, RegisterEngine, StartCacheSweeper),
)

const SweepInterval = time.Minute

// Engine owns durable order state and serializes only reservation/accounting
// transitions.
type Engine struct {
	client broker.OrderClient
	orders broker.OrderReconciler
	quotes broker.QuoteClient
	cache  *fluxkv.KV
	cfg    *config.ExecutionConfig
	log    *zap.Logger

	checks          contracts.Checks
	book            *risk.Book
	clock           risk.Clock
	store           *journal
	state           journalState
	mu              sync.Mutex
	submitWG        sync.WaitGroup
	closeOnce       sync.Once
	closeErr        error
	ready           bool
	readyErr        error
	closed          bool
	exitPolicy      ExitPolicy
	reconcileCancel context.CancelFunc
	reconcileDone   chan struct{}

	debounceTTL time.Duration

	ordersPlaced prometheus.Counter
	rejected     prometheus.Counter
	rejections   *prometheus.CounterVec
}

// NewEngine opens durable execution state.
func NewEngine(client broker.OrderClient, cache *fluxkv.KV, cfg *config.Config, reg *prometheus.Registry, logger *zap.Logger) (*Engine, error) {
	if cfg == nil {
		return nil, fmt.Errorf("execution: config must be non-nil")
	}
	return buildEngine(client, cache, cfg, reg, logger, risk.FromConfig(cfg.Execution), risk.SystemClock)
}

// NewEngineWithClock builds the production engine with a caller supplied clock for
// deterministic integrations.
func NewEngineWithClock(client broker.OrderClient, cache *fluxkv.KV, cfg *config.Config, reg *prometheus.Registry, logger *zap.Logger, clock risk.Clock) (*Engine, error) {
	if cfg == nil {
		return nil, fmt.Errorf("execution: config must be non-nil")
	}
	if clock == nil {
		clock = risk.SystemClock
	}
	policy := risk.FromConfig(cfg.Execution)
	policy.Clock = clock
	return buildEngine(client, cache, cfg, reg, logger, policy, clock)
}

func newEngine(client broker.OrderClient, cache *fluxkv.KV, cfg *config.Config, reg *prometheus.Registry, logger *zap.Logger, policy risk.Policy, clock risk.Clock) *Engine {
	e, err := buildEngine(client, cache, cfg, reg, logger, policy, clock)
	if err != nil {
		panic(err)
	}

	e.ready = true
	return e
}

func buildEngine(client broker.OrderClient, cache *fluxkv.KV, cfg *config.Config, reg *prometheus.Registry, logger *zap.Logger, policy risk.Policy, clock risk.Clock) (*Engine, error) {
	if client == nil || cache == nil || cfg == nil || reg == nil || logger == nil {
		return nil, fmt.Errorf("execution: engine dependencies must be non-nil")
	}
	store, state, err := openJournal(cfg.Execution.JournalPath)
	if err != nil {
		return nil, err
	}
	var book *risk.Book
	if state.Initialized {
		book, err = risk.RestoreBook(state.Book)
		if err != nil {
			_ = store.Close()
			return nil, err
		}
	} else {
		book = risk.NewBook(cfg.Execution.Capital)
		state.Initialized = true
		state.Book = book.State()
		if err := store.Save(state); err != nil {
			_ = store.Close()
			return nil, err
		}
	}
	if state.Orders == nil {
		state.Orders = make(map[string]OrderRecord)
	}
	for key, rec := range state.Orders {
		if rec.Status == contracts.OrderStatusSubmitting {
			rec.Status = contracts.OrderStatusUnknown
			rec.LastError = "process stopped during broker submission"
			rec.UpdatedAt = clock().UTC()
			state.Orders[key] = rec
		}
	}
	factory := promauto.With(reg)
	e := &Engine{
		client: client, cache: cache, cfg: &cfg.Execution, log: logger,
		checks: policy.New(cache), book: book, clock: clock, store: store, state: state,
		debounceTTL:  policy.DebounceTTL,
		ordersPlaced: factory.NewCounter(prometheus.CounterOpts{Name: "mft_execution_orders_placed_total", Help: "Broker orders submitted by the execution engine."}),
		rejected:     factory.NewCounter(prometheus.CounterOpts{Name: "mft_execution_orders_rejected_total", Help: "Order requests rejected before broker submission."}),
		rejections:   factory.NewCounterVec(prometheus.CounterOpts{Name: "mft_execution_rejections_total", Help: "Signals rejected by risk reason code."}, []string{"reason"}),
	}
	e.orders, _ = client.(broker.OrderReconciler)
	e.quotes, _ = client.(broker.QuoteClient)
	restoreDebounceCache(e, clock().UTC())
	e.ready = cfg.Execution.PaperTrading && !hasLivePending(state.Orders)
	if !cfg.Execution.PaperTrading && cfg.Execution.APIToken == "" {
		_ = store.Close()
		return nil, fmt.Errorf("execution: live mode requires execution.api_token")
	}
	return e, nil
}

// ExecuteSignal validates and submits an inference signal under its durable key.
func (e *Engine) ExecuteSignal(ctx context.Context, sig contracts.Signal) (string, error) {
	sig.Symbol = strings.ToUpper(strings.TrimSpace(sig.Symbol))
	sig.Side = strings.ToUpper(strings.TrimSpace(sig.Side))
	sig.IdempotencyKey = strings.TrimSpace(sig.IdempotencyKey)
	if err := risk.ValidateSignal(sig); err != nil {
		e.rejected.Inc()
		return "", err
	}
	req := contracts.OrderRequest{Symbol: sig.Symbol, Side: sig.Side, Quantity: sig.Quantity, Price: sig.Price, Type: contracts.OrderTypeLimit, IdempotencyKey: sig.IdempotencyKey}
	return e.submit(ctx, req, sig.AsOf, sig.Score, sig.Price)
}

// ExecuteOrder submits a direct order using the request's idempotency key.
func (e *Engine) ExecuteOrder(ctx context.Context, req contracts.OrderRequest) (string, error) {
	req.Symbol = strings.ToUpper(strings.TrimSpace(req.Symbol))
	req.Side = strings.ToUpper(strings.TrimSpace(req.Side))
	req.Type = strings.ToUpper(strings.TrimSpace(req.Type))
	req.IdempotencyKey = strings.TrimSpace(req.IdempotencyKey)
	if req.Type == "" {
		if req.Price > 0 {
			req.Type = contracts.OrderTypeLimit
		} else {
			req.Type = contracts.OrderTypeMarket
		}
	}
	if req.IdempotencyKey == "" {
		return "", fmt.Errorf("execution: idempotency_key is required")
	}
	if req.Symbol == "" || (req.Side != contracts.SideBuy && req.Side != contracts.SideSell) || req.Quantity < 1 || (req.Type != contracts.OrderTypeLimit && req.Type != contracts.OrderTypeMarket) || math.IsNaN(req.Price) || math.IsInf(req.Price, 0) || req.Price < 0 {
		return "", fmt.Errorf("execution: invalid order fields")
	}
	if req.Type == contracts.OrderTypeLimit && req.Price <= 0 {
		return "", fmt.Errorf("execution: limit order requires positive price")
	}
	price := req.Price
	if req.Type == contracts.OrderTypeMarket {
		price = 0
	}
	if req.Type == contracts.OrderTypeLimit && (price <= 0 || math.IsNaN(price) || math.IsInf(price, 0)) {
		return "", fmt.Errorf("execution: order risk price must be positive")
	}
	sig := contracts.Signal{Symbol: req.Symbol, Side: req.Side, Quantity: req.Quantity, Price: price, AsOf: e.clock().UTC(), IdempotencyKey: req.IdempotencyKey}
	return e.submit(ctx, req, sig.AsOf, 0, price)
}

// Execute preserves the old internal surface while assigning a one-shot direct-order
// key.
func (e *Engine) Execute(ctx context.Context, symbol, side string, quantity int, price float64) (string, error) {
	key := fmt.Sprintf("ORDER:%s:%s:%d:%.4f:%d", symbol, side, quantity, price, e.clock().UnixNano())
	return e.ExecuteOrder(ctx, contracts.OrderRequest{Symbol: symbol, Side: side, Quantity: quantity, Price: price, Type: contracts.OrderTypeLimit, IdempotencyKey: key})
}

func (e *Engine) submit(ctx context.Context, req contracts.OrderRequest, asOf time.Time, score, riskPrice float64) (string, error) {
	key := strings.TrimSpace(req.IdempotencyKey)
	if key == "" {
		return "", fmt.Errorf("execution: idempotency key is required")
	}
	req.IdempotencyKey = key
	tag := orderTag(key)
	e.mu.Lock()
	if prior, ok := e.state.Orders[key]; ok {
		e.mu.Unlock()
		return e.duplicate(key, prior)
	}
	if e.closed {
		e.mu.Unlock()
		return "", fmt.Errorf("execution: engine is closed")
	}
	if !e.ready {
		err := e.unreadyErrorLocked()
		e.mu.Unlock()
		return "", err
	}
	e.mu.Unlock()

	if e.cfg.MaxSignalAgeSeconds > 0 && !asOf.IsZero() {
		age := e.clock().Sub(asOf)
		if age < -time.Minute || age > time.Duration(e.cfg.MaxSignalAgeSeconds)*time.Second {
			return "", e.reject(contracts.ReasonStaleSignal, "signal timestamp is outside the allowed age")
		}
	}
	if req.Type == contracts.OrderTypeMarket {
		if e.cfg.PaperTrading {
			return "", fmt.Errorf("execution: paper MARKET orders require an injected reference-price source; use a priced LIMIT order")
		}
		marks, err := e.currentPrices(ctx, []string{req.Symbol})
		if err != nil {
			return "", fmt.Errorf("execution: market order risk price: %w", err)
		}
		riskPrice = marks[req.Symbol]
	}
	if riskPrice <= 0 || math.IsNaN(riskPrice) || math.IsInf(riskPrice, 0) {
		return "", fmt.Errorf("execution: order risk price must be positive")
	}
	if !e.cfg.PaperTrading {
		if err := e.refreshMarks(ctx); err != nil {
			return "", e.setUnready(fmt.Errorf("execution: current marks unavailable: %w", err))
		}
	}
	now := e.clock().UTC()
	e.mu.Lock()
	if prior, ok := e.state.Orders[key]; ok {
		e.mu.Unlock()
		return e.duplicate(key, prior)
	}
	if e.closed {
		e.mu.Unlock()
		return "", fmt.Errorf("execution: engine is closed")
	}
	if !e.ready {
		err := e.unreadyErrorLocked()
		e.mu.Unlock()
		return "", err
	}
	if e.cfg.PaperTrading {
		if err := e.book.Mark(req.Symbol, riskPrice, now); err != nil {
			e.mu.Unlock()
			return "", err
		}
	}
	portfolio := e.portfolioLocked(now)
	if countPending(e.state.Orders) >= e.cfg.MaxPendingOrders && e.cfg.MaxPendingOrders > 0 {
		e.mu.Unlock()
		return "", e.reject(contracts.ReasonMaxPositions, "pending order limit reached")
	}
	sig := contracts.Signal{Symbol: req.Symbol, Side: req.Side, Quantity: req.Quantity, Price: riskPrice, Score: score, AsOf: asOf, IdempotencyKey: key}
	if err := e.checkOrder(ctx, sig, portfolio); err != nil {
		e.rejected.Inc()
		var rej *contracts.Rejection
		if errors.As(err, &rej) {
			e.rejections.WithLabelValues(rej.Code).Inc()
		}
		e.mu.Unlock()
		return "", err
	}
	rec := OrderRecord{Key: key, Tag: tag, Request: req, AsOf: asOf.UTC(), Status: contracts.OrderStatusSubmitting, CreatedAt: now, UpdatedAt: now, RiskPrice: riskPrice}
	e.state.Orders[key] = rec
	if err := e.saveLocked(); err != nil {
		e.ready = false
		e.readyErr = fmt.Errorf("persist order reservation: %w", err)
		e.mu.Unlock()
		return "", fmt.Errorf("persist order reservation: %w", err)
	}
	e.cache.Set(risk.DebounceKey(req.Symbol, req.Side), tag, e.debounceTTL)
	if e.cfg.PaperTrading {
		rec.Status = contracts.OrderStatusPaperFilled
		rec.OrderID = "paper-" + tag
		rec.FilledQuantity = req.Quantity
		rec.AverageFillPrice = riskPrice
		rec.AppliedNotional = float64(req.Quantity) * riskPrice
		if err := e.book.ApplyFill(sig, req.Quantity, riskPrice, now); err != nil {
			e.ready = false
			e.readyErr = err
			rec.Status = contracts.OrderStatusRejected
			rec.FilledQuantity = 0
			rec.AppliedNotional = 0
			rec.AverageFillPrice = 0
			rec.LastError = err.Error()
			rec.UpdatedAt = now
			e.state.Orders[key] = rec
			_ = e.saveLocked()
			e.mu.Unlock()
			return rec.OrderID, fmt.Errorf("paper fill accounting: %w", err)
		}
		rec.UpdatedAt = now
		e.state.Orders[key] = rec
		if err := e.saveLocked(); err != nil {
			e.ready = false
			e.readyErr = err
			e.mu.Unlock()
			return rec.OrderID, fmt.Errorf("persist paper fill: %w", err)
		}
		e.mu.Unlock()
		return rec.OrderID, nil
	}
	e.submitWG.Add(1)
	e.mu.Unlock()

	defer e.submitWG.Done()
	orderID, err := e.place(ctx, req, tag)
	if err == nil && strings.TrimSpace(orderID) == "" {
		err = fmt.Errorf("broker returned an empty order id")
	}
	if err != nil {
		if isDefinitiveBrokerError(err) {
			e.cache.Delete(risk.DebounceKey(req.Symbol, req.Side))
			e.mu.Lock()
			rec.Status = contracts.OrderStatusRejected
			rec.LastError = err.Error()
			rec.UpdatedAt = e.clock().UTC()
			e.state.Orders[key] = rec
			saveErr := e.saveLocked()
			if saveErr != nil {
				e.ready = false
				e.readyErr = saveErr
			}
			e.mu.Unlock()
			if saveErr != nil {
				return "", fmt.Errorf("broker rejected order and journal update failed: %w", saveErr)
			}
			e.rejected.Inc()
			return "", fmt.Errorf("place order: %w", err)
		}
		e.mu.Lock()
		if strings.TrimSpace(orderID) != "" {
			rec.OrderID = strings.TrimSpace(orderID)
		}
		rec.Status = contracts.OrderStatusUnknown
		rec.LastError = err.Error()
		rec.UpdatedAt = e.clock().UTC()
		e.state.Orders[key] = rec
		e.ready = false
		e.readyErr = fmt.Errorf("unresolved broker submission %s", key)
		saveErr := e.saveLocked()
		e.mu.Unlock()
		if saveErr != nil {
			return "", fmt.Errorf("submission outcome unknown and journal update failed: %w", saveErr)
		}
		return rec.OrderID, &IndeterminateError{Key: key, Cause: err}
	}
	e.mu.Lock()
	rec.OrderID = orderID
	rec.Status = contracts.OrderStatusOpen
	rec.UpdatedAt = e.clock().UTC()
	e.state.Orders[key] = rec
	if err := e.saveLocked(); err != nil {
		e.ready = false
		e.readyErr = err
		e.mu.Unlock()
		return orderID, fmt.Errorf("persist broker order acknowledgement: %w", err)
	}
	e.ordersPlaced.Inc()
	e.mu.Unlock()

	if e.orders != nil {
		if order, err := e.orders.GetOrder(ctx, orderID); err == nil {
			if err := e.applyBrokerOrder(key, order); err != nil {
				return orderID, err
			}
			if mapBrokerStatus(order.Status) == contracts.OrderStatusUnknown {
				return orderID, &IndeterminateError{Key: key, Cause: fmt.Errorf("broker reported an unknown order state")}
			}
		}
	}
	return orderID, nil
}

// OrderStatus returns the latest durable state for one caller key.
func (e *Engine) OrderStatus(key string) contracts.OrderResult {
	e.mu.Lock()
	defer e.mu.Unlock()
	rec, ok := e.state.Orders[key]
	if !ok {
		return contracts.OrderResult{Status: contracts.OrderStatusUnknown}
	}
	return resultOf(rec)
}

// Portfolio returns the latest marked portfolio with pending orders reserved.
func (e *Engine) Portfolio() contracts.Portfolio {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.portfolioLocked(e.clock())
}

// Ready reports whether the live account has been reconciled and is safe to accept
// work.
func (e *Engine) Ready() (bool, string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ready {
		return true, ""
	}
	if e.readyErr != nil {
		return false, e.readyErr.Error()
	}
	return false, "execution state is reconciling"
}

// Initialize performs the initial live broker/journal reconciliation.
func (e *Engine) Initialize(ctx context.Context) error {
	e.mu.Lock()
	for key, rec := range e.state.Orders {
		if rec.Status == contracts.OrderStatusSubmitting {
			rec.Status = contracts.OrderStatusUnknown
			rec.UpdatedAt = e.clock().UTC()
			e.state.Orders[key] = rec
		}
	}
	if err := e.saveLocked(); err != nil {
		e.readyErr = err
		e.mu.Unlock()
		return err
	}
	if e.cfg.PaperTrading {
		if hasLivePending(e.state.Orders) {
			e.ready = false
			e.readyErr = fmt.Errorf("paper journal contains unresolved live orders")
			e.mu.Unlock()
			return e.readyErr
		}
		e.mu.Unlock()
		e.mu.Lock()
		err := e.saveLocked()
		e.ready = err == nil
		e.readyErr = err
		e.mu.Unlock()
		if err != nil {
			return err
		}
		return e.RunExitPolicy(ctx)
	}
	e.ready = false
	e.mu.Unlock()
	return e.Reconcile(ctx)
}

// Reconcile refreshes uncertain orders, fills and marks without holding the engine
// mutex during broker I/O.
func (e *Engine) Reconcile(ctx context.Context) error {
	if e.cfg.PaperTrading {
		e.mu.Lock()
		pending := hasLivePending(e.state.Orders)
		e.mu.Unlock()
		if pending {
			return e.setUnready(fmt.Errorf("paper journal contains unresolved live orders"))
		}
		e.mu.Lock()
		err := e.saveLocked()
		e.ready = err == nil
		e.readyErr = err
		e.mu.Unlock()
		if err != nil {
			return err
		}
		return e.RunExitPolicy(ctx)
	}
	if e.orders == nil {
		return e.setUnready(fmt.Errorf("broker order reconciliation is unavailable"))
	}
	brokerOrders, err := e.orders.GetOrders(ctx)
	if err != nil {
		return e.setUnready(fmt.Errorf("list broker orders: %w", err))
	}
	e.mu.Lock()
	keys := make([]string, 0, len(e.state.Orders))
	for key, rec := range e.state.Orders {
		if activeOrderState(rec.Status) {
			keys = append(keys, key)
		}
	}
	e.mu.Unlock()
	sort.Strings(keys)
	for _, key := range keys {
		e.mu.Lock()
		rec, exists := e.state.Orders[key]
		e.mu.Unlock()
		if !exists {
			continue
		}
		id := rec.OrderID
		if id == "" {
			for _, candidate := range brokerOrders {
				if candidate.Tag == rec.Tag {
					id = candidate.ID
					break
				}
			}
			if id == "" {
				continue
			}
		}
		order, err := e.orders.GetOrder(ctx, id)
		if err != nil {
			continue
		}
		if err := e.applyBrokerOrder(key, order); err != nil {
			return e.setUnready(err)
		}
	}
	if err := e.refreshMarks(ctx); err != nil {
		return e.setUnready(err)
	}
	positions, err := e.client.GetPositions(ctx)
	if err != nil {
		return e.setUnready(fmt.Errorf("read broker positions: %w", err))
	}
	if err := e.comparePositions(positions); err != nil {
		return e.setUnready(err)
	}
	e.mu.Lock()
	if hasUnknown(e.state.Orders) {
		e.ready = false
		e.readyErr = fmt.Errorf("one or more order outcomes remain unresolved")
	} else {
		e.ready = true
		e.readyErr = nil
	}
	err = e.saveLocked()
	if err != nil {
		e.ready = false
		e.readyErr = err
	}
	e.mu.Unlock()
	if err != nil {
		return err
	}
	return e.RunExitPolicy(ctx)
}

func (e *Engine) applyBrokerOrder(key string, order broker.Order) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	rec, ok := e.state.Orders[key]
	if !ok {
		return fmt.Errorf("execution: unknown journal key %q", key)
	}
	if rec.OrderID != "" && order.ID != rec.OrderID {
		return fmt.Errorf("execution: broker order id mismatch for %q", key)
	}
	if order.Tag != "" && order.Tag != rec.Tag {
		return fmt.Errorf("execution: broker tag mismatch for %q", key)
	}
	if order.FilledQuantity < rec.FilledQuantity || order.FilledQuantity > rec.Request.Quantity {
		return fmt.Errorf("execution: invalid cumulative fill quantity for %q", key)
	}
	if order.Quantity != 0 && order.Quantity != rec.Request.Quantity ||
		(order.Symbol != "" && !strings.EqualFold(order.Symbol, rec.Request.Symbol)) ||
		(order.Side != "" && !strings.EqualFold(order.Side, rec.Request.Side)) {
		return fmt.Errorf("execution: broker order details mismatch for %q", key)
	}
	nextStatus := mapBrokerStatus(order.Status)
	if nextStatus == contracts.OrderStatusFilled && order.FilledQuantity != rec.Request.Quantity {
		return fmt.Errorf("execution: broker marked incomplete order %q FILLED (%d/%d)", key, order.FilledQuantity, rec.Request.Quantity)
	}
	if nextStatus == contracts.OrderStatusRejected && order.FilledQuantity != 0 {
		return fmt.Errorf("execution: broker marked filled order %q REJECTED", key)
	}
	if rec.Status == contracts.OrderStatusRejected && order.FilledQuantity > 0 {
		return fmt.Errorf("execution: rejected order %q later reports fills", key)
	}
	if order.FilledQuantity > 0 {
		if order.AverageFillPrice <= 0 || math.IsNaN(order.AverageFillPrice) || math.IsInf(order.AverageFillPrice, 0) {
			return fmt.Errorf("execution: invalid average fill price for %q", key)
		}
		cumulativeNotional := float64(order.FilledQuantity) * order.AverageFillPrice
		if cumulativeNotional+0.01 < rec.AppliedNotional {
			return fmt.Errorf("execution: cumulative fill notional decreased for %q", key)
		}
		if order.FilledQuantity == rec.FilledQuantity && math.Abs(cumulativeNotional-rec.AppliedNotional) > 0.01 {
			return fmt.Errorf("execution: cumulative fill notional changed without a new fill for %q", key)
		}
	}
	if order.FilledQuantity > rec.FilledQuantity {
		filledNotional := float64(order.FilledQuantity) * order.AverageFillPrice
		delta := order.FilledQuantity - rec.FilledQuantity
		deltaNotional := filledNotional - rec.AppliedNotional
		fillPrice := deltaNotional / float64(delta)
		if fillPrice <= 0 || math.IsNaN(fillPrice) || math.IsInf(fillPrice, 0) {
			return fmt.Errorf("execution: invalid incremental fill notional for %q", key)
		}
		sig := contracts.Signal{Symbol: rec.Request.Symbol, Side: rec.Request.Side}
		filledAt := order.UpdatedAt
		if filledAt.IsZero() {
			filledAt = e.clock()
		}
		if err := e.book.ApplyFill(sig, delta, fillPrice, filledAt); err != nil {
			e.ready = false
			e.readyErr = err
			return fmt.Errorf("account broker fill: %w", err)
		}
		rec.FilledQuantity = order.FilledQuantity
		rec.AppliedNotional = filledNotional
		rec.AverageFillPrice = order.AverageFillPrice
	}
	rec.OrderID = order.ID
	if !terminalOrderState(rec.Status) {
		rec.Status = nextStatus
	} else if rec.Status == contracts.OrderStatusCancelled && nextStatus == contracts.OrderStatusFilled {

		rec.Status = nextStatus
	}
	if rec.FilledQuantity == 0 && (rec.Status == contracts.OrderStatusRejected || rec.Status == contracts.OrderStatusCancelled) {
		e.cache.Delete(risk.DebounceKey(rec.Request.Symbol, rec.Request.Side))
	}
	if rec.Status == contracts.OrderStatusUnknown {
		e.ready = false
		e.readyErr = fmt.Errorf("broker reported an unknown order state for %q", key)
	}
	rec.UpdatedAt = order.UpdatedAt.UTC()
	if rec.UpdatedAt.IsZero() {
		rec.UpdatedAt = e.clock().UTC()
	}
	rec.LastError = ""
	e.state.Orders[key] = rec
	if err := e.saveLocked(); err != nil {
		e.ready = false
		e.readyErr = err
		return fmt.Errorf("persist broker update: %w", err)
	}
	return nil
}

func (e *Engine) comparePositions(brokerPositions []contracts.Position) error {
	want := e.book.State().Held
	got := make(map[string]int, len(brokerPositions))
	for _, pos := range brokerPositions {
		if pos.Symbol != "" {
			got[strings.ToUpper(pos.Symbol)] += pos.Quantity
		}
	}
	for sym, qty := range want {
		if got[sym] != qty {
			return fmt.Errorf("execution: broker position mismatch for %s: journal=%d broker=%d", sym, qty, got[sym])
		}
	}
	for sym, qty := range got {
		if want[sym] != qty {
			return fmt.Errorf("execution: untracked broker position %s x%d", sym, qty)
		}
	}
	return nil
}

func (e *Engine) refreshMarks(ctx context.Context) error {
	state := e.book.State()
	symbols := make([]string, 0, len(state.Held))
	for sym := range state.Held {
		symbols = append(symbols, sym)
	}
	if len(symbols) == 0 {
		return nil
	}
	if e.quotes == nil {
		return fmt.Errorf("broker quote client is unavailable for open positions")
	}
	marks, err := e.quotes.GetLastPrices(ctx, symbols)
	if err != nil {
		return err
	}
	now := e.clock().UTC()
	for _, sym := range symbols {
		if px := marks[sym]; px > 0 {
			if err := e.book.Mark(sym, px, now); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("missing current mark for %s", sym)
		}
	}
	return nil
}

func (e *Engine) currentPrices(ctx context.Context, symbols []string) (map[string]float64, error) {
	if e.quotes == nil {
		return nil, fmt.Errorf("broker quote client is unavailable")
	}
	return e.quotes.GetLastPrices(ctx, symbols)
}

func (e *Engine) place(ctx context.Context, req contracts.OrderRequest, tag string) (orderID string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("broker submission panicked: %v", recovered)
		}
	}()
	if tagged, ok := e.client.(interface {
		PlaceOrderTagged(context.Context, contracts.OrderRequest, string) (string, error)
	}); ok {
		return tagged.PlaceOrderTagged(ctx, req, tag)
	}
	return e.client.PlaceOrder(ctx, req)
}

func (e *Engine) portfolioLocked(now time.Time) contracts.Portfolio {
	p := e.book.SnapshotAt(now)
	p.ReservedPositions = make(map[string]int)
	p.ReservedSells = make(map[string]int)
	p.ReservedValues = make(map[string]float64)
	for _, rec := range e.state.Orders {
		if !activeOrderState(rec.Status) {
			continue
		}
		left := rec.Request.Quantity - rec.FilledQuantity
		if left <= 0 {
			continue
		}
		if rec.Request.Side == contracts.SideBuy {
			value := float64(left) * rec.RiskPrice
			p.ReservedCash += value
			p.ReservedPositions[rec.Request.Symbol] += left
			p.ReservedValues[rec.Request.Symbol] += value
		} else {
			p.ReservedSells[rec.Request.Symbol] += left
		}
	}
	return p
}

func restoreDebounceCache(e *Engine, now time.Time) {
	if e.debounceTTL <= 0 {
		return
	}
	for _, rec := range e.state.Orders {
		if rec.CreatedAt.IsZero() || !(rec.FilledQuantity > 0 || activeOrderState(rec.Status)) {
			continue
		}
		age := now.Sub(rec.CreatedAt)
		if age >= 0 && age < e.debounceTTL {
			e.cache.Set(risk.DebounceKey(rec.Request.Symbol, rec.Request.Side), rec.Tag, e.debounceTTL-age)
		}
	}
}

func (e *Engine) saveLocked() error {
	e.state.Book = e.book.State()
	return e.store.Save(e.state)
}

func (e *Engine) checkOrder(ctx context.Context, sig contracts.Signal, portfolio contracts.Portfolio) error {
	for _, check := range e.checks {
		if sig.Side == contracts.SideSell {
			switch check.(type) {
			case risk.DrawdownCheck, risk.DailyLossCheck:
				continue
			}
		}
		if err := check.Check(ctx, sig, portfolio); err != nil {
			return err
		}
	}
	return nil
}

func (e *Engine) duplicate(key string, prior OrderRecord) (string, error) {
	e.rejected.Inc()
	e.rejections.WithLabelValues(contracts.ReasonDuplicate).Inc()
	return prior.OrderID, contracts.Reject(contracts.ReasonDuplicate,
		fmt.Sprintf("idempotency key already processed as order %s (%s)", prior.OrderID, prior.Status))
}

func (e *Engine) reject(code, message string) error {
	e.rejected.Inc()
	e.rejections.WithLabelValues(code).Inc()
	return contracts.Reject(code, message)
}

func (e *Engine) setUnready(err error) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ready = false
	e.readyErr = err
	return err
}

func (e *Engine) unreadyErrorLocked() error {
	if e.readyErr != nil {
		return fmt.Errorf("execution is not ready: %w", e.readyErr)
	}
	return fmt.Errorf("execution is not ready")
}

func (e *Engine) Close() error {
	return e.CloseContext(context.Background())
}

// CloseContext waits for active broker writes before releasing the durable journal.
func (e *Engine) CloseContext(ctx context.Context) error {
	e.mu.Lock()
	e.closed = true
	e.mu.Unlock()
	done := make(chan struct{})
	go func() { e.submitWG.Wait(); close(done) }()
	select {
	case <-done:
	case <-ctx.Done():
		return ctx.Err()
	}
	e.closeOnce.Do(func() { e.closeErr = e.store.Close() })
	return e.closeErr
}

func orderTag(key string) string {
	h := sha256.Sum256([]byte(key))
	return "mft" + hex.EncodeToString(h[:8])
}

func resultOf(rec OrderRecord) contracts.OrderResult {
	return contracts.OrderResult{OrderID: rec.OrderID, Status: rec.Status, FilledQuantity: rec.FilledQuantity, AveragePrice: rec.AverageFillPrice}
}

func countPending(records map[string]OrderRecord) int {
	n := 0
	for _, rec := range records {
		if activeOrderState(rec.Status) {
			n++
		}
	}
	return n
}

func activeOrderState(status string) bool {
	return status == contracts.OrderStatusSubmitting || status == contracts.OrderStatusUnknown || status == contracts.OrderStatusOpen || status == contracts.OrderStatusPartial
}

func terminalOrderState(status string) bool {
	return status == contracts.OrderStatusFilled || status == contracts.OrderStatusCancelled || status == contracts.OrderStatusRejected || status == contracts.OrderStatusPaperFilled
}

func hasUnknown(records map[string]OrderRecord) bool {
	for _, rec := range records {
		if rec.Status == contracts.OrderStatusUnknown || rec.Status == contracts.OrderStatusSubmitting {
			return true
		}
	}
	return false
}

func hasLivePending(records map[string]OrderRecord) bool {
	for _, rec := range records {
		if activeOrderState(rec.Status) {
			return true
		}
	}
	return false
}

func mapBrokerStatus(status broker.OrderState) string {
	switch status {
	case broker.OrderOpen:
		return contracts.OrderStatusOpen
	case broker.OrderPartial:
		return contracts.OrderStatusPartial
	case broker.OrderFilled:
		return contracts.OrderStatusFilled
	case broker.OrderCancelled:
		return contracts.OrderStatusCancelled
	case broker.OrderRejected:
		return contracts.OrderStatusRejected
	default:
		return contracts.OrderStatusUnknown
	}
}

// IndeterminateError means the broker may have accepted a request whose reply was lost.
type IndeterminateError struct {
	Key   string
	Cause error
}

func (e *IndeterminateError) Error() string {
	return fmt.Sprintf("order outcome for key %q is unknown: %v", e.Key, e.Cause)
}
func (e *IndeterminateError) Unwrap() error { return e.Cause }

func isDefinitiveBrokerError(err error) bool { return errors.Is(err, broker.ErrInvalidOrder) }

// StartExecutionEngine reconciles before readiness and owns the background poller.
func StartExecutionEngine(lc fx.Lifecycle, e *Engine, cfg *config.Config, logger *zap.Logger) {
	metrics.DefaultChecks().Add(metrics.Checker{Name: "execution", Check: func(context.Context) error {
		if ready, reason := e.Ready(); !ready {
			return fmt.Errorf("execution: %s", reason)
		}
		return nil
	}})
	lc.Append(fx.Hook{
		OnStart: func(startCtx context.Context) error {
			runCtx, cancel := context.WithCancel(context.Background())
			done := make(chan struct{})
			e.mu.Lock()
			e.reconcileCancel = cancel
			e.reconcileDone = done
			e.mu.Unlock()
			initCtx, initCancel := context.WithTimeout(startCtx, 15*time.Second)
			if err := e.Initialize(initCtx); err != nil {
				logger.Error("execution startup reconciliation incomplete", zap.Error(err))
			}
			initCancel()
			go func() {
				defer close(done)
				ticker := time.NewTicker(time.Duration(cfg.Execution.ReconcileIntervalSeconds) * time.Second)
				defer ticker.Stop()
				for {
					select {
					case <-runCtx.Done():
						return
					case <-ticker.C:
						reconcileCtx, timeoutCancel := context.WithTimeout(runCtx, 15*time.Second)
						if err := e.Reconcile(reconcileCtx); err != nil {
							logger.Warn("execution reconciliation incomplete", zap.Error(err))
						}
						timeoutCancel()
					}
				}
			}()
			return nil
		},
		OnStop: func(ctx context.Context) error {
			e.mu.Lock()
			cancel := e.reconcileCancel
			done := e.reconcileDone
			e.reconcileCancel = nil
			e.reconcileDone = nil
			e.mu.Unlock()
			if cancel != nil {
				cancel()
			}
			if done != nil {
				select {
				case <-done:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return e.CloseContext(ctx)
		},
	})
}

// StartCacheSweeper reclaims expired debounce entries.
func StartCacheSweeper(lc fx.Lifecycle, cache *fluxkv.KV, logger *zap.Logger) {
	var cancel context.CancelFunc
	var done chan struct{}
	lc.Append(fx.Hook{OnStart: func(context.Context) error {
		ctx, stop := context.WithCancel(context.Background())
		cancel, done = stop, make(chan struct{})
		go func() {
			defer close(done)
			ticker := time.NewTicker(SweepInterval)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					if n := cache.Sweep(time.Now()); n > 0 {
						logger.Debug("swept expired fluxkv entries", zap.Int("count", n))
					}
				}
			}
		}()
		return nil
	}, OnStop: func(ctx context.Context) error {
		if cancel != nil {
			cancel()
		}
		if done != nil {
			select {
			case <-done:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	}})
}
