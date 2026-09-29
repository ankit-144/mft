package metrics

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"sync"
	"time"
)

// StatusOK and StatusDegraded are the values of Report.Status.
const (
	StatusOK       = "ok"
	StatusDegraded = "degraded"
)

// Checker is a named dependency probe, e.g. broker connectivity or Parquet
// writability. A component registers one and the metrics server answers
// /readyz from it, so an orchestrator can act on "process is up but cannot
// trade" without parsing logs.
type Checker struct {
	// Name identifies the dependency in the report, e.g. "broker".
	Name string
	// Check reports nil when the dependency is usable. It must respect ctx.
	Check func(ctx context.Context) error
}

// CheckResult is one dependency's outcome in a health report.
type CheckResult struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

// Report is the JSON body of /healthz and /readyz.
type Report struct {
	Status        string        `json:"status"`
	Service       string        `json:"service,omitempty"`
	UptimeSeconds float64       `json:"uptime_seconds"`
	Checks        []CheckResult `json:"checks,omitempty"`
}

// Checks is a concurrency-safe set of dependency probes. Components add to it
// from their constructors; the health endpoints read it on every request.
type Checks struct {
	mu    sync.RWMutex
	items []Checker
}

// NewChecks returns an empty probe set.
func NewChecks() *Checks { return &Checks{} }

// Add appends a probe. Adding the same name twice keeps the first, so a
// component rebuilt in a test cannot double-report a dependency.
func (c *Checks) Add(checker Checker) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, existing := range c.items {
		if existing.Name == checker.Name {
			return
		}
	}
	c.items = append(c.items, checker)
}

// Len returns the number of registered probes.
func (c *Checks) Len() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Run executes every probe concurrently and returns the results. A nil *Checks
// reports success with no dependencies, which is the correct answer for a
// service that has nothing external to depend on.
func (c *Checks) Run(ctx context.Context) []CheckResult {
	if c == nil {
		return nil
	}
	c.mu.RLock()
	items := make([]Checker, len(c.items))
	copy(items, c.items)
	c.mu.RUnlock()

	results := make([]CheckResult, len(items))
	var wg sync.WaitGroup
	for i, item := range items {
		wg.Add(1)
		go func(i int, item Checker) {
			defer wg.Done()
			result := CheckResult{Name: item.Name, Status: StatusOK}
			if item.Check == nil {
				result.Status = StatusDegraded
				result.Error = "no check function"
			} else if err := item.Check(ctx); err != nil {
				result.Status = StatusDegraded
				result.Error = err.Error()
			}
			results[i] = result
		}(i, item)
	}
	wg.Wait()
	return results
}

// defaultChecks is the process-wide probe set used by Server. It exists so
// that a component can participate in readiness with a single call and no
// change to the shared fx wiring in core/fx.go, which this component does not
// own.
var (
	defaultChecksOnce sync.Once
	defaultChecks     *Checks
)

// DefaultChecks returns the process-wide probe set backing /readyz. Services
// add to it during construction:
//
//	metrics.DefaultChecks().Add(metrics.Checker{Name: "broker", Check: c.Ping})
func DefaultChecks() *Checks {
	defaultChecksOnce.Do(func() { defaultChecks = NewChecks() })
	return defaultChecks
}

// Health answers liveness and readiness for one service. It is an
// http.Handler, so it can be mounted directly on the metrics server.
type Health struct {
	service string
	checks  *Checks
	started time.Time
}

// NewHealth returns a health handler for the named service, probing checks.
func NewHealth(service string, checks *Checks) *Health {
	return &Health{service: service, checks: checks, started: processStart}
}

// Livez reports liveness: the process is running and able to answer. It never
// fails, because a failing liveness probe restarts a process that is merely
// unhealthy, which is the opposite of what you want from a trading service.
func (h *Health) Livez() Report {
	return Report{
		Status:        StatusOK,
		Service:       h.service,
		UptimeSeconds: round3(time.Since(h.started).Seconds()),
	}
}

// Readyz reports readiness: liveness plus every dependency probe. Callers get
// 503 when any probe fails.
func (h *Health) Readyz(ctx context.Context) (Report, bool) {
	report := h.Livez()
	report.Checks = h.checks.Run(ctx)
	for _, c := range report.Checks {
		if c.Status != StatusOK {
			report.Status = StatusDegraded
			return report, false
		}
	}
	return report, true
}

// ServeHTTP routes /healthz to liveness and /readyz to readiness, answering
// 404 for anything else.
func (h *Health) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/healthz":
		writeJSON(w, http.StatusOK, h.Livez())
	case "/readyz":
		report, ok := h.Readyz(r.Context())
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, report)
			return
		}
		writeJSON(w, http.StatusOK, report)
	default:
		http.NotFound(w, r)
	}
}

// writeJSON renders v as an indented JSON body with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		http.Error(w, `{"error":"encode_failed"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(append(body, '\n'))
}

// round3 trims float noise so uptime reads cleanly in a health payload.
func round3(f float64) float64 {
	return math.Round(f*1000) / 1000
}
