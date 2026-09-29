package metrics

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mft/core/config"
)

func TestChecksRunReportsFailures(t *testing.T) {
	t.Parallel()
	checks := NewChecks()
	checks.Add(Checker{Name: "broker", Check: func(context.Context) error { return nil }})
	checks.Add(Checker{Name: "storage", Check: func(context.Context) error { return errors.New("disk full") }})
	checks.Add(Checker{Name: "kite", Check: nil})

	results := checks.Run(context.Background())
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	byName := map[string]CheckResult{}
	for _, r := range results {
		byName[r.Name] = r
	}
	if byName["broker"].Status != StatusOK {
		t.Errorf("broker = %+v, want ok", byName["broker"])
	}
	if byName["storage"].Status != StatusDegraded || byName["storage"].Error != "disk full" {
		t.Errorf("storage = %+v, want degraded with the cause", byName["storage"])
	}
	if byName["kite"].Status != StatusDegraded {
		t.Errorf("a nil Check function must not be reported as healthy: %+v", byName["kite"])
	}
}

func TestChecksAddIsIdempotent(t *testing.T) {
	t.Parallel()
	checks := NewChecks()
	checks.Add(Checker{Name: "broker", Check: func(context.Context) error { return nil }})
	checks.Add(Checker{Name: "broker", Check: func(context.Context) error { return errors.New("ignored") }})
	if got := checks.Len(); got != 1 {
		t.Fatalf("Len = %d, want 1", got)
	}
	if checks.Run(context.Background())[0].Status != StatusOK {
		t.Error("the second Add replaced the first probe")
	}
}

func TestHealthEndpoints(t *testing.T) {
	t.Parallel()
	checks := NewChecks()
	checks.Add(Checker{Name: "broker", Check: func(context.Context) error { return errors.New("no session") }})
	health := NewHealth("execution", checks)

	cases := []struct {
		path string
		want int
	}{
		{path: "/healthz", want: http.StatusOK},
		{path: "/readyz", want: http.StatusServiceUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			t.Parallel()
			rec := httptest.NewRecorder()
			health.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tc.path, nil))
			if rec.Code != tc.want {
				t.Errorf("GET %s = %d, want %d", tc.path, rec.Code, tc.want)
			}
			if ct := rec.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
				t.Errorf("Content-Type = %q", ct)
			}
		})
	}

	rec := httptest.NewRecorder()
	health.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("GET /metrics on the health handler = %d, want 404", rec.Code)
	}

	var report Report
	if err := json.Unmarshal(recorder(t, health, "/readyz").Body.Bytes(), &report); err != nil {
		t.Fatalf("decode readyz: %v", err)
	}
	if report.Status != StatusDegraded {
		t.Errorf("readyz status = %q, want %q", report.Status, StatusDegraded)
	}
	if report.Service != "execution" {
		t.Errorf("service = %q, want execution", report.Service)
	}
	if len(report.Checks) != 1 || report.Checks[0].Name != "broker" {
		t.Errorf("checks = %+v, want the broker probe", report.Checks)
	}

	var live Report
	if err := json.Unmarshal(recorder(t, health, "/healthz").Body.Bytes(), &live); err != nil {
		t.Fatalf("decode healthz: %v", err)
	}
	if live.Status != StatusOK || len(live.Checks) != 0 {
		t.Errorf("healthz = %+v, want ok with no dependency detail", live)
	}
	if live.UptimeSeconds < 0 {
		t.Errorf("uptime = %v, want >= 0", live.UptimeSeconds)
	}
}

func TestMuxRoutes(t *testing.T) {
	cfg := &config.Config{
		App:     config.AppConfig{Name: "ingestion"},
		Metrics: config.MetricsConfig{Path: "/internal/metrics"},
	}
	mux := Mux(cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	}))

	cases := []struct {
		path string
		want int
	}{
		{path: "/internal/metrics", want: http.StatusTeapot},
		{path: "/metrics", want: http.StatusNotFound},
		{path: "/healthz", want: http.StatusOK},
		{path: "/readyz", want: http.StatusOK},
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tc.path, nil))
			if rec.Code != tc.want {
				t.Errorf("GET %s = %d, want %d", tc.path, rec.Code, tc.want)
			}
		})
	}
}

func TestDefaultChecksIsShared(t *testing.T) {
	first, second := DefaultChecks(), DefaultChecks()
	if first != second {
		t.Fatal("DefaultChecks returned different sets; components would report into the void")
	}
	before := first.Len()
	first.Add(Checker{Name: "test-probe", Check: func(context.Context) error { return nil }})
	t.Cleanup(func() { first.mu.Lock(); first.items = first.items[:before]; first.mu.Unlock() })

	if got := DefaultChecks().Len(); got != before+1 {
		t.Errorf("Len = %d, want %d", got, before+1)
	}
}

// recorder runs one health request and returns the recorder.
func recorder(t *testing.T, health *Health, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	health.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec
}
