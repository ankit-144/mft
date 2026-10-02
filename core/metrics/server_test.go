package metrics

import (
	"context"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/testutil"
	"go.uber.org/fx/fxtest"
)

// TestServerLifecycle covers the two properties an operator depends on: the
// listener is opened during start, so a port clash fails startup, and stop
// drains in-flight scrapes and closes the port.
func TestServerLifecycle(t *testing.T) {
	addr := freeAddr(t)
	cfg := &config.Config{
		App:     config.AppConfig{Name: "ingestion"},
		Metrics: config.MetricsConfig{Addr: addr, Path: "/metrics"},
	}
	reg := testutil.NewRegistry()
	Counter(reg, "mft_ingestion_ticks_processed_total", "Ticks processed from the broker.")

	lc := fxtest.NewLifecycle(t)
	Server(lc, cfg, Handler(reg), testutil.NewLogger())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := lc.Start(ctx); err != nil {
		t.Fatalf("start metrics server: %v", err)
	}

	body, status := get(t, "http://"+addr+"/metrics")
	if status != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", status)
	}
	if !strings.Contains(body, "mft_ingestion_ticks_processed_total") {
		t.Errorf("/metrics does not expose the registered metric:\n%s", body)
	}
	if body, status := get(t, "http://"+addr+"/healthz"); status != http.StatusOK {
		t.Errorf("GET /healthz = %d, want 200: %s", status, body)
	}

	if err := lc.Stop(ctx); err != nil {
		t.Fatalf("stop metrics server: %v", err)
	}
	if _, err := http.Get("http://" + addr + "/healthz"); err == nil {
		t.Error("the metrics server is still accepting connections after Stop")
	}
}

// TestServerFailsLoudlyOnABusyPort guards against the failure mode where a
// process starts, looks healthy, and silently is not scrapeable.
func TestServerFailsLoudlyOnABusyPort(t *testing.T) {
	addr := freeAddr(t)
	cfg := &config.Config{
		App:     config.AppConfig{Name: "ingestion"},
		Metrics: config.MetricsConfig{Addr: addr, Path: "/metrics"},
	}
	handler := Handler(testutil.NewRegistry())

	first, second := fxtest.NewLifecycle(t), fxtest.NewLifecycle(t)
	Server(first, cfg, handler, testutil.NewLogger())
	Server(second, cfg, handler, testutil.NewLogger())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := first.Start(ctx); err != nil {
		t.Fatalf("start the first metrics server: %v", err)
	}
	t.Cleanup(func() { _ = first.Stop(ctx) })

	if err := second.Start(ctx); err == nil {
		t.Error("a second metrics server bound the same port without complaining")
	}
}

// freeAddr returns a loopback address that is free right now.
func freeAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("cannot bind a loopback port: %v", err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatalf("release %s: %v", addr, err)
	}
	return addr
}

// get performs a GET against a loopback address and returns the body and status.
func get(t *testing.T, url string) (string, int) {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		t.Fatalf("build request for %s: %v", url, err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read %s: %v", url, err)
	}
	return string(body), resp.StatusCode
}
