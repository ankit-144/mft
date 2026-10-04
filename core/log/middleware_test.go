package log

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestCorrelationIDRoundTrip(t *testing.T) {
	t.Parallel()
	id := NewCorrelationID()
	if len(id) < 16 {
		t.Errorf("NewCorrelationID = %q, want at least 128 bits of entropy", id)
	}
	if other := NewCorrelationID(); other == id {
		t.Error("NewCorrelationID returned the same id twice")
	}
	if got := CorrelationIDFromContext(context.Background()); got != "" {
		t.Errorf("CorrelationIDFromContext on a bare context = %q, want empty", got)
	}
	ctx := WithCorrelationID(context.Background(), id)
	if got := CorrelationIDFromContext(ctx); got != id {
		t.Errorf("CorrelationIDFromContext = %q, want %q", got, id)
	}
}

func TestWithIsANoOpWithoutAnID(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")
	With(logger, "").Info("no correlation id")
	if strings.Contains(buf.String(), "correlation_id") {
		t.Errorf("an empty correlation id must not add a field:\n%s", buf.String())
	}
}

// TestMiddlewarePropagatesCorrelationID covers the inference → execution hop:
// the id the caller sent must be the id in the context, the log line and the
// response header.
func TestMiddlewarePropagatesCorrelationID(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")

	var seen string
	handler := Middleware(logger)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = CorrelationIDFromContext(r.Context())
		WithContext(r.Context(), logger).Info("inside handler")
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"order_id":"4412"}`))
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/signals", nil)
	req.Header.Set(RequestIDHeader, "inference-1031-RELIANCE")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	const want = "inference-1031-RELIANCE"
	if seen != want {
		t.Errorf("context id = %q, want %q", seen, want)
	}
	if got := rec.Header().Get(RequestIDHeader); got != want {
		t.Errorf("response header %s = %q, want %q", RequestIDHeader, got, want)
	}
	if rec.Code != http.StatusAccepted {
		t.Errorf("status = %d, want 202", rec.Code)
	}
	if !strings.Contains(buf.String(), want) {
		t.Errorf("the correlation id is missing from the log:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "/v1/signals") || !strings.Contains(buf.String(), `"status":202`) {
		t.Errorf("the request line is incomplete:\n%s", buf.String())
	}
}

func TestMiddlewareGeneratesCorrelationID(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")

	var seen string
	handler := Middleware(logger)(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		seen = CorrelationIDFromContext(r.Context())
	}))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/portfolio", nil))

	if seen == "" {
		t.Fatal("no correlation id was generated")
	}
	if got := rec.Header().Get(RequestIDHeader); got != seen {
		t.Errorf("response header = %q, want the generated id %q", got, seen)
	}
	if !strings.Contains(buf.String(), seen) {
		t.Errorf("the generated id is missing from the log:\n%s", buf.String())
	}
}

// TestMiddlewareRedactsURIGuards the one place a token could still escape: a
// caller who puts one in the query string.
func TestMiddlewareRedactsURIGuards(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")
	handler := Middleware(logger)(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	handler.ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest(http.MethodGet, "/v1/context?symbol=RELIANCE&access_token="+accessToken, nil))

	assertNoSecrets(t, buf.String())
	if !strings.Contains(buf.String(), "RELIANCE") {
		t.Errorf("the useful part of the URI was dropped:\n%s", buf.String())
	}
}

func TestMiddlewareLogsByStatus(t *testing.T) {
	t.Parallel()
	cases := []struct {
		status int
		level  string
	}{
		{status: http.StatusOK, level: `"level":"debug"`},
		{status: http.StatusBadRequest, level: `"level":"warn"`},
		{status: http.StatusInternalServerError, level: `"level":"error"`},
	}
	for _, tc := range cases {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			t.Parallel()
			logger, buf := loggerFor(t, "prod", "debug")
			handler := Middleware(logger)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
			}))
			handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/signals", nil))
			if !strings.Contains(buf.String(), tc.level) {
				t.Errorf("status %d logged at %s, want %s:\n%s", tc.status, "?", tc.level, buf.String())
			}
			if !strings.Contains(buf.String(), "127.0.0.1") && !strings.Contains(buf.String(), "remote_addr") {
				t.Errorf("the remote address was not logged:\n%s", buf.String())
			}
		})
	}
}

func TestMiddlewareMeasuresDuration(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")
	handler := Middleware(logger)(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		time.Sleep(2 * time.Millisecond)
	}))
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/v1/portfolio", nil))
	if !strings.Contains(buf.String(), `"duration":`) {
		t.Errorf("the request duration was not logged:\n%s", buf.String())
	}
}

func TestClientIP(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"10.0.0.5:54321": "10.0.0.5",
		"[::1]:443":      "::1",
		"":               "",
		"garbage":        "garbage",
	}
	for in, want := range cases {
		if got := clientIP(in); got != want {
			t.Errorf("clientIP(%q) = %q, want %q", in, got, want)
		}
	}
}
