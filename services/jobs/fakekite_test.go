package jobs

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// errorBody is a Kite error envelope served verbatim when a test does not
// care about the wording.
const errorBody = `{"status":"error","errors":[{"error_code":"invalid_token","message":"Invalid access token"}]}`

// reply is what the fake Kite answers one request with. A zero status means
// 200; a non-empty body is served verbatim; otherwise rows is encoded into a
// success envelope.
type reply struct {
	status int
	body   string
	rows   [][]any
}

// fakeKite is a stand-in for Kite's /data/historical endpoint. Every test that
// exercises the fetch path talks to it over a real loopback connection, so the
// URL shape, the auth header, the 429 status and the wire format are all
// covered without a packet leaving the machine.
type fakeKite struct {
	mu sync.Mutex

	// defaultReply is served when responder is nil.
	defaultReply reply
	// responder, when set, computes the reply for a request. It lets a test
	// vary the answer by date range or by symbol.
	responder func(r *http.Request) reply
	// onRequest runs after a request is recorded and before it is answered.
	// It is how a test interrupts a run mid-flight.
	onRequest func(r *http.Request)
	// throttles is the number of leading requests answered with a 429.
	throttles int

	requests int
	paths    []string
	queries  []string
	auths    []string
}

// ServeHTTP implements the Kite historical route.
func (f *fakeKite) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.requests++
	f.paths = append(f.paths, r.URL.Path)
	f.queries = append(f.queries, r.URL.RawQuery)
	f.auths = append(f.auths, r.Header.Get("Authorization"))
	throttles := f.throttles
	if throttles > 0 {
		f.throttles--
	}
	responder, onRequest, fallback := f.responder, f.onRequest, f.defaultReply
	f.mu.Unlock()

	if onRequest != nil {
		onRequest(r)
	}
	if throttles > 0 {
		writeJSON(w, http.StatusTooManyRequests,
			`{"status":"error","errors":[{"error_code":"429","message":"Too many requests"}]}`)
		return
	}

	rep := fallback
	if responder != nil {
		rep = responder(r)
	}
	if rep.body != "" || rep.status != 0 {
		status := rep.status
		if status == 0 {
			status = http.StatusOK
		}
		body := rep.body
		if body == "" {
			body = errorBody
		}
		writeJSON(w, status, body)
		return
	}
	writeJSON(w, http.StatusOK, successBody(rep.rows))
}

// count returns how many requests the fake has served.
func (f *fakeKite) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests
}

// lastPath returns the most recent request path.
func (f *fakeKite) lastPath() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.paths) == 0 {
		return ""
	}
	return f.paths[len(f.paths)-1]
}

// lastQuery returns the most recent request query string.
func (f *fakeKite) lastQuery() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.queries) == 0 {
		return ""
	}
	return f.queries[len(f.queries)-1]
}

// lastAuth returns the most recent Authorization header.
func (f *fakeKite) lastAuth() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.auths) == 0 {
		return ""
	}
	return f.auths[len(f.auths)-1]
}

// successBody encodes rows into a Kite success envelope.
func successBody(rows [][]any) string {
	if rows == nil {
		rows = [][]any{}
	}
	raw, err := json.Marshal(map[string]any{
		"status": "success",
		"data":   map[string]any{"candles": rows},
	})
	if err != nil {
		return errorBody
	}
	return string(raw)
}

// writeJSON writes body with a status code.
func writeJSON(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(body))
}

// rangeOf extracts the [from, to) bounds a Kite historical URL carries. The
// path is /data/historical/<exchange>/<symbol>/<interval>/<from>/<to>, so the
// bounds are the fifth and sixth segments.
func rangeOf(t *testing.T, r *http.Request) (from, to time.Time) {
	t.Helper()
	parts := splitPath(r.URL.Path)
	if len(parts) != 7 {
		t.Fatalf("path %q has %d segments, want 7", r.URL.Path, len(parts))
	}
	from, err := time.Parse(time.DateOnly, parts[5])
	if err != nil {
		t.Fatalf("parse from %q: %v", parts[5], err)
	}
	to, err = time.Parse(time.DateOnly, parts[6])
	if err != nil {
		t.Fatalf("parse to %q: %v", parts[6], err)
	}
	return from.UTC(), to.UTC()
}

// splitPath splits a URL path into its non-empty segments.
func splitPath(path string) []string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	out := parts[:0]
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// startFakeKite starts a fake Kite on loopback and returns it with the server
// URL. The server is closed when the test finishes.
func startFakeKite(t *testing.T, f *fakeKite) string {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return srv.URL
}
