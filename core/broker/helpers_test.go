package broker

import (
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mft/core/config"
	"go.uber.org/zap"
)

// testAccessToken is a fake token. No test in this package ever reaches the
// real Kite API: every server is a local httptest instance.
const testAccessToken = "test-access-token"

// streamTimeout bounds every wait in these tests, so a broken streamer fails
// the test instead of hanging the package.
const streamTimeout = 10 * time.Second

func localTestServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if errors.Is(err, syscall.EPERM) || errors.Is(err, syscall.EACCES) {
		t.Skipf("socket integration unavailable: %v", err)
	}
	if err != nil {
		t.Fatalf("listen for fixture: %v", err)
	}
	server := httptest.NewUnstartedServer(handler)
	_ = server.Listener.Close()
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)
	return server
}

// The tokens the instrument dump maps RELIANCE and TCS onto.
const (
	relianceToken int64 = 738560
	tcsToken      int64 = 3419705
)

// instrumentDump is a trimmed stand-in for the Kite CSV instrument dump. It
// covers the cases the mapping has to get right: NSE and BSE listings of the
// same symbol, a future and an option leg that must be ignored, and a symbol
// that is listed but misspelled in config.
const instrumentDump = `instrument_token,exchange,tradingsymbol,name,expiry,strike,option_type,tick_size,lot_size
256265,NFO,RELIANCE25SEPFUT,RELIANCE SEP FUT,"",0,,5,500
738560,NSE,RELIANCE,RELIANCE EQUITY,"",0,,0.05,1
3419705,NSE,TCS,TCS EQUITY,"",0,,0.05,1
256265,BSE,RELIANCE,RELIANCE EQUITY BSE,"",0,,0.05,1
256265,NFO,RELIANCE25SEP900CE,RELIANCE CALL,"2026-09-25",900,CE,0.05,500
nope,NSE,MISSPELT,MISSPELT EQUITY,"",0,,0.05,1
`

// newTestKite returns a Kite wired to a local httptest REST server. Call
// attachWS to point the WebSocket half at a scripted local server.
func newTestKite(t *testing.T, rest http.Handler) *Kite {
	t.Helper()

	restServer := localTestServer(t, rest)

	k, err := NewKiteFromConfig(config.BrokerConfig{
		APIKey:                  "test-api-key",
		APISecret:               "test-api-secret",
		AccessToken:             testAccessToken,
		Instruments:             []string{"RELIANCE", "TCS"},
		RequestTimeoutSeconds:   5,
		ReconnectMaxBackoffSecs: 1,
	})
	if err != nil {
		t.Fatalf("NewKiteFromConfig: %v", err)
	}
	k.SetLogger(zap.NewNop())
	k.endpoints.httpBase = restServer.URL
	k.reconnectBase = 5 * time.Millisecond
	k.pongTimeout = 2 * time.Second
	k.writeWait = time.Second
	return k
}

// instrumentHandler serves the CSV instrument dump and, when auth is
// non-nil, records the headers of the last request that fetched it.
func instrumentHandler(t *testing.T, auth *http.Header) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/instruments" {
			http.NotFound(w, r)
			return
		}
		if auth != nil {
			*auth = r.Header.Clone()
		}
		w.Header().Set("Content-Type", "text/csv")
		_, _ = w.Write([]byte(instrumentDump))
	})
}

// setEndpoints points an existing connector at local test servers.
func (k *Kite) setEndpoints(httpBase, wsBase string) {
	k.endpoints = endpoints{httpBase: httpBase, wsBase: wsBase}
}

// wsHandler runs once per accepted WebSocket connection, on the server's
// goroutine, with a 1-based connection index.
type wsHandler func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, n int)

// wsTestServer is a local WebSocket server standing in for
// wss://ws.kite.trade. It counts connections and records the frames the
// client sends.
type wsTestServer struct {
	*httptest.Server

	mu    sync.Mutex
	dials int
	subs  [][]int64
	modes []json.RawMessage
	pongs int
}

// attachWS starts a local WebSocket test server and points the connector's
// WebSocket half at it.
func attachWS(t *testing.T, k *Kite, handle wsHandler) *wsTestServer {
	t.Helper()
	ws := newWSTestServer(t, handle)
	k.endpoints.wsBase = ws.wsBaseURL()
	return ws
}

// newWSTestServer starts a local WebSocket test server. handle, when non-nil,
// runs once per accepted connection.
func newWSTestServer(t *testing.T, handle wsHandler) *wsTestServer {
	t.Helper()

	ws := &wsTestServer{}
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}

	ws.Server = localTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("websocket upgrade: %v", err)
			return
		}
		defer conn.Close()

		ws.mu.Lock()
		ws.dials++
		n := ws.dials
		ws.mu.Unlock()

		if handle != nil {
			handle(t, ws, conn, n)
		}
	}))
	return ws
}

// wsBaseURL returns the ws:// form of the test server address.
func (ws *wsTestServer) wsBaseURL() string {
	return "ws" + strings.TrimPrefix(ws.URL, "http")
}

// Connections returns the number of accepted WebSocket connections.
func (ws *wsTestServer) Connections() int {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	return ws.dials
}

// Subscriptions returns the "v" list of each subscribe frame received, in
// arrival order.
func (ws *wsTestServer) Subscriptions() [][]int64 {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	return append([][]int64(nil), ws.subs...)
}

func (ws *wsTestServer) Modes() []json.RawMessage {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	return append([]json.RawMessage(nil), ws.modes...)
}

func (ws *wsTestServer) Pongs() int {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	return ws.pongs
}

// note files a decoded client frame under subscribe or pong.
func (ws *wsTestServer) note(cmd wsCommand) {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	switch cmd.Action {
	case "subscribe":
		values, _ := json.Marshal(cmd.Values)
		var tokens []int64
		_ = json.Unmarshal(values, &tokens)
		ws.subs = append(ws.subs, tokens)
	case "mode":
		values, _ := json.Marshal(cmd.Values)
		ws.modes = append(ws.modes, values)
	}
}

func (ws *wsTestServer) notePong() {
	ws.mu.Lock()
	ws.pongs++
	ws.mu.Unlock()
}

// record reads one client frame and classifies it as a subscribe or a pong.
// A read failure is only logged: a connection closing is how most handlers
// here end, and asserting on it would race with test teardown.
func (ws *wsTestServer) record(t *testing.T, conn *websocket.Conn) {
	t.Helper()
	_ = conn.SetReadDeadline(time.Now().Add(streamTimeout))
	_, data, err := conn.ReadMessage()
	if err != nil {
		t.Logf("read client frame: %v", err)
		return
	}
	var cmd wsCommand
	if err := json.Unmarshal(data, &cmd); err != nil {
		t.Errorf("decode client frame %q: %v", data, err)
		return
	}
	ws.note(cmd)
	if cmd.Action == "pong" {
		ws.notePong()
	}
}

// readSubscribe drains the subscribe and mode frames a client sends on
// connect and returns the subscription list it asked for.
func (ws *wsTestServer) readSubscribe(t *testing.T, conn *websocket.Conn) []int64 {
	t.Helper()
	ws.record(t, conn)
	ws.record(t, conn)
	subs := ws.Subscriptions()
	if len(subs) == 0 {
		return nil
	}
	return subs[len(subs)-1]
}
