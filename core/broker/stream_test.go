package broker

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mft/core/config"
)

func symbolMap() map[int64]string {
	return map[int64]string{relianceToken: "RELIANCE", tcsToken: "TCS"}
}

func TestDecodeTickFrame(t *testing.T) {
	ts := time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC)
	frame := encodeTickFrame(relianceToken, ts, 2934.5, 7, [4]float64{2900, 2950, 2899.5, 2934.5}, 1_000_000, 1_000_000)

	tick, err := decodeTickFrame(frame, symbolMap())
	if err != nil {
		t.Fatalf("decodeTickFrame: %v", err)
	}
	want := Tick{
		Symbol:    "RELIANCE",
		Token:     relianceToken,
		Price:     2934.5,
		Volume:    1_000_000,
		Timestamp: ts,
	}
	if tick != want {
		t.Errorf("got %+v, want %+v", tick, want)
	}
	if tick.Timestamp.Location() != time.UTC {
		t.Errorf("Timestamp location = %v, want UTC", tick.Timestamp.Location())
	}
}

// TestDecodeTickFrameMatchesKiteWireLayout pins the frame layout
// independently of encodeTickFrame, so a change to either half of the round
// trip cannot silently agree with the other.
func TestDecodeTickFrameMatchesKiteWireLayout(t *testing.T) {
	ts := time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC)
	frame := make([]byte, 68)
	binary.BigEndian.PutUint32(frame[0:], 3419705)
	binary.BigEndian.PutUint32(frame[4:], uint32(ts.Unix()))
	binary.BigEndian.PutUint64(frame[8:], math.Float64bits(4100.25))
	binary.BigEndian.PutUint32(frame[16:], 12)
	binary.BigEndian.PutUint64(frame[20:], math.Float64bits(4000))
	binary.BigEndian.PutUint64(frame[28:], math.Float64bits(4110))
	binary.BigEndian.PutUint64(frame[36:], math.Float64bits(3990))
	binary.BigEndian.PutUint64(frame[44:], math.Float64bits(4100.25))
	binary.BigEndian.PutUint64(frame[52:], 555)
	binary.BigEndian.PutUint64(frame[60:], 555)

	tick, err := decodeTickFrame(frame, symbolMap())
	if err != nil {
		t.Fatalf("decodeTickFrame: %v", err)
	}
	want := Tick{Symbol: "TCS", Token: 3419705, Price: 4100.25, Volume: 555, Timestamp: ts}
	if tick != want {
		t.Errorf("got %+v, want %+v", tick, want)
	}
}

func TestDecodeTickFrameErrors(t *testing.T) {
	t.Run("short frame", func(t *testing.T) {
		if _, err := decodeTickFrame(make([]byte, 12), symbolMap()); !errors.Is(err, ErrInvalidOrder) {
			t.Fatalf("error = %v, want ErrInvalidOrder", err)
		}
	})
	t.Run("unsubscribed token", func(t *testing.T) {
		frame := encodeTickFrame(999, time.Now(), 1, 1, [4]float64{}, 0, 0)
		if _, err := decodeTickFrame(frame, symbolMap()); !errors.Is(err, ErrInstrumentNotFound) {
			t.Fatalf("error = %v, want ErrInstrumentNotFound", err)
		}
	})
}

func TestPongReplyEchoesTimestamp(t *testing.T) {
	got := pongReply([]byte(`{"a":["ping",1788000000]}`))
	if got.Action != "pong" {
		t.Errorf("Action = %q, want pong", got.Action)
	}
	if len(got.Values) != 1 || got.Values[0] != "1788000000" {
		t.Errorf("Values = %v, want [1788000000]", got.Values)
	}

	// A ping with no timestamp must still produce a well-formed pong rather
	// than an empty frame Kite would reject.
	if got := pongReply([]byte(`{"a":["ping"]}`)); got.Action != "pong" || len(got.Values) != 1 {
		t.Errorf("pongReply with no timestamp = %+v, want a single-value pong", got)
	}
}

func TestStreamDeliversTicksAndSubscribes(t *testing.T) {
	ts := time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC)
	frames := [][]byte{
		encodeTickFrame(relianceToken, ts, 2934.5, 7, [4]float64{}, 100, 100),
		encodeTickFrame(tcsToken, ts.Add(time.Second), 4100.25, 3, [4]float64{}, 200, 200),
	}

	var subscribed []string
	k := newTestKite(t, instrumentHandler(t, nil))
	attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, _ int) {
		subscribed = ws.readSubscribe(t, conn)
		for _, f := range frames {
			if err := conn.WriteMessage(websocket.BinaryMessage, f); err != nil {
				t.Logf("write tick: %v", err)
				return
			}
		}
		// Block until the client goes away so the handler outlives the
		// sends without racing the assertions.
		ws.record(t, conn)
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := make(chan Tick, 8)

	done := make(chan error, 1)
	go func() { done <- k.Stream(ctx, nil, out) }()

	got := make([]Tick, 0, 2)
	for len(got) < 2 {
		select {
		case tick := <-out:
			got = append(got, tick)
		case <-time.After(streamTimeout):
			t.Fatal("timed out waiting for ticks")
		}
	}

	if got[0].Symbol != "RELIANCE" || got[0].Price != 2934.5 || got[0].Token != relianceToken {
		t.Errorf("first tick = %+v, want RELIANCE token 738560 at 2934.5", got[0])
	}
	if got[1].Symbol != "TCS" || got[1].Price != 4100.25 || got[1].Token != tcsToken {
		t.Errorf("second tick = %+v, want TCS token 3419705 at 4100.25", got[1])
	}
	if !got[0].Timestamp.Equal(ts) {
		t.Errorf("tick timestamp = %v, want %v", got[0].Timestamp, ts)
	}

	want := []string{"NSE|RELIANCE|738560", "NSE|TCS|3419705"}
	if len(subscribed) != len(want) {
		t.Fatalf("subscribed to %v, want %v", subscribed, want)
	}
	for i := range want {
		if subscribed[i] != want[i] {
			t.Errorf("subscription %d = %q, want %q", i, subscribed[i], want[i])
		}
	}

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream = %v, want context.Canceled", err)
		}
	case <-time.After(streamTimeout):
		t.Fatal("Stream did not return after the context was cancelled")
	}
	if n := k.Delivered(); n != 2 {
		t.Errorf("Delivered = %d, want 2", n)
	}
	if k.Sequence() != 2 {
		t.Errorf("Sequence = %d, want 2", k.Sequence())
	}
}

// TestStreamGracefulShutdown asserts the read goroutine exits on cancellation
// without waiting for a socket timeout. If the cancellation hook did not close
// the connection, ReadMessage would block until the pong deadline and this
// test would time out.
func TestStreamGracefulShutdown(t *testing.T) {
	handlerDone := make(chan struct{})

	k := newTestKite(t, instrumentHandler(t, nil))
	attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, _ int) {
		defer close(handlerDone)
		_ = ws.readSubscribe(t, conn)
		_ = conn.WriteMessage(websocket.BinaryMessage,
			encodeTickFrame(relianceToken, time.Now(), 2934.5, 1, [4]float64{}, 1, 1))
		// Stay on the socket until the client hangs up.
		ws.record(t, conn)
	})

	// A long pong timeout: shutdown must not depend on it.
	k.pongTimeout = time.Hour

	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan Tick, 1)
	done := make(chan error, 1)
	go func() { done <- k.Stream(ctx, nil, out) }()

	select {
	case tick := <-out:
		if tick.Symbol != "RELIANCE" {
			t.Fatalf("tick = %+v, want RELIANCE", tick)
		}
	case <-time.After(streamTimeout):
		t.Fatal("timed out waiting for the first tick")
	}

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream = %v, want context.Canceled", err)
		}
	case <-time.After(streamTimeout):
		t.Fatal("Stream did not return after the context was cancelled")
	}

	// The server side must see the hang-up too, which proves the client
	// closed the socket rather than abandoning it.
	select {
	case <-handlerDone:
	case <-time.After(streamTimeout):
		t.Fatal("the server handler never observed the client disconnecting")
	}
}

func TestStreamSequenceIsMonotonicAcrossReconnects(t *testing.T) {
	ts := time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC)

	// Each connection sends one tick and then drops, forcing a reconnect.
	// The gate keeps the test deterministic: the test releases exactly one
	// tick at a time, so the sequence it observes cannot race ahead.
	gate := make(chan struct{})

	k := newTestKite(t, instrumentHandler(t, nil))
	ws := attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, n int) {
		_ = ws.readSubscribe(t, conn)
		if n > 1 {
			<-gate
		}
		_ = conn.WriteMessage(websocket.BinaryMessage,
			encodeTickFrame(relianceToken, ts.Add(time.Duration(n)*time.Second), float64(100+n), 1, [4]float64{}, int64(n), int64(n)))
		// Returning drops the socket, which is what forces the reconnect.
	})
	k.reconnectBase = 10 * time.Millisecond
	k.maxBackoff = 20 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := make(chan Tick, 4)

	done := make(chan error, 1)
	go func() { done <- k.Stream(ctx, nil, out) }()

	for i := 0; i < 3; i++ {
		select {
		case tick := <-out:
			if want := float64(101 + i); tick.Price != want {
				t.Errorf("tick %d price = %v, want %v", i, tick.Price, want)
			}
			// The sequence must be exactly one per delivered tick and must
			// not restart when the socket is re-established.
			if got, want := k.Sequence(), uint64(i+1); got != want {
				t.Errorf("after tick %d Sequence = %d, want %d", i, got, want)
			}
		case <-time.After(streamTimeout):
			t.Fatalf("timed out waiting for tick %d", i)
		}
		select {
		case gate <- struct{}{}:
		case <-time.After(streamTimeout):
			t.Fatalf("timed out releasing tick %d", i+1)
		}
	}

	if got := k.Reconnects(); got < 2 {
		t.Errorf("Reconnects = %d, want at least 2", got)
	}
	if got := ws.Connections(); got < 3 {
		t.Errorf("server saw %d connections, want at least 3", got)
	}

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream = %v, want context.Canceled", err)
		}
	case <-time.After(streamTimeout):
		t.Fatal("Stream did not return after the context was cancelled")
	}
}

func TestStreamAnswersKitePing(t *testing.T) {
	// A control pong is dispatched to the handler during a read, so the
	// server must keep reading after the data frame to observe it. The read
	// loop therefore runs on its own goroutine and the handler waits on the
	// two signals.
	jsonPong := make(chan struct{}, 1)
	controlPong := make(chan struct{}, 1)
	bothAnswered := make(chan struct{})

	k := newTestKite(t, instrumentHandler(t, nil))
	ws := attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, _ int) {
		_ = ws.readSubscribe(t, conn)
		conn.SetPongHandler(func(string) error {
			select {
			case controlPong <- struct{}{}:
			default:
			}
			return nil
		})

		// The JSON keepalive Kite sends every ~30 seconds.
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"a":["ping",1788000000]}`)); err != nil {
			t.Errorf("write json ping: %v", err)
			return
		}
		// The RFC 6455 control ping. Kite does not send this, but a
		// reconnecting client meets proxies that do.
		if err := conn.WriteMessage(websocket.PingMessage, []byte("hb")); err != nil {
			t.Errorf("write control ping: %v", err)
			return
		}

		go func() {
			defer conn.Close()
			_ = conn.SetReadDeadline(time.Now().Add(streamTimeout))
			for {
				_, data, err := conn.ReadMessage()
				if err != nil {
					return
				}
				var cmd wsCommand
				if err := json.Unmarshal(data, &cmd); err != nil {
					continue
				}
				ws.note(cmd)
				if cmd.Action == "pong" {
					select {
					case jsonPong <- struct{}{}:
					default:
					}
				}
			}
		}()

		for i := 0; i < 2; i++ {
			select {
			case <-jsonPong:
			case <-controlPong:
			case <-time.After(streamTimeout):
				close(bothAnswered)
				return
			}
		}
		close(bothAnswered)
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- k.Stream(ctx, nil, make(chan Tick, 1)) }()

	select {
	case <-bothAnswered:
	case <-time.After(streamTimeout):
		t.Fatal("the client did not answer both pings")
	}

	if pongs := ws.Pongs(); len(pongs) != 1 || pongs[0] != "1788000000" {
		t.Errorf("pongs = %v, want the ping timestamp echoed back", pongs)
	}

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream = %v, want context.Canceled", err)
		}
	case <-time.After(streamTimeout):
		t.Fatal("Stream did not return after the context was cancelled")
	}
}

func TestStreamReconnectsWithCappedBackoff(t *testing.T) {
	k := newTestKite(t, instrumentHandler(t, nil))
	ws := attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, _ int) {
		// Drop the socket immediately; the streamer must back off and retry.
		_ = ws.readSubscribe(t, conn)
	})
	k.reconnectBase = 40 * time.Millisecond
	k.maxBackoff = 80 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- k.Stream(ctx, nil, make(chan Tick, 1)) }()

	// Delays run 40, 80, 80, 80ms, so 400ms allows at most ten attempts. The
	// bound is loose on purpose: it asserts the backoff is honoured without
	// making the test timing-sensitive.
	time.Sleep(400 * time.Millisecond)
	if got := ws.Connections(); got > 10 {
		t.Errorf("server saw %d connections in 400ms, backoff is not being honoured", got)
	}
	if got := k.Reconnects(); got < 1 {
		t.Errorf("Reconnects = %d, want the streamer to have retried", got)
	}

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream = %v, want context.Canceled", err)
		}
	case <-time.After(streamTimeout):
		t.Fatal("Stream did not return after the context was cancelled")
	}
}

func TestStreamRetriesUntilContextEnds(t *testing.T) {
	// Port 1 on the loopback interface refuses connections, so the dial
	// fails without the test ever leaving the machine.
	k := newTestKite(t, instrumentHandler(t, nil))
	k.setEndpoints(k.endpoints.httpBase, "ws://127.0.0.1:1")

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	err := k.Stream(ctx, nil, make(chan Tick, 1))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Stream = %v, want context.DeadlineExceeded", err)
	}
	if k.Reconnects() < 1 {
		t.Errorf("Reconnects = %d, want the streamer to have retried", k.Reconnects())
	}
}

func TestStreamValidation(t *testing.T) {
	k := newTestKite(t, instrumentHandler(t, nil))
	k.setEndpoints(k.endpoints.httpBase, "ws://127.0.0.1:1")

	if err := k.Stream(context.Background(), nil, nil); err == nil {
		t.Error("Stream with a nil channel: want an error")
	}
	if err := k.Stream(context.Background(), []string{"MISSPELT"}, make(chan Tick, 1)); !errors.Is(err, ErrInstrumentNotFound) {
		t.Errorf("Stream with an unknown symbol: error = %v, want ErrInstrumentNotFound", err)
	}

	empty, err := NewKiteFromConfig(config.BrokerConfig{AccessToken: testAccessToken})
	if err != nil {
		t.Fatalf("NewKiteFromConfig: %v", err)
	}
	if err := empty.Stream(context.Background(), nil, make(chan Tick, 1)); err == nil {
		t.Error("Stream with an empty watchlist: want an error")
	}

	noToken, err := NewKiteFromConfig(config.BrokerConfig{Instruments: []string{"RELIANCE"}})
	if err != nil {
		t.Fatalf("NewKiteFromConfig: %v", err)
	}
	if err := noToken.Stream(context.Background(), nil, make(chan Tick, 1)); !errors.Is(err, ErrAuth) {
		t.Errorf("Stream without a token: error = %v, want ErrAuth", err)
	}
}

func TestBackoffFor(t *testing.T) {
	tests := []struct {
		name    string
		attempt int
		base    time.Duration
		max     time.Duration
		want    time.Duration
	}{
		{name: "first attempt", attempt: 0, base: time.Second, max: time.Minute, want: time.Second},
		{name: "second attempt", attempt: 1, base: time.Second, max: time.Minute, want: 2 * time.Second},
		{name: "third attempt", attempt: 2, base: time.Second, max: time.Minute, want: 4 * time.Second},
		{name: "saturates at max", attempt: 10, base: time.Second, max: 5 * time.Second, want: 5 * time.Second},
		{name: "base already at max", attempt: 3, base: 5 * time.Second, max: 5 * time.Second, want: 5 * time.Second},
		{name: "huge attempt does not overflow", attempt: 200, base: time.Second, max: time.Minute, want: time.Minute},
		{name: "negative attempt", attempt: -5, base: time.Second, max: time.Minute, want: time.Second},
		{name: "zero base uses default", attempt: 0, base: 0, max: time.Minute, want: defaultReconnectBase},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := backoffFor(tc.attempt, tc.base, tc.max); got != tc.want {
				t.Errorf("backoffFor(%d, %v, %v) = %v, want %v", tc.attempt, tc.base, tc.max, got, tc.want)
			}
		})
	}
}
