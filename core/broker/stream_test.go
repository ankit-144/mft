package broker

import (
	"context"
	"encoding/binary"
	"errors"
	"net/url"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
)

func symbolMap() map[int64]contracts.Instrument {
	return map[int64]contracts.Instrument{
		relianceToken: {Token: relianceToken, Symbol: "RELIANCE", Exchange: "NSE"},
		tcsToken:      {Token: tcsToken, Symbol: "TCS", Exchange: "NSE"},
	}
}

// kiteFullPacketFixture lays out fields according to Kite's published packet
// table, independently of the production decoder.
func kiteFullPacketFixture(token int64, pricePaise int32, volume uint32, exchangeTS time.Time) []byte {
	packet := make([]byte, 184)
	binary.BigEndian.PutUint32(packet[0:4], uint32(token))
	binary.BigEndian.PutUint32(packet[4:8], uint32(pricePaise))
	binary.BigEndian.PutUint32(packet[8:12], 7)
	binary.BigEndian.PutUint32(packet[12:16], uint32(pricePaise))
	binary.BigEndian.PutUint32(packet[16:20], volume)
	binary.BigEndian.PutUint32(packet[28:32], uint32(pricePaise))
	binary.BigEndian.PutUint32(packet[32:36], uint32(pricePaise))
	binary.BigEndian.PutUint32(packet[36:40], uint32(pricePaise))
	binary.BigEndian.PutUint32(packet[40:44], uint32(pricePaise))
	binary.BigEndian.PutUint32(packet[lastTradeTimeOff:lastTradeTimeOff+4], uint32(exchangeTS.Unix()))
	binary.BigEndian.PutUint32(packet[exchangeTimeOff:exchangeTimeOff+4], uint32(exchangeTS.Unix()))
	return packet
}

func kiteMessageFixture(packets ...[]byte) []byte {
	message := make([]byte, 2)
	binary.BigEndian.PutUint16(message, uint16(len(packets)))
	for _, packet := range packets {
		length := []byte{byte(len(packet) >> 8), byte(len(packet))}
		message = append(message, length...)
		message = append(message, packet...)
	}
	return message
}

func TestStreamURLUsesBothKiteCredentials(t *testing.T) {
	endpoint, err := url.Parse((endpoints{wsBase: "wss://ws.kite.trade"}).streamURL("key value", "access/token"))
	if err != nil {
		t.Fatal(err)
	}
	query := endpoint.Query()
	if query.Get("api_key") != "key value" || query.Get("access_token") != "access/token" {
		t.Fatalf("query = %v, want both credentials", query)
	}
}

func TestDecodeKiteFullPacketAndMultiPacketFraming(t *testing.T) {
	ts := time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC)
	message := kiteMessageFixture(
		kiteFullPacketFixture(relianceToken, 293450, 555, ts),
		kiteFullPacketFixture(tcsToken, 410025, 42, ts.Add(time.Second)),
	)
	ticks, err := decodeMarketMessage(message, symbolMap(), ts.Add(2*time.Second))
	if err != nil {
		t.Fatalf("decodeMarketMessage: %v", err)
	}
	want := []Tick{
		{Symbol: "RELIANCE", Token: relianceToken, Price: 2934.5, Volume: 555, Timestamp: ts},
		{Symbol: "TCS", Token: tcsToken, Price: 4100.25, Volume: 42, Timestamp: ts.Add(time.Second)},
	}
	if len(ticks) != len(want) {
		t.Fatalf("got %d ticks, want %d: %+v", len(ticks), len(want), ticks)
	}
	for i := range want {
		if ticks[i] != want[i] {
			t.Errorf("tick %d = %+v, want %+v", i, ticks[i], want[i])
		}
	}
}

func TestDecodeKiteHeartbeatAndInvalidFraming(t *testing.T) {
	if ticks, err := decodeMarketMessage([]byte{0}, symbolMap(), time.Now()); err != nil || len(ticks) != 0 {
		t.Fatalf("heartbeat = %v, %v; want no ticks and no error", ticks, err)
	}
	if _, err := decodeMarketMessage([]byte{0, 1, 0, 8}, symbolMap(), time.Now()); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("truncated frame error = %v, want ErrUnavailable", err)
	}
}

func TestStreamDeliversTicksAndSubscribes(t *testing.T) {
	ts := time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC)
	frames := [][]byte{
		kiteMessageFixture(kiteFullPacketFixture(relianceToken, 293450, 100, ts)),
		kiteMessageFixture(kiteFullPacketFixture(tcsToken, 410025, 200, ts.Add(time.Second))),
	}

	var subscribed []int64
	k := newTestKite(t, instrumentHandler(t, nil))
	ws := attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, _ int) {
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

	want := []int64{738560, 3419705}
	if len(subscribed) != len(want) {
		t.Fatalf("subscribed to %v, want %v", subscribed, want)
	}
	for i := range want {
		if subscribed[i] != want[i] {
			t.Errorf("subscription %d = %d, want %d", i, subscribed[i], want[i])
		}
	}
	if modes := ws.Modes(); len(modes) != 1 || string(modes[0]) != `["full",[738560,3419705]]` {
		t.Errorf("mode payloads = %s, want [\"full\",[738560,3419705]]", modes)
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
			kiteMessageFixture(kiteFullPacketFixture(relianceToken, 293450, 1, time.Now())))
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
			kiteMessageFixture(kiteFullPacketFixture(relianceToken, int32((100+n)*100), uint32(n), ts.Add(time.Duration(n)*time.Second))))
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

func TestStreamIgnoresTextHeartbeatAndAnswersControlPing(t *testing.T) {
	controlPong := make(chan struct{}, 1)
	serverDone := make(chan struct{})

	k := newTestKite(t, instrumentHandler(t, nil))
	attachWS(t, k, func(t *testing.T, ws *wsTestServer, conn *websocket.Conn, _ int) {
		_ = ws.readSubscribe(t, conn)
		conn.SetPongHandler(func(string) error {
			select {
			case controlPong <- struct{}{}:
			default:
			}
			return nil
		})
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"message","data":"notice"}`))
		_ = conn.WriteMessage(websocket.PingMessage, []byte("hb"))
		_ = conn.SetReadDeadline(time.Now().Add(streamTimeout))
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				close(serverDone)
				return
			}
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- k.Stream(ctx, nil, make(chan Tick, 1)) }()

	select {
	case <-controlPong:
	case <-time.After(streamTimeout):
		t.Fatal("WebSocket control ping was not answered")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Stream = %v, want context.Canceled", err)
		}
	case <-time.After(streamTimeout):
		t.Fatal("Stream did not stop")
	}
	if wsPongs := k.Delivered(); wsPongs != 0 {
		t.Fatalf("delivered %d ticks from text heartbeat", wsPongs)
	}
	select {
	case <-serverDone:
	case <-time.After(streamTimeout):
		t.Fatal("server did not observe client close")
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
