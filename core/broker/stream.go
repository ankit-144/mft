package broker

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// tickFrameLen is the size of a Kite binary LTP frame.
//
// The frame is two big-endian structs:
//
//	struct 1: instrument_token uint32, exchange_timestamp uint32      (8 bytes)
//	struct 2: last_price float64, last_quantity int32,
//	         ohlc [4]float64, total_traded_volume int64,
//	         total_traded_quantity int64                             (60 bytes)
const tickFrameLen = 8 + 60

// Byte offsets into a binary LTP frame.
const (
	offInstrumentToken = 0
	offTimestamp       = 4
	offLastPrice       = 8
	offLastQuantity    = 16
	offTotalVolume     = 52
	offTotalQuantity   = 60
)

// wsCommand is an outbound subscribe, unsubscribe or mode-change request, and
// the reply to a server ping.
type wsCommand struct {
	Action string   `json:"a"`
	Values []string `json:"v"`
}

// wsPing is the server-initiated control frame. Kite sends
// {"a":["ping",<timestamp>]} roughly every 30 seconds and expects
// {"a":"pong","v":["<timestamp>"]} in reply; dropping the reply is how a
// session gets closed.
type wsPing struct {
	Action []json.RawMessage `json:"a"`
}

// Stream subscribes to symbols and delivers ticks on out until ctx is
// cancelled, then returns ctx.Err().
//
// A nil or empty symbols slice means "the instruments declared in broker
// config". The socket is re-established with exponential backoff capped at
// broker.reconnect_max_backoff_seconds for as long as ctx is live. Sends on
// out are cancellable, so a consumer that stops reading cannot wedge the
// streamer.
func (k *Kite) Stream(ctx context.Context, symbols []string, out chan<- Tick) error {
	if out == nil {
		return errors.New("kite: stream: nil tick channel")
	}

	targets := symbols
	if len(targets) == 0 {
		targets = k.wantedSymbols()
	}
	if len(targets) == 0 {
		return errors.New("kite: stream: no symbols to subscribe to")
	}
	if err := k.checkAuth(); err != nil {
		return err
	}

	all, err := k.Instruments(ctx)
	if err != nil {
		return err
	}

	subs := make([]string, 0, len(targets))
	symbolByToken := make(map[int64]string, len(targets))
	for _, raw := range targets {
		inst, err := lookupToken(all, raw)
		if err != nil {
			return err
		}
		if _, dup := symbolByToken[inst.Token]; dup {
			continue
		}
		symbolByToken[inst.Token] = inst.Symbol
		subs = append(subs, fmt.Sprintf("%s|%s|%d", inst.Exchange, inst.Symbol, inst.Token))
	}

	attempt := 0
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sessionErr := k.session(ctx, subs, symbolByToken, out)
		if ctx.Err() != nil {
			return ctx.Err()
		}

		delay := backoffFor(attempt, k.reconnectBase, k.maxBackoff)
		attempt++
		k.reconnects.Add(1)
		k.log().Warn("kite stream reconnecting",
			zap.Int("attempt", attempt),
			zap.Duration("backoff", delay),
			zap.Error(sessionErr))

		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

// session runs one WebSocket connection to exhaustion: dial, subscribe,
// request full mode, then pump frames until the socket fails or ctx ends.
func (k *Kite) session(ctx context.Context, subs []string, symbolByToken map[int64]string, out chan<- Tick) error {
	conn, err := k.dial(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Closing the socket is the only way to unblock ReadMessage, so the
	// cancellation hook does that. Without it the read below would sit there
	// until the pong timeout expired.
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer stop()

	// gorilla forbids concurrent writers, and the pong replies below are
	// written from the same goroutine as the subscribe, so a single writer
	// lock covers every frame this client sends.
	var writeMu sync.Mutex
	writeJSON := func(v any) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		if err := conn.SetWriteDeadline(time.Now().Add(k.writeWait)); err != nil {
			return err
		}
		return conn.WriteJSON(v)
	}

	// A pong proves the peer is alive, so every pong buys another full pong
	// window. Any successful read is equally good evidence.
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(k.pongTimeout))
	})
	conn.SetPingHandler(func(appData string) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		if err := conn.SetWriteDeadline(time.Now().Add(k.writeWait)); err != nil {
			return err
		}
		return conn.WriteMessage(websocket.PongMessage, []byte(appData))
	})
	if err := conn.SetReadDeadline(time.Now().Add(k.pongTimeout)); err != nil {
		return fmt.Errorf("kite websocket: set read deadline: %w: %w", ErrUnavailable, err)
	}

	if err := writeJSON(wsCommand{Action: "subscribe", Values: subs}); err != nil {
		return fmt.Errorf("kite websocket: subscribe: %w: %w", ErrUnavailable, err)
	}
	if err := writeJSON(wsCommand{Action: "mode", Values: append([]string{"full"}, subs...)}); err != nil {
		return fmt.Errorf("kite websocket: set mode: %w: %w", ErrUnavailable, err)
	}
	k.log().Debug("kite stream subscribed", zap.Int("instruments", len(subs)))

	for {
		msgType, data, err := conn.ReadMessage()
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("kite websocket: read: %w: %w", ErrUnavailable, err)
		}
		if err := conn.SetReadDeadline(time.Now().Add(k.pongTimeout)); err != nil {
			return fmt.Errorf("kite websocket: extend read deadline: %w: %w", ErrUnavailable, err)
		}

		if msgType == websocket.BinaryMessage {
			tick, err := decodeTickFrame(data, symbolByToken)
			if err != nil {
				if errors.Is(err, ErrInstrumentNotFound) {
					k.log().Debug("kite stream: tick for unsubscribed token", zap.Error(err))
					continue
				}
				return err
			}
			if err := k.deliver(ctx, out, tick); err != nil {
				return err
			}
			continue
		}

		// Text frames carry the JSON control channel: acks and pings.
		var ping wsPing
		if err := json.Unmarshal(data, &ping); err != nil || len(ping.Action) == 0 {
			k.log().Debug("kite stream: unrecognised text frame")
			continue
		}
		if wsAction(ping.Action[0]) == "ping" {
			if err := writeJSON(pongReply(data)); err != nil {
				return fmt.Errorf("kite websocket: pong: %w: %w", ErrUnavailable, err)
			}
		}
	}
}

// dial opens the Kite WebSocket, carrying the access token as a query
// parameter as the protocol requires.
func (k *Kite) dial(ctx context.Context) (*websocket.Conn, error) {
	header := http.Header{}
	header.Set("User-Agent", "mft-broker/1.0")

	conn, resp, err := k.dialer.DialContext(ctx, k.endpoints.streamURL(k.accessToken), header)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf("kite websocket dial: http %d: %w: %w",
				resp.StatusCode, classifyStatus(resp.StatusCode, "", ""), err)
		}
		return nil, fmt.Errorf("kite websocket dial: %w: %w", ErrUnavailable, err)
	}
	return conn, nil
}

// wsAction decodes the leading action name out of a Kite control frame. The
// element is a json.RawMessage, so it still carries its quotes and must be
// decoded rather than compared as a string.
func wsAction(raw json.RawMessage) string {
	var action string
	if err := json.Unmarshal(raw, &action); err != nil {
		return strings.Trim(string(raw), `"`)
	}
	return action
}

// pongReply builds the JSON pong Kite expects in reply to its ping, echoing
// the timestamp back so the server can measure round-trip time.
func pongReply(pingFrame []byte) wsCommand {
	var ping wsPing
	if err := json.Unmarshal(pingFrame, &ping); err != nil || len(ping.Action) < 2 {
		return wsCommand{Action: "pong", Values: []string{""}}
	}
	var ts string
	if err := json.Unmarshal(ping.Action[1], &ts); err != nil {
		ts = strings.Trim(string(ping.Action[1]), `"`)
	}
	return wsCommand{Action: "pong", Values: []string{ts}}
}

// deliver stamps a monotonically increasing sequence number and hands the
// tick to the consumer, giving up when ctx ends.
func (k *Kite) deliver(ctx context.Context, out chan<- Tick, tick Tick) error {
	select {
	case out <- tick:
		k.seq.Add(1)
		k.delivered.Add(1)
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// decodeTickFrame translates one binary LTP frame into a contracts.Tick.
// A frame carrying a token outside the subscription is rejected with
// ErrInstrumentNotFound so the caller drops it rather than guessing a symbol.
func decodeTickFrame(frame []byte, symbolByToken map[int64]string) (Tick, error) {
	if len(frame) != tickFrameLen {
		return Tick{}, fmt.Errorf("kite tick frame: got %d bytes, want %d: %w", len(frame), tickFrameLen, ErrInvalidOrder)
	}

	token := int64(binary.BigEndian.Uint32(frame[offInstrumentToken : offInstrumentToken+4]))
	symbol, ok := symbolByToken[token]
	if !ok {
		return Tick{}, fmt.Errorf("kite tick frame: token %d is not subscribed: %w", token, ErrInstrumentNotFound)
	}

	// Kite sends cumulative day volume and a day timestamp, not per-tick
	// deltas, so the frame is forwarded as-is and the consumer aggregates.
	volume := int64(binary.BigEndian.Uint64(frame[offTotalVolume : offTotalVolume+8]))

	return Tick{
		Symbol:    symbol,
		Token:     token,
		Price:     math.Float64frombits(binary.BigEndian.Uint64(frame[offLastPrice : offLastPrice+8])),
		Volume:    volume,
		Timestamp: time.Unix(int64(binary.BigEndian.Uint32(frame[offTimestamp:offTimestamp+4])), 0).UTC(),
	}, nil
}

// encodeTickFrame builds a binary LTP frame. It is the exact inverse of
// decodeTickFrame and exists so the test server produces real wire bytes from
// one offset table rather than a second hand-written copy of the layout.
func encodeTickFrame(token int64, ts time.Time, price float64, lastQty int32, ohlc [4]float64, volume, quantity int64) []byte {
	frame := make([]byte, tickFrameLen)
	binary.BigEndian.PutUint32(frame[offInstrumentToken:], uint32(token))
	binary.BigEndian.PutUint32(frame[offTimestamp:], uint32(ts.Unix()))
	binary.BigEndian.PutUint64(frame[offLastPrice:], math.Float64bits(price))
	binary.BigEndian.PutUint32(frame[offLastQuantity:], uint32(lastQty))
	for i, v := range ohlc {
		binary.BigEndian.PutUint64(frame[offLastQuantity+4+i*8:], math.Float64bits(v))
	}
	binary.BigEndian.PutUint64(frame[offTotalVolume:], uint64(volume))
	binary.BigEndian.PutUint64(frame[offTotalQuantity:], uint64(quantity))
	return frame
}

// backoffFor returns the delay before reconnect attempt+1, doubling from base
// and saturating at max. The doubling is iterative rather than a shift, so a
// large attempt count cannot overflow the duration into a negative wait and
// turn into a hot reconnect loop.
func backoffFor(attempt int, base, max time.Duration) time.Duration {
	if base <= 0 {
		base = defaultReconnectBase
	}
	if max <= 0 {
		max = defaultMaxBackoff
	}
	if attempt < 0 {
		attempt = 0
	}

	delay := base
	for i := 0; i < attempt; i++ {
		if delay >= max {
			return max
		}
		delay *= 2
	}
	if delay > max || delay <= 0 {
		return max
	}
	return delay
}
