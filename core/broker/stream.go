package broker

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mft/core/contracts"
	"go.uber.org/zap"
)

// Kite full quote packet fields use network-order int32 values.
const (
	minLTPPacketLen   = 8
	quotePacketLen    = 44
	fullPacketLen     = 184
	fullVolumeOffset  = 16
	lastTradeTimeOff  = 44
	exchangeTimeOff   = 60
	maxMarketDataSize = 2 + (fullPacketLen+2)*3000
)

// wsCommand is an outbound subscribe, unsubscribe or mode-change request, and the reply
// to a server ping.
type wsCommand struct {
	Action string `json:"a"`
	Values any    `json:"v"`
}

// Stream subscribes to symbols and delivers ticks on out until ctx is cancelled, then
// returns ctx.Err().
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

	subs := make([]int64, 0, len(targets))
	instrumentByToken := make(map[int64]contracts.Instrument, len(targets))
	for _, raw := range targets {
		inst, err := lookupToken(all, raw)
		if err != nil {
			return err
		}
		if _, dup := instrumentByToken[inst.Token]; dup {
			continue
		}
		instrumentByToken[inst.Token] = inst
		subs = append(subs, inst.Token)
	}

	attempt := 0
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sessionErr := k.session(ctx, subs, instrumentByToken, out)
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

// session runs one WebSocket connection to exhaustion: dial, subscribe, request full
// mode, then pump frames until the socket fails or ctx ends.
func (k *Kite) session(ctx context.Context, subs []int64, instrumentByToken map[int64]contracts.Instrument, out chan<- Tick) error {
	conn, err := k.dial(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()

	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer stop()

	// gorilla forbids concurrent writers, and the pong replies below are written from the
	// same goroutine as the subscribe, so a single writer lock covers every frame this
	// client sends.
	var writeMu sync.Mutex
	writeJSON := func(v any) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		if err := conn.SetWriteDeadline(time.Now().Add(k.writeWait)); err != nil {
			return err
		}
		return conn.WriteJSON(v)
	}

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
	if err := writeJSON(wsCommand{Action: "mode", Values: []any{"full", subs}}); err != nil {
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
			ticks, err := decodeMarketMessage(data, instrumentByToken, time.Now())
			if err != nil {
				return err
			}
			for _, tick := range ticks {
				if err := k.deliver(ctx, out, tick); err != nil {
					return err
				}
			}
			continue
		}

		k.log().Debug("kite websocket text update", zap.Int("bytes", len(data)))
	}
}

// dial opens the Kite WebSocket, carrying the access token as a query parameter as the
// protocol requires.
func (k *Kite) dial(ctx context.Context) (*websocket.Conn, error) {
	header := http.Header{}
	header.Set("User-Agent", "mft-broker/1.0")

	conn, resp, err := k.dialer.DialContext(ctx, k.endpoints.streamURL(k.apiKey, k.accessToken), header)
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

// deliver stamps a monotonically increasing sequence number and hands the tick to the
// consumer, giving up when ctx ends.
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

// decodeMarketMessage unpacks the count/length framing and each quote packet.
func decodeMarketMessage(message []byte, instruments map[int64]contracts.Instrument, receivedAt time.Time) ([]Tick, error) {
	if len(message) == 1 {
		return nil, nil
	}
	if len(message) < 2 {
		return nil, fmt.Errorf("kite websocket: short market message: %w", ErrUnavailable)
	}
	count := int(binary.BigEndian.Uint16(message[:2]))
	if count == 0 || count > 3000 {
		return nil, fmt.Errorf("kite websocket: invalid packet count %d: %w", count, ErrUnavailable)
	}
	if len(message) > maxMarketDataSize {
		return nil, fmt.Errorf("kite websocket: market message too large: %d: %w", len(message), ErrUnavailable)
	}

	ticks := make([]Tick, 0, count)
	offset := 2
	for i := 0; i < count; i++ {
		if len(message)-offset < 2 {
			return nil, fmt.Errorf("kite websocket: missing packet %d length: %w", i, ErrUnavailable)
		}
		size := int(binary.BigEndian.Uint16(message[offset : offset+2]))
		offset += 2
		if size < minLTPPacketLen || size > len(message)-offset {
			return nil, fmt.Errorf("kite websocket: invalid packet %d size %d: %w", i, size, ErrUnavailable)
		}
		tick, ok, err := decodeQuotePacket(message[offset:offset+size], instruments, receivedAt)
		if err != nil {
			return nil, fmt.Errorf("kite websocket: packet %d: %w", i, err)
		}
		if ok {
			ticks = append(ticks, tick)
		}
		offset += size
	}
	if offset != len(message) {
		return nil, fmt.Errorf("kite websocket: %d trailing market bytes: %w", len(message)-offset, ErrUnavailable)
	}
	return ticks, nil
}

func decodeQuotePacket(packet []byte, instruments map[int64]contracts.Instrument, receivedAt time.Time) (Tick, bool, error) {
	token := int64(binary.BigEndian.Uint32(packet[:4]))
	inst, ok := instruments[token]
	if !ok {
		return Tick{}, false, nil
	}
	if len(packet) != minLTPPacketLen && len(packet) != quotePacketLen && len(packet) != fullPacketLen {
		return Tick{}, false, fmt.Errorf("unsupported quote packet size %d: %w", len(packet), ErrUnavailable)
	}
	price := float64(int32(binary.BigEndian.Uint32(packet[4:8])))
	if inst.Exchange == "CDS" || inst.Exchange == "BCD" {
		price /= 10_000_000
	} else {
		price /= 100
	}
	if math.IsNaN(price) || math.IsInf(price, 0) {
		return Tick{}, false, fmt.Errorf("invalid price: %w", ErrUnavailable)
	}
	ts := receivedAt.UTC()
	volume := int64(0)
	if len(packet) >= quotePacketLen {
		volume = int64(binary.BigEndian.Uint32(packet[fullVolumeOffset : fullVolumeOffset+4]))
	}
	if len(packet) == fullPacketLen {
		exchangeTS := binary.BigEndian.Uint32(packet[exchangeTimeOff : exchangeTimeOff+4])
		lastTradeTS := binary.BigEndian.Uint32(packet[lastTradeTimeOff : lastTradeTimeOff+4])
		if exchangeTS > 0 {
			ts = time.Unix(int64(exchangeTS), 0).UTC()
		} else if lastTradeTS > 0 {
			ts = time.Unix(int64(lastTradeTS), 0).UTC()
		}
	}
	return Tick{Symbol: inst.Symbol, Token: token, Price: price, Volume: volume, Timestamp: ts}, true, nil
}

// backoffFor returns the delay before reconnect attempt+1, doubling from base and
// saturating at max.
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
