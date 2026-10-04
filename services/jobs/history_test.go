package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/testutil"
)

// fakeResolver stands in for C1's *broker.Kite. It is a one-method fake
// precisely because the seam is one method wide.
type fakeResolver struct {
	instruments []contracts.Instrument
	err         error
	calls       int
}

// Instruments implements InstrumentResolver.
func (f *fakeResolver) Instruments(context.Context) ([]contracts.Instrument, error) {
	f.calls++
	return f.instruments, f.err
}

// masterResolver is the instrument master the history fixtures resolve against.
func masterResolver() *fakeResolver {
	return &fakeResolver{instruments: []contracts.Instrument{
		{Token: 256265, Symbol: "RELIANCE", Exchange: "NSE", LotSize: 1},
		{Token: 11536, Symbol: "TCS", Exchange: "NSE", LotSize: 1},
		{Token: 15909, Symbol: "INFY", Exchange: "NSE", LotSize: 1},
	}}
}

// historyFixture wires a KiteHistory to a fakeKite over a loopback server.
type historyFixture struct {
	hist *KiteHistory
	fake *fakeKite
}

// newHistoryFixture starts a fake Kite and returns a client pointed at it.
func newHistoryFixture(t *testing.T, fake *fakeKite) historyFixture {
	t.Helper()

	cfg := &config.Config{}
	cfg.Broker.APIKey = "test-key"
	cfg.Broker.AccessToken = "test-token"
	cfg.Broker.RequestTimeoutSeconds = 5
	cfg.Broker.Instruments = []string{"RELIANCE", "TCS"}

	return historyFixture{
		hist: newKiteHistory(cfg, masterResolver(), startFakeKite(t, fake), testutil.NewLogger()),
		fake: fake,
	}
}

// from and to are the bounds used by the history tests.
var (
	from = time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	to   = time.Date(2024, 1, 8, 0, 0, 0, 0, time.UTC)
)

// kites returns a fake Kite serving one good row.
func kites() *fakeKite {
	return &fakeKite{defaultReply: reply{rows: [][]any{
		{"2024-01-02T09:15:00+05:30", "2543.0", "2549.0", "2535.0", "2545.5", 123456.0},
		{"2024-01-02T09:16:00+05:30", "2545.5", "2551.25", "2540.0", "2548.0", 98765.0},
	}}}
}

func TestKiteHistoryParsesCandles(t *testing.T) {
	fx := newHistoryFixture(t, kites())

	candles, err := fx.hist.HistoricalCandles(context.Background(), "reliance", from, to, "minute")
	if err != nil {
		t.Fatalf("HistoricalCandles() error = %v", err)
	}
	if len(candles) != 2 {
		t.Fatalf("got %d candles, want 2", len(candles))
	}

	first := candles[0]
	if first.Symbol != "RELIANCE" {
		t.Errorf("Symbol = %q, want the normalised RELIANCE", first.Symbol)
	}
	if want := time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC); !first.Timestamp.Equal(want) {
		t.Errorf("Timestamp = %s, want %s (09:15 IST in UTC)", first.Timestamp, want)
	}
	if first.Open != 2543.0 || first.High != 2549.0 || first.Low != 2535.0 || first.Close != 2545.5 {
		t.Errorf("OHLC = %v/%v/%v/%v, want 2543/2549/2535/2545.5", first.Open, first.High, first.Low, first.Close)
	}
	if first.Volume != 123456 {
		t.Errorf("Volume = %d, want 123456", first.Volume)
	}
	if got := first.Timestamp.Location(); got != time.UTC {
		t.Errorf("Timestamp location = %v, want UTC", got)
	}
}

func TestKiteHistoryUsesHalfOpenRangeAndParsesKiteOffset(t *testing.T) {
	fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{rows: [][]any{
		{"2024-01-02T09:14:00+0530", 1, 2, 0.5, 1.5, 1},
		{"2024-01-02T09:15:00+0530", 1, 2, 0.5, 1.5, 2},
		{"2024-01-02T09:16:00+0530", 1, 2, 0.5, 1.5, 3},
		{"2024-01-02T09:17:00+0530", 1, 2, 0.5, 1.5, 4},
	}}})
	start := time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC)
	end := start.Add(2 * time.Minute)
	got, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", start, end, "minute")
	if err != nil {
		t.Fatalf("HistoricalCandles() error = %v", err)
	}
	if len(got) != 2 || got[0].Volume != 2 || got[1].Volume != 3 {
		t.Fatalf("candles = %+v, want the two rows in [%s,%s)", got, start, end)
	}
}

func TestKiteHistoryProtocolRequestAndRangeAreSocketFree(t *testing.T) {
	inst := contracts.Instrument{Token: 256265, Symbol: "RELIANCE", Exchange: "NSE"}
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := start.Add(24 * time.Hour)
	request, err := url.Parse(historyRequestPath(inst, start, end, "minute"))
	if err != nil {
		t.Fatal(err)
	}
	if request.Path != "/instruments/historical/256265/minute" {
		t.Fatalf("path = %q", request.Path)
	}
	query := request.Query()
	if query.Get("from") != "2024-01-01 05:30:00" || query.Get("to") != "2024-01-02 05:30:00" || query.Get("oi") != "0" {
		t.Fatalf("query = %v", query)
	}
	candles := []contracts.Candle{
		{Timestamp: start.Add(-time.Minute)},
		{Timestamp: start},
		{Timestamp: end.Add(-time.Minute)},
		{Timestamp: end},
	}
	filtered := historicalRange(candles, start, end)
	if len(filtered) != 2 || !filtered[0].Timestamp.Equal(start) || !filtered[1].Timestamp.Equal(end.Add(-time.Minute)) {
		t.Fatalf("range filter = %+v", filtered)
	}
	row := json.RawMessage(`[` + `"2024-01-02T09:15:00+0530",1,2,0.5,1.5,10]`)
	candle, err := decodeHistoryRow("RELIANCE", row)
	if err != nil || !candle.Timestamp.Equal(time.Date(2024, 1, 2, 3, 45, 0, 0, time.UTC)) {
		t.Fatalf("decode Kite +0530 row = %+v, %v", candle, err)
	}
}

func TestKiteHistoryBuildsTheKiteURL(t *testing.T) {
	fx := newHistoryFixture(t, kites())

	if _, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute"); err != nil {
		t.Fatalf("HistoricalCandles() error = %v", err)
	}

	want := "/instruments/historical/256265/minute"
	if got := fx.fake.lastPath(); got != want {
		t.Errorf("path = %q, want %q", got, want)
	}
	query, err := url.ParseQuery(fx.fake.lastQuery())
	if err != nil {
		t.Fatalf("parse query: %v", err)
	}
	for field, want := range map[string]string{"from": "2024-01-01 05:30:00", "to": "2024-01-08 05:30:00", "continuous": "0", "oi": "0"} {
		if got := query.Get(field); got != want {
			t.Errorf("query %s = %q, want %q", field, got, want)
		}
	}
	if want := "token test-key:test-token"; fx.fake.lastAuth() != want {
		t.Errorf("Authorization = %q, want %q", fx.fake.lastAuth(), want)
	}
}

func TestKiteHistoryWithoutCredentialsFailsBeforeAnyRequest(t *testing.T) {
	fx := newHistoryFixture(t, kites())
	fx.hist.accessToken = ""

	_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if !errors.Is(err, broker.ErrAuth) {
		t.Fatalf("error = %v, want it to wrap broker.ErrAuth", err)
	}
	if n := fx.fake.count(); n != 0 {
		t.Fatalf("made %d requests, want 0: an unauthenticated client must not call out", n)
	}
}

func TestKiteHistoryMapsThrottlingToErrRateLimit(t *testing.T) {
	for _, body := range []string{
		`{"status":"error","errors":[{"error_code":"429","message":"Too many requests"}]}`,
		`{"status":"error","errors":[{"error_code":"bad_request","message":"Too Many Requests in a second"}]}`,
	} {
		fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{status: http.StatusTooManyRequests, body: body}})

		_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
		if !errors.Is(err, broker.ErrRateLimit) {
			t.Fatalf("body %s: error = %v, want it to wrap broker.ErrRateLimit", body, err)
		}
	}
}

func TestKiteHistoryMapsServerErrorsToErrUnavailable(t *testing.T) {
	// A 5xx is transport, not throttling: the job retries it on the backoff
	// schedule rather than believing Kite asked it to slow down.
	fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{status: http.StatusBadGateway}})

	_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if !errors.Is(err, broker.ErrUnavailable) {
		t.Fatalf("error = %v, want it to wrap broker.ErrUnavailable", err)
	}
}

func TestKiteHistoryMapsAuthenticationFailuresToErrAuth(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{status: status}})

		_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
		if !errors.Is(err, broker.ErrAuth) {
			t.Fatalf("status %d: error = %v, want it to wrap broker.ErrAuth", status, err)
		}
	}
}

func TestKiteHistoryMapsAnUnknownRouteToErrInstrumentNotFound(t *testing.T) {
	// A 404 on the historical route means Kite does not know the exchange or
	// symbol. Reporting broker.ErrOrderNotFound here would be a lie, and
	// ErrInstrumentNotFound is what lets the job stop retrying the symbol.
	fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{status: http.StatusNotFound}})

	_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if !errors.Is(err, broker.ErrInstrumentNotFound) {
		t.Fatalf("error = %v, want it to wrap broker.ErrInstrumentNotFound", err)
	}
	if errors.Is(err, broker.ErrOrderNotFound) {
		t.Fatal("a 404 on the historical route was reported as an order-not-found error")
	}
}

func TestKiteHistoryReadsErrorEnvelopeServedWith200(t *testing.T) {
	// Kite answers with a 2xx and an error envelope when a token expires
	// mid-session, so the status line alone cannot classify the failure.
	body := `{"status":"error","errors":[{"error_code":"invalid_token","message":"Session expired"}]}`
	fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{body: body}})

	_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if !errors.Is(err, broker.ErrAuth) {
		t.Fatalf("error = %v, want it to wrap broker.ErrAuth", err)
	}
}

func TestKiteHistoryAcceptsUppercaseStatus(t *testing.T) {
	body := `{"status":"SUCCESS","data":{"candles":[["2024-01-02T09:15:00+05:30",1,2,0.5,1.5,10]]}}`
	fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{body: body}})

	candles, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if err != nil {
		t.Fatalf("HistoricalCandles() error = %v", err)
	}
	if len(candles) != 1 {
		t.Fatalf("got %d candles, want 1", len(candles))
	}
}

func TestKiteHistoryUnknownSymbolIsNotFound(t *testing.T) {
	fx := newHistoryFixture(t, kites())

	_, err := fx.hist.HistoricalCandles(context.Background(), "NOSUCH", from, to, "minute")
	if !errors.Is(err, broker.ErrInstrumentNotFound) {
		t.Fatalf("error = %v, want it to wrap broker.ErrInstrumentNotFound", err)
	}
}

func TestKiteHistoryPropagatesResolverFailure(t *testing.T) {
	fx := newHistoryFixture(t, kites())
	fx.hist.resolver = &fakeResolver{err: broker.ErrAuth}

	_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if !errors.Is(err, broker.ErrAuth) {
		t.Fatalf("error = %v, want the resolver's broker.ErrAuth to survive", err)
	}
}

func TestKiteHistoryWithoutAResolverIsNotFound(t *testing.T) {
	fx := newHistoryFixture(t, kites())
	fx.hist.resolver = nil

	_, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if !errors.Is(err, broker.ErrInstrumentNotFound) {
		t.Fatalf("error = %v, want it to wrap broker.ErrInstrumentNotFound", err)
	}
}

func TestKiteHistorySkipsMalformedRowsAndKeepsTheRest(t *testing.T) {
	fake := &fakeKite{defaultReply: reply{rows: [][]any{
		{"2024-01-02T09:15:00+05:30", "2543.0", "2549.0", "2535.0", "2545.5", 123456.0},
		{"not-a-timestamp", "1", "2", "0.5", "1.5", "10"},
		{"2024-01-02T09:16:00+05:30"},
		{"2024-01-02T09:17:00+05:30", "1", "2", "0.5", "1.5", "10"},
	}}}
	fx := newHistoryFixture(t, fake)

	candles, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute")
	if err != nil {
		t.Fatalf("HistoricalCandles() error = %v", err)
	}
	if len(candles) != 2 {
		t.Fatalf("got %d candles, want the 2 well-formed rows", len(candles))
	}
}

func TestKiteHistoryRejectsAnEmptyRange(t *testing.T) {
	fx := newHistoryFixture(t, kites())

	if _, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", to, from, "minute"); err == nil {
		t.Fatal("HistoricalCandles() with from after to returned nil, want an error")
	}
	if n := fx.fake.count(); n != 0 {
		t.Fatalf("made %d requests for an empty range, want 0", n)
	}
}

func TestKiteHistoryRejectsABlankInterval(t *testing.T) {
	fx := newHistoryFixture(t, kites())

	if _, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "  "); err == nil {
		t.Fatal("HistoricalCandles() with a blank interval returned nil, want an error")
	}
}

func TestKiteHistoryRejectsAnUnparseableBody(t *testing.T) {
	fx := newHistoryFixture(t, &fakeKite{defaultReply: reply{body: "<html>502</html>"}})

	if _, err := fx.hist.HistoricalCandles(context.Background(), "RELIANCE", from, to, "minute"); err == nil {
		t.Fatal("HistoricalCandles() on a non-JSON body returned nil, want an error")
	}
}

func TestKiteHistoryHonoursACancelledContext(t *testing.T) {
	fx := newHistoryFixture(t, kites())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := fx.hist.HistoricalCandles(ctx, "RELIANCE", from, to, "minute"); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

func TestKiteHistoryDefaultsToTheProductionEndpoint(t *testing.T) {
	h := NewKiteHistory(&config.Config{}, masterResolver(), testutil.NewLogger())
	if h.base() != kiteAPIURL {
		t.Fatalf("base() = %q, want the production %q", h.base(), kiteAPIURL)
	}
	if h.client.Timeout != defaultHistoryTimeout {
		t.Fatalf("timeout = %v, want the %v default", h.client.Timeout, defaultHistoryTimeout)
	}
}

func TestKiteHistoryUsesTheConfiguredRequestTimeout(t *testing.T) {
	cfg := &config.Config{}
	cfg.Broker.RequestTimeoutSeconds = 3

	h := NewKiteHistory(cfg, masterResolver(), testutil.NewLogger())
	if h.client.Timeout != 3*time.Second {
		t.Fatalf("timeout = %v, want 3s", h.client.Timeout)
	}
}
