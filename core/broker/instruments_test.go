package broker

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
)

func TestInstrumentsMapsConfiguredSymbolsToTokens(t *testing.T) {
	var auth http.Header
	k := newTestKite(t, instrumentHandler(t, &auth))

	got, err := k.Instruments(context.Background())
	if err != nil {
		t.Fatalf("Instruments: %v", err)
	}

	want := []contracts.Instrument{
		{Token: 738560, Symbol: "RELIANCE", Exchange: "NSE", LotSize: 1},
		{Token: 3419705, Symbol: "TCS", Exchange: "NSE", LotSize: 1},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d instruments, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("instrument %d: got %+v, want %+v", i, got[i], want[i])
		}
	}

	if want := "token test-api-key:" + testAccessToken; auth.Get("Authorization") != want {
		t.Errorf("Authorization = %q, want %q", auth.Get("Authorization"), want)
	}
}

func TestInstrumentsCachesWithinTTL(t *testing.T) {
	var calls atomic.Int32
	k := newTestKite(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(instrumentDump))
	}))

	for i := 0; i < 3; i++ {
		if _, err := k.Instruments(context.Background()); err != nil {
			t.Fatalf("Instruments call %d: %v", i, err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("instrument dump fetched %d times, want 1", got)
	}

	k.instMu.Lock()
	k.loadedAt = k.loadedAt.Add(-2 * k.instrumentTTL)
	k.instMu.Unlock()

	if _, err := k.Instruments(context.Background()); err != nil {
		t.Fatalf("Instruments after TTL: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("instrument dump fetched %d times, want 2 after TTL expiry", got)
	}
}

func TestInstrumentsRejectsUnknownSymbol(t *testing.T) {
	k := newTestKite(t, instrumentHandler(t, nil))
	k.watchlist = append(k.watchlist, "MISSPELT")

	_, err := k.Instruments(context.Background())
	if err == nil {
		t.Fatal("Instruments: want error for a symbol absent from the dump")
	}
	if !errors.Is(err, ErrInstrumentNotFound) {
		t.Fatalf("error = %v, want it to wrap ErrInstrumentNotFound", err)
	}
	if !strings.Contains(err.Error(), "MISSPELT") {
		t.Errorf("error = %v, want it to name the missing symbol", err)
	}
}

func TestParseInstrumentDumpPrefersNSEOverDerivatives(t *testing.T) {
	got, err := parseInstrumentDump([]byte(instrumentDump), []string{"RELIANCE"})
	if err != nil {
		t.Fatalf("parseInstrumentDump: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d instruments, want 1: %+v", len(got), got)
	}
	// The dump lists RELIANCE on NSE, on BSE and as an option leg. The cash
	// NSE row must win, and the 900 strike call must not shadow it.
	want := contracts.Instrument{Token: 738560, Symbol: "RELIANCE", Exchange: "NSE", LotSize: 1}
	if got[0] != want {
		t.Errorf("got %+v, want %+v", got[0], want)
	}
}

func TestParseInstrumentDumpErrors(t *testing.T) {
	tests := []struct {
		name string
		dump string
		want error
	}{
		{
			name: "missing column",
			dump: "instrument_token,exchange,name\n1,NSE,RELIANCE\n",
			want: ErrUnavailable,
		},
		{
			name: "unparsable token",
			dump: "instrument_token,exchange,tradingsymbol,lot_size\nnot-a-number,NSE,RELIANCE,1\n",
			want: ErrInstrumentNotFound,
		},
		{
			name: "symbol absent",
			dump: "instrument_token,exchange,tradingsymbol,lot_size\n1,NSE,TCS,1\n",
			want: ErrInstrumentNotFound,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseInstrumentDump([]byte(tc.dump), []string{"RELIANCE"})
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want it to wrap %v", err, tc.want)
			}
		})
	}
}

func TestInstrumentLookup(t *testing.T) {
	k := newTestKite(t, instrumentHandler(t, nil))

	got, err := k.Instrument(context.Background(), "reliance")
	if err != nil {
		t.Fatalf("Instrument: %v", err)
	}
	if got.Token != 738560 {
		t.Errorf("Token = %d, want 738560", got.Token)
	}

	if _, err := k.Instrument(context.Background(), "NOPE"); !errors.Is(err, ErrInstrumentNotFound) {
		t.Fatalf("error = %v, want ErrInstrumentNotFound", err)
	}
}

func TestNewKiteFromConfigValidatesAndAppliesConfig(t *testing.T) {
	if _, err := NewKiteFromConfig(config.BrokerConfig{RequestTimeoutSeconds: -1}); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("negative request timeout: error = %v, want ErrInvalidOrder", err)
	}
	if _, err := NewKiteFromConfig(config.BrokerConfig{ReconnectMaxBackoffSecs: -5}); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("negative backoff: error = %v, want ErrInvalidOrder", err)
	}

	// A config with no credentials must still construct: services start
	// without Kite credentials in dev, and the failure is reported per call.
	k, err := NewKiteFromConfig(config.BrokerConfig{
		AccessToken:             "tok",
		Instruments:             []string{"TCS", "TCS", " reliance "},
		RequestTimeoutSeconds:   7,
		ReconnectMaxBackoffSecs: 42,
	})
	if err != nil {
		t.Fatalf("NewKiteFromConfig: %v", err)
	}
	if k.requestTimeout.Seconds() != 7 {
		t.Errorf("requestTimeout = %v, want 7s", k.requestTimeout)
	}
	if k.maxBackoff.Seconds() != 42 {
		t.Errorf("maxBackoff = %v, want 42s", k.maxBackoff)
	}
	if got, want := strings.Join(k.wantedSymbols(), ","), "TCS,RELIANCE"; got != want {
		t.Errorf("wantedSymbols = %q, want %q", got, want)
	}
	if k.endpoints.httpBase != defaultHTTPBaseURL || k.endpoints.wsBase != defaultWSBaseURL {
		t.Errorf("endpoints = %+v, want the production Kite hosts", k.endpoints)
	}
}

func TestSetProduct(t *testing.T) {
	k := NewKite()
	if k.product != defaultProduct {
		t.Errorf("default product = %q, want %q", k.product, defaultProduct)
	}
	if err := k.SetProduct("mis"); err != nil {
		t.Fatalf("SetProduct: %v", err)
	}
	if k.product != "MIS" {
		t.Errorf("product = %q, want MIS", k.product)
	}
	if err := k.SetProduct("SOMEDAY"); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("SetProduct: error = %v, want ErrInvalidOrder", err)
	}
}

func TestRequestsWithoutTokenFailAsAuth(t *testing.T) {
	srv := localTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("no request should reach the server without an access token")
	}))

	k := NewKite()
	k.setEndpoints(srv.URL, "ws://unused")

	if _, err := k.Instruments(context.Background()); !errors.Is(err, ErrAuth) {
		t.Errorf("Instruments: error = %v, want ErrAuth", err)
	}
	if _, err := k.PlaceOrder(context.Background(), "RELIANCE", "BUY", 1, 0); !errors.Is(err, ErrAuth) {
		t.Errorf("PlaceOrder: error = %v, want ErrAuth", err)
	}
	if err := k.CancelOrder(context.Background(), "1"); !errors.Is(err, ErrAuth) {
		t.Errorf("CancelOrder: error = %v, want ErrAuth", err)
	}
	if _, err := k.GetPositions(context.Background()); !errors.Is(err, ErrAuth) {
		t.Errorf("GetPositions: error = %v, want ErrAuth", err)
	}
}
