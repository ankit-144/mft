package broker

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/contracts"
)

// orderServer is a local stand-in for the Kite REST API.
type orderServer struct {
	*httptest.Server

	placed  []url.Values
	cancels []string
}

func newOrderServer(t *testing.T, routes map[string]http.HandlerFunc) *orderServer {
	t.Helper()
	s := &orderServer{}

	// A single handler with an override table, rather than a ServeMux, so a
	// test can replace any endpoint without colliding with the defaults.
	handler := func(w http.ResponseWriter, r *http.Request) {
		for pattern, h := range routes {
			if strings.HasPrefix(r.URL.Path, pattern) {
				h(w, r)
				return
			}
		}
		switch {
		case r.URL.Path == "/instruments":
			_, _ = w.Write([]byte(instrumentDump))
		case r.URL.Path == "/orders/regular" && r.Method == httpPost:
			if err := r.ParseForm(); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			s.placed = append(s.placed, r.PostForm)
			reply(w, `{"status":"success","data":{"order_id":"4412"}}`)
		case strings.HasPrefix(r.URL.Path, "/orders/regular/") && r.Method == httpDelete:
			s.cancels = append(s.cancels, strings.TrimPrefix(r.URL.Path, "/orders/regular/"))
			reply(w, `{"status":"success","data":{"order_id":"4412"}}`)
		case r.URL.Path == "/portfolio/positions":
			reply(w, `{"status":"success","data":{"net":[
				{"tradingsymbol":"RELIANCE","instrument_token":738560,"exchange":"NSE","quantity":25,"average_price":2934.5,"product":"NRML"},
				{"tradingsymbol":"TCS","instrument_token":3419705,"exchange":"NSE","quantity":-10,"average_price":4100.25,"product":"NRML"}
			],"day":[]}}`)
		default:
			http.NotFound(w, r)
		}
	}

	s.Server = localTestServer(t, http.HandlerFunc(handler))
	return s
}

func reply(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(body))
}

func newOrderKite(t *testing.T, srv *orderServer) *Kite {
	t.Helper()
	k := newTestKite(t, instrumentHandler(t, nil))
	k.setEndpoints(srv.URL, "ws://unused")
	return k
}

func TestPlaceOrderMarket(t *testing.T) {
	srv := newOrderServer(t, nil)
	k := newOrderKite(t, srv)

	id, err := k.PlaceOrder(context.Background(), "reliance", "BUY", 10, 0)
	if err != nil {
		t.Fatalf("PlaceOrder: %v", err)
	}
	if id != "4412" {
		t.Errorf("order id = %q, want 4412", id)
	}
	if len(srv.placed) != 1 {
		t.Fatalf("server saw %d orders, want 1", len(srv.placed))
	}

	form := srv.placed[0]
	for field, want := range map[string]string{
		"tradingsymbol":    "RELIANCE",
		"exchange":         "NSE",
		"transaction_type": "BUY",
		"order_type":       "MARKET",
		"product":          "NRML",
		"quantity":         "10",
		"variety":          "regular",
	} {
		if got := form.Get(field); got != want {
			t.Errorf("form %s = %q, want %q", field, got, want)
		}
	}
	if form.Has("price") {
		t.Error("a market order must not send a price")
	}
}

func TestPlaceOrderLimitCarriesPrice(t *testing.T) {
	srv := newOrderServer(t, nil)
	k := newOrderKite(t, srv)

	if _, err := k.PlaceOrder(context.Background(), "TCS", "SELL", 3, 4100.25); err != nil {
		t.Fatalf("PlaceOrder: %v", err)
	}
	form := srv.placed[0]
	if got := form.Get("order_type"); got != "LIMIT" {
		t.Errorf("order_type = %q, want LIMIT", got)
	}
	if got := form.Get("price"); got != "4100.25" {
		t.Errorf("price = %q, want 4100.25", got)
	}
}

func TestPlaceOrderValidation(t *testing.T) {
	srv := newOrderServer(t, nil)
	k := newOrderKite(t, srv)
	ctx := context.Background()

	tests := []struct {
		name   string
		symbol string
		side   string
		qty    int
		price  float64
		want   error
	}{
		{name: "empty symbol", symbol: "", side: "BUY", qty: 1, want: ErrInvalidOrder},
		{name: "bad side", symbol: "RELIANCE", side: "HOLD", qty: 1, want: ErrInvalidOrder},
		{name: "zero quantity", symbol: "RELIANCE", side: "BUY", qty: 0, want: ErrInvalidOrder},
		{name: "negative quantity", symbol: "RELIANCE", side: "BUY", qty: -5, want: ErrInvalidOrder},
		{name: "negative price", symbol: "RELIANCE", side: "BUY", qty: 1, price: -1, want: ErrInvalidOrder},
		{name: "unknown symbol", symbol: "NOPE", side: "BUY", qty: 1, want: ErrInstrumentNotFound},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := k.PlaceOrder(ctx, tc.symbol, tc.side, tc.qty, tc.price); !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}

	if len(srv.placed) != 0 {
		t.Fatalf("%d orders reached the server, want 0 for invalid requests", len(srv.placed))
	}
}

func TestPlaceOrderMapsBrokerErrors(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   error
	}{
		{
			name:   "rate limited",
			status: http.StatusTooManyRequests,
			body:   `{"status":"error","errors":[{"error_code":"429","message":"Too many requests"}]}`,
			want:   ErrRateLimit,
		},
		{
			name:   "expired token",
			status: http.StatusUnauthorized,
			body:   `{"status":"error","errors":[{"error_code":"invalid_token","message":"Invalid access token"}]}`,
			want:   ErrAuth,
		},
		{
			name:   "insufficient margin",
			status: http.StatusBadRequest,
			body:   `{"status":"error","errors":[{"error_code":"invalid_order","message":"Insufficient margin"}]}`,
			want:   ErrInvalidOrder,
		},
		{
			name:   "unknown instrument",
			status: http.StatusBadRequest,
			body:   `{"status":"error","errors":[{"error_code":"invalid_order","message":"unknown tradingsymbol"}]}`,
			want:   ErrInstrumentNotFound,
		},
		{
			name:   "server error",
			status: http.StatusBadGateway,
			body:   `{"status":"error"}`,
			want:   ErrUnavailable,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := newOrderServer(t, map[string]http.HandlerFunc{
				"/orders/regular": func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(tc.status)
					reply(w, tc.body)
				},
			})
			k := newOrderKite(t, srv)

			_, err := k.PlaceOrder(context.Background(), "RELIANCE", "BUY", 1, 0)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want it to wrap %v", err, tc.want)
			}
		})
	}
}

func TestPlaceOrderRejectsResponseWithoutOrderID(t *testing.T) {
	srv := newOrderServer(t, map[string]http.HandlerFunc{
		"/orders/regular": func(w http.ResponseWriter, r *http.Request) {
			reply(w, `{"status":"success"}`)
		},
	})
	k := newOrderKite(t, srv)

	if _, err := k.PlaceOrder(context.Background(), "RELIANCE", "BUY", 1, 0); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("error = %v, want ErrUnavailable", err)
	}
}

func TestCancelOrder(t *testing.T) {
	srv := newOrderServer(t, nil)
	k := newOrderKite(t, srv)

	if err := k.CancelOrder(context.Background(), "4412"); err != nil {
		t.Fatalf("CancelOrder: %v", err)
	}
	if len(srv.cancels) != 1 || srv.cancels[0] != "4412" {
		t.Fatalf("server saw cancels %v, want [4412]", srv.cancels)
	}

	if err := k.CancelOrder(context.Background(), "  "); !errors.Is(err, ErrOrderNotFound) {
		t.Fatalf("empty order id: error = %v, want ErrOrderNotFound", err)
	}
}

func TestCancelOrderNotFound(t *testing.T) {
	srv := newOrderServer(t, map[string]http.HandlerFunc{
		"/orders/regular/": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
			reply(w, `{"status":"error","errors":[{"error_code":"order_id_not_found","message":"Order not found"}]}`)
		},
	})
	k := newOrderKite(t, srv)

	if err := k.CancelOrder(context.Background(), "9999"); !errors.Is(err, ErrOrderNotFound) {
		t.Fatalf("error = %v, want ErrOrderNotFound", err)
	}
}

func TestGetPositions(t *testing.T) {
	srv := newOrderServer(t, nil)
	k := newOrderKite(t, srv)

	got, err := k.GetPositions(context.Background())
	if err != nil {
		t.Fatalf("GetPositions: %v", err)
	}
	want := []contracts.Position{
		{Symbol: "RELIANCE", Quantity: 25, AvgPrice: 2934.5},
		{Symbol: "TCS", Quantity: -10, AvgPrice: 4100.25},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d positions, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("position %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestGetPositionsAcceptsWrappedPayload(t *testing.T) {
	srv := newOrderServer(t, map[string]http.HandlerFunc{
		"/portfolio/positions": func(w http.ResponseWriter, r *http.Request) {
			reply(w, `{"status":"success","data":{"net":[{"tradingsymbol":"RELIANCE","quantity":5,"average_price":100.5}],"day":[]}}`)
		},
	})
	k := newOrderKite(t, srv)

	got, err := k.GetPositions(context.Background())
	if err != nil {
		t.Fatalf("GetPositions: %v", err)
	}
	if len(got) != 1 || got[0].Quantity != 5 {
		t.Fatalf("positions = %+v, want one position of 5", got)
	}
}

func TestDecodeKitePortfolioAndOrderResponses(t *testing.T) {
	positions, err := decodePositions([]byte(`{"status":"success","data":{"net":[{"tradingsymbol":"RELIANCE","quantity":5,"average_price":100.5}],"day":[{"tradingsymbol":"RELIANCE","quantity":3}]}}`))
	if err != nil || len(positions) != 1 || positions[0].Quantity != 5 {
		t.Fatalf("decodePositions = %+v, %v; want only net position", positions, err)
	}

	orders, err := decodeOrderRows([]byte(`{"status":"success","data":[{"order_id":"abc","tag":"mftdeadbeef","status":"COMPLETE","tradingsymbol":"RELIANCE","transaction_type":"BUY","quantity":5,"price":100,"order_type":"LIMIT","filled_quantity":5,"average_price":99.5,"order_timestamp":"2026-09-29 10:31:00","exchange_update_timestamp":"2026-09-29 10:31:02"}]}`))
	if err != nil || len(orders) != 1 {
		t.Fatalf("decodeOrderRows = %+v, %v", orders, err)
	}
	order := orders[0].contract()
	if order.Status != OrderFilled || order.Tag != "mftdeadbeef" || order.FilledQuantity != 5 || order.AverageFillPrice != 99.5 {
		t.Fatalf("normalized order = %+v", order)
	}
	if got := order.CreatedAt.Format(time.RFC3339); got != "2026-09-29T05:01:00Z" {
		t.Fatalf("created timestamp = %s, want UTC conversion from IST", got)
	}
}

func TestDecodeKiteLTPResponse(t *testing.T) {
	got, err := decodeLastPrices([]byte(`{"status":"success","data":{"NSE:RELIANCE":{"last_price":2934.5}}}`))
	if err != nil || got["RELIANCE"] != 2934.5 {
		t.Fatalf("decodeLastPrices = %v, %v", got, err)
	}
	if _, err := decodeLastPrices([]byte(`{"status":"success","data":{"RELIANCE":{"last_price":0}}}`)); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("invalid quote error = %v, want ErrUnavailable", err)
	}
}

func TestKiteEndpointRoutes(t *testing.T) {
	e := endpoints{httpBase: "https://api.kite.trade", wsBase: "wss://ws.kite.trade"}
	if e.positionsURL() != "https://api.kite.trade/portfolio/positions" || e.ordersListURL() != "https://api.kite.trade/orders" || e.orderHistoryURL("id") != "https://api.kite.trade/orders/id" {
		t.Fatalf("Kite REST routes: positions=%s orders=%s history=%s", e.positionsURL(), e.ordersListURL(), e.orderHistoryURL("id"))
	}
}

func TestOrderClientUsesContractsOrderRequest(t *testing.T) {
	srv := newOrderServer(t, nil)
	k := newOrderKite(t, srv)
	client := NewOrderClient(k)

	// An empty type is inferred from the price, as contracts.OrderRequest
	// documents: a non-zero price means LIMIT.
	if _, err := client.PlaceOrder(context.Background(), contracts.OrderRequest{
		Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 1, Price: 100.5,
	}); err != nil {
		t.Fatalf("PlaceOrder: %v", err)
	}
	if got := srv.placed[0].Get("order_type"); got != "LIMIT" {
		t.Errorf("inferred order_type = %q, want LIMIT", got)
	}

	if _, err := client.PlaceOrder(context.Background(), contracts.OrderRequest{
		Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 1, Type: "MARKET",
	}); err != nil {
		t.Fatalf("PlaceOrder: %v", err)
	}
	if got := srv.placed[1].Get("order_type"); got != "MARKET" {
		t.Errorf("explicit order_type = %q, want MARKET", got)
	}

	// MARKET with a price, and LIMIT without one, are contradictory.
	for _, req := range []contracts.OrderRequest{
		{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 1, Price: 100, Type: "MARKET"},
		{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 1, Type: "LIMIT"},
		{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 1, Type: "STOPLOSS"},
	} {
		if _, err := client.PlaceOrder(context.Background(), req); !errors.Is(err, ErrInvalidOrder) {
			t.Errorf("PlaceOrder(%+v): error = %v, want ErrInvalidOrder", req, err)
		}
	}

	if err := client.CancelOrder(context.Background(), "4412"); err != nil {
		t.Fatalf("CancelOrder: %v", err)
	}
	positions, err := client.GetPositions(context.Background())
	if err != nil {
		t.Fatalf("GetPositions: %v", err)
	}
	if len(positions) != 2 {
		t.Fatalf("positions = %+v, want 2", positions)
	}
}

func TestPlaceOrderHonoursLotSize(t *testing.T) {
	dump := `instrument_token,exchange,tradingsymbol,name,expiry,strike,option_type,tick_size,lot_size
738560,NSE,RELIANCE,RELIANCE EQUITY,"",0,,0.05,50
`
	srv := newOrderServer(t, map[string]http.HandlerFunc{
		"/instruments": func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(dump))
		},
	})
	k := newOrderKite(t, srv)
	k.watchlist = []string{"RELIANCE"}

	if _, err := k.PlaceOrder(context.Background(), "RELIANCE", "BUY", 25, 0); !errors.Is(err, ErrInvalidOrder) {
		t.Fatalf("error = %v, want ErrInvalidOrder for a quantity that is not a lot multiple", err)
	}
	if _, err := k.PlaceOrder(context.Background(), "RELIANCE", "BUY", 50, 0); err != nil {
		t.Fatalf("PlaceOrder: %v", err)
	}
}

func TestRequestsHonourContextCancellation(t *testing.T) {
	block := make(chan struct{})
	srv := newOrderServer(t, map[string]http.HandlerFunc{
		"/orders/regular": func(w http.ResponseWriter, r *http.Request) {
			<-block
			reply(w, `{"status":"success","order_id":"1"}`)
		},
	})
	t.Cleanup(func() { close(block) })
	k := newOrderKite(t, srv)

	ctx, cancel := context.WithCancel(context.Background())
	go cancel()

	if _, err := k.PlaceOrder(ctx, "RELIANCE", "BUY", 1, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}
