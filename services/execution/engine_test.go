package execution

import (
	"context"
	"testing"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/testutil"
)

func newTestEngine(t *testing.T, client *testutil.MockClient) *Engine {
	t.Helper()
	cfg := &config.Config{}
	cfg.Execution.DebounceTTLSeconds = 300
	cfg.Execution.Addr = ":0"
	return NewEngine(client, fluxkv.New(), cfg, testutil.NewRegistry(), testutil.NewLogger())
}

func TestExecutePlacesOrder(t *testing.T) {
	client := &testutil.MockClient{}
	engine := newTestEngine(t, client)

	orderID, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 100)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if orderID != "mock-order" {
		t.Fatalf("order id = %q, want %q", orderID, "mock-order")
	}
	if len(client.Orders) != 1 {
		t.Fatalf("expected 1 order, got %d", len(client.Orders))
	}
	if client.Orders[0].Symbol != "RELIANCE" || client.Orders[0].Side != "BUY" {
		t.Fatalf("unexpected order: %+v", client.Orders[0])
	}
}

func TestExecuteDebounced(t *testing.T) {
	client := &testutil.MockClient{}
	engine := newTestEngine(t, client)

	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 100); err != nil {
		t.Fatalf("first Execute() error = %v", err)
	}

	// A second identical signal within the debounce window must be rejected.
	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 100); err == nil {
		t.Fatal("expected debounced signal to be rejected")
	}
	if len(client.Orders) != 1 {
		t.Fatalf("expected only 1 order placed, got %d", len(client.Orders))
	}
}

func TestExecuteDebounceExpires(t *testing.T) {
	client := &testutil.MockClient{}
	cfg := &config.Config{}
	cfg.Execution.DebounceTTLSeconds = 0 // immediate expiry
	engine := NewEngine(client, fluxkv.New(), cfg, testutil.NewRegistry(), testutil.NewLogger())

	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 100); err != nil {
		t.Fatalf("first Execute() error = %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 100); err != nil {
		t.Fatalf("expected debounce to expire, got error: %v", err)
	}
	if len(client.Orders) != 2 {
		t.Fatalf("expected 2 orders after expiry, got %d", len(client.Orders))
	}
}

func TestExecuteBrokerError(t *testing.T) {
	client := &testutil.MockClient{FailErr: context.Canceled}
	engine := newTestEngine(t, client)

	if _, err := engine.Execute(context.Background(), "RELIANCE", "BUY", 10, 100); err == nil {
		t.Fatal("expected broker error to propagate")
	}
}
