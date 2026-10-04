package execution

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mft/core/contracts"
	"github.com/mft/services/execution/risk"
)

func TestJournalPersistsOrderAndBookAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "execution", "state.json")
	j, state, err := openJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	book := risk.NewBook(100_000)
	at := time.Date(2026, 10, 4, 5, 0, 0, 0, time.UTC)
	if err := book.ApplyFill(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy}, 10, 1000, at); err != nil {
		t.Fatal(err)
	}
	state.Initialized = true
	state.Book = book.State()
	key := "signal-key"
	state.Orders[key] = OrderRecord{Key: key, Tag: orderTag(key), Status: contracts.OrderStatusOpen, OrderID: "broker-1", FilledQuantity: 4, AppliedNotional: 4000, AverageFillPrice: 1000, RiskPrice: 1000, Request: contracts.OrderRequest{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000, IdempotencyKey: key}}
	if err := j.Save(state); err != nil {
		t.Fatal(err)
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}

	j2, got, err := openJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	defer j2.Close()
	if !got.Initialized || got.Orders["signal-key"].OrderID != "broker-1" || got.Book.Held["RELIANCE"] != 10 {
		t.Fatalf("reopened state = %+v", got)
	}
}

func TestJournalIsSingleWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	j, _, err := openJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	if _, _, err := openJournal(path); err == nil {
		t.Fatal("second journal owner succeeded")
	}
}

func TestJournalRejectsCorruptState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"initialized":true} trailing`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := openJournal(path); err == nil {
		t.Fatal("corrupt journal was accepted")
	}
}
