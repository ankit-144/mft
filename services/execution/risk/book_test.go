package risk

import (
	"testing"

	"github.com/mft/core/contracts"
)

func TestBookEquityAccounting(t *testing.T) {
	book := NewBook(100_000)

	// A flat book is all cash.
	if got := book.Snapshot(); got.Cash != 100_000 || got.RealisedPnL != 0 {
		t.Fatalf("initial book = %+v, want cash 100000 and no PnL", got)
	}

	// Opening a position moves cash into the position, which does not leave
	// the account: total cash and equity are unchanged, and the peak is not
	// advanced by the trade itself.
	if err := book.Apply(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000}); err != nil {
		t.Fatalf("Apply(buy) error = %v", err)
	}
	got := book.Snapshot()
	if got.Cash != 100_000 {
		t.Fatalf("account cash after buy = %v, want 100000", got.Cash)
	}
	if Equity(got) != 100_000 {
		t.Fatalf("equity after buy = %v, want 100000", Equity(got))
	}
	if book.FreeCash() != 90_000 {
		t.Fatalf("free cash after buy = %v, want 90000", book.FreeCash())
	}
	if got.PeakEquity != 100_000 {
		t.Fatalf("peak after buy = %v, want 100000", got.PeakEquity)
	}
	if got.OpenPositions["RELIANCE"] != 10 {
		t.Fatalf("positions after buy = %+v, want RELIANCE 10", got.OpenPositions)
	}

	// Closing above cost books the profit exactly once: the entry cost comes
	// back to cash and the difference lands in realised PnL.
	if err := book.Apply(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 10, Price: 1100}); err != nil {
		t.Fatalf("Apply(sell) error = %v", err)
	}
	got = book.Snapshot()
	if got.Cash != 100_000 {
		t.Fatalf("cash after sell = %v, want 100000", got.Cash)
	}
	if got.RealisedPnL != 1_000 {
		t.Fatalf("realised PnL = %v, want 1000", got.RealisedPnL)
	}
	if Equity(got) != 101_000 {
		t.Fatalf("equity after sell = %v, want 101000", Equity(got))
	}
	if got.PeakEquity != 101_000 {
		t.Fatalf("peak after sell = %v, want 101000", got.PeakEquity)
	}
	if len(got.OpenPositions) != 0 {
		t.Fatalf("positions after close = %+v, want none", got.OpenPositions)
	}
}

func TestBookPartialCloseKeepsAverageCost(t *testing.T) {
	book := NewBook(100_000)
	mustApply(t, book, contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000})
	mustApply(t, book, contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1200})

	// Average cost is now 1100; selling 20 at 1300 realises 4000.
	mustApply(t, book, contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 20, Price: 1300})

	got := book.Snapshot()
	if got.RealisedPnL != 4_000 {
		t.Fatalf("realised PnL = %v, want 4000", got.RealisedPnL)
	}
	if got.Cash != 100_000 {
		t.Fatalf("cash = %v, want 100000", got.Cash)
	}
}

func TestBookRejectsUnrepresentableFills(t *testing.T) {
	cases := []struct {
		name string
		book *Book
		sig  contracts.Signal
	}{
		{
			name: "short",
			book: NewBook(100_000),
			sig:  contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 10, Price: 1000},
		},
		{
			name: "sell beyond the long position",
			book: NewBook(100_000),
			sig:  contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 1, Price: 1000},
		},
		{
			name: "unaffordable buy",
			book: NewBook(1_000),
			sig:  contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000},
		},
		{
			name: "unknown side",
			book: NewBook(100_000),
			sig:  contracts.Signal{Symbol: "RELIANCE", Side: "HOLD", Quantity: 10, Price: 1000},
		},
		{
			name: "non-positive price",
			book: NewBook(100_000),
			sig:  contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10},
		},
		{
			name: "non-positive quantity",
			book: NewBook(100_000),
			sig:  contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 0, Price: 1000},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := tc.book.Snapshot()
			if err := tc.book.Apply(tc.sig); err == nil {
				t.Fatal("Apply() = nil, want a refusal")
			}
			after := tc.book.Snapshot()
			if after.Cash != before.Cash || after.RealisedPnL != before.RealisedPnL {
				t.Fatalf("book changed on a refused fill: before %+v, after %+v", before, after)
			}
		})
	}
}

func TestBookSnapshotIsACopy(t *testing.T) {
	book := NewBook(100_000)
	mustApply(t, book, contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000})

	snap := book.Snapshot()
	snap.OpenPositions["RELIANCE"] = 999
	snap.OpenPositions["INFY"] = 5

	if got := book.Snapshot().OpenPositions["RELIANCE"]; got != 10 {
		t.Fatalf("RELIANCE quantity = %d, want 10: the snapshot leaked internal state", got)
	}
	if _, ok := book.Snapshot().OpenPositions["INFY"]; ok {
		t.Fatal("a symbol added to the snapshot reached the book")
	}
}

func mustApply(t *testing.T, b *Book, sig contracts.Signal) {
	t.Helper()
	if err := b.Apply(sig); err != nil {
		t.Fatalf("Apply(%+v) error = %v", sig, err)
	}
}
