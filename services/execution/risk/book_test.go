package risk

import (
	"testing"
	"time"

	"github.com/mft/core/contracts"
)

func TestBookEquityAccounting(t *testing.T) {
	book := NewBook(100_000)

	// A flat book is all cash.
	if got := book.Snapshot(); got.Cash != 100_000 || got.RealisedPnL != 0 {
		t.Fatalf("initial book = %+v, want cash 100000 and no PnL", got)
	}

	// Opening a position moves cash into the position: marked equity is unchanged and the peak is not
	// advanced by the trade itself.
	if err := book.Apply(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000}); err != nil {
		t.Fatalf("Apply(buy) error = %v", err)
	}
	got := book.Snapshot()
	if got.Cash != 90_000 {
		t.Fatalf("available cash after buy = %v, want 90000", got.Cash)
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

	// Closing above cost books actual sale proceeds and realized PnL once.
	if err := book.Apply(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 10, Price: 1100}); err != nil {
		t.Fatalf("Apply(sell) error = %v", err)
	}
	got = book.Snapshot()
	if got.Cash != 101_000 {
		t.Fatalf("cash after sell = %v, want 101000", got.Cash)
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
	if got.Cash != 104_000 {
		t.Fatalf("cash = %v, want 104000", got.Cash)
	}
}

func TestBookMarksPositionsAndTracksTradingDayPnL(t *testing.T) {
	book := NewBook(100_000)
	dayOne := time.Date(2026, 10, 1, 5, 0, 0, 0, time.UTC)
	sig := contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy, Quantity: 10, Price: 1000}
	if err := book.ApplyFill(sig, 10, 1000, dayOne); err != nil {
		t.Fatal(err)
	}
	if err := book.Mark("RELIANCE", 1500, dayOne.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if got := book.SnapshotAt(dayOne.Add(time.Minute)); got.Equity != 105_000 {
		t.Fatalf("marked equity = %.2f, want 105000", got.Equity)
	}
	if err := book.Mark("RELIANCE", 900, dayOne.Add(2*time.Minute)); err != nil {
		t.Fatal(err)
	}
	if got := book.SnapshotAt(dayOne.Add(2 * time.Minute)); got.Equity != 99_000 {
		t.Fatalf("marked equity = %.2f, want 99000", got.Equity)
	}

	sell := contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell, Quantity: 10, Price: 900}
	if err := book.ApplyFill(sell, 10, 900, dayOne.Add(3*time.Minute)); err != nil {
		t.Fatal(err)
	}
	dayTwo := dayOne.Add(24 * time.Hour)
	if got := book.SnapshotAt(dayOne.Add(4 * time.Minute)); got.RealisedPnLToday != -1_000 {
		t.Fatalf("day one PnL = %.2f, want -1000", got.RealisedPnLToday)
	}
	if got := book.SnapshotAt(dayTwo); got.RealisedPnLToday != 0 || got.RealisedPnL != -1_000 {
		t.Fatalf("day two snapshot = %+v, want zero daily and -1000 lifetime PnL", got)
	}
}

func TestBookStateRoundTripPreservesCashPositionsMarksAndDailyPnL(t *testing.T) {
	book := NewBook(100_000)
	at := time.Date(2026, 10, 1, 5, 0, 0, 0, time.UTC)
	mustFill(t, book, contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy}, 10, 1000, at)
	mustFill(t, book, contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideSell}, 2, 900, at)
	state := book.State()
	restored, err := RestoreBook(state)
	if err != nil {
		t.Fatal(err)
	}
	got, want := restored.SnapshotAt(at), book.SnapshotAt(at)
	if got.Cash != want.Cash || got.Equity != want.Equity || got.RealisedPnLToday != want.RealisedPnLToday || got.OpenPositions["RELIANCE"] != want.OpenPositions["RELIANCE"] {
		t.Fatalf("restored = %+v, want %+v", got, want)
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

func TestBookRecordsActualFillEvenWhenItExceedsReservedCash(t *testing.T) {
	book := NewBook(1_000)
	if err := book.ApplyFill(contracts.Signal{Symbol: "RELIANCE", Side: contracts.SideBuy}, 2, 1000, time.Now()); err != nil {
		t.Fatalf("actual broker fill must be recorded: %v", err)
	}
	got := book.Snapshot()
	if got.Cash != -1_000 || got.OpenPositions["RELIANCE"] != 2 || got.Equity != 1_000 {
		t.Fatalf("actual fill accounting = %+v; want -1000 cash, two units, 1000 equity", got)
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

func mustFill(t *testing.T, b *Book, sig contracts.Signal, qty int, price float64, at time.Time) {
	t.Helper()
	if err := b.ApplyFill(sig, qty, price, at); err != nil {
		t.Fatalf("ApplyFill(%+v) error = %v", sig, err)
	}
}
