package ingestion

import (
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/storage"
	"github.com/mft/core/testutil"
)

// testPipeline builds a pipeline with a real (unstarted) storage writer; its
// Append only queues into a buffered channel, so it is safe without Start.
func testPipeline(t *testing.T) *Pipeline {
	t.Helper()
	return NewPipeline(
		&testutil.MockStreamer{},
		fluxkv.New(),
		storage.NewWriter(t.TempDir(), 100),
		testutil.NewRegistry(),
		testutil.NewLogger(),
	)
}

func TestProcessUpdatesCandleAndStoresTick(t *testing.T) {
	p := testPipeline(t)
	ticks := make(chan broker.Tick)

	done := make(chan struct{})
	go func() {
		defer close(done)
		p.process(ticks)()
	}()

	base := time.Date(2026, 8, 1, 10, 0, 0, 0, time.UTC)
	ticks <- testutil.Tick("RELIANCE", 100, base)
	ticks <- testutil.Tick("RELIANCE", 105, base.Add(5*time.Second))
	close(ticks)

	<-done

	c := p.cache.Candle("RELIANCE")
	if c == nil {
		t.Fatal("expected candle to be created")
	}
	if c.Open != 100 || c.Close != 105 {
		t.Fatalf("unexpected candle: %+v", c)
	}
}
