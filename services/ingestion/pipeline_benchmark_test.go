package ingestion

import (
	"testing"
	"time"

	"github.com/mft/core/broker"
	"github.com/mft/core/fluxkv"
)

func BenchmarkPipelineFold(b *testing.B) {
	p := &Pipeline{cache: fluxkv.New(), state: make(map[string]symbolState)}
	base := time.Date(2026, 8, 3, 9, 15, 0, 0, indiaLocation)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.fold(broker.Tick{
			Symbol: "RELIANCE", Token: token, Price: 100 + float64(i%50), Volume: int64(i),
			Timestamp: base.Add(time.Duration(i) * time.Nanosecond),
		})
	}
}
