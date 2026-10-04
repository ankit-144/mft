package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// DefaultBuckets covers the range of operations this platform actually has:
// microsecond-scale cache reads, millisecond-scale order placement against the broker,
// and second-scale backfill jobs.
var DefaultBuckets = []float64{
	0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
	0.25, 0.5, 1, 2.5, 5, 10, 30,
}

// DurationHistogram registers a latency histogram in seconds with no labels.
func DurationHistogram(reg prometheus.Registerer, name, help string) prometheus.Histogram {
	s, err := newDurationSpec(name, help)
	must(err)
	h := prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: s.name, Help: s.help, Buckets: DefaultBuckets,
	})
	MustRegister(reg, h)
	return h
}

// DurationHistogramVec registers a latency histogram in seconds, partitioned by labels,
// using DefaultBuckets.
func DurationHistogramVec(reg prometheus.Registerer, name, help string, labels ...string) *prometheus.HistogramVec {
	s, err := newDurationSpec(name, help)
	must(err)
	h := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{Name: s.name, Help: s.help, Buckets: DefaultBuckets}, labels)
	MustRegister(reg, h)
	return h
}

// ObserveDuration records the time elapsed since start, in seconds, and returns it so
// it can be used directly in a defer:
func ObserveDuration(h prometheus.Observer, start time.Time, id string) time.Duration {
	d := time.Since(start)
	if id != "" {
		if eo, ok := h.(prometheus.ExemplarObserver); ok {
			eo.ObserveWithExemplar(d.Seconds(), prometheus.Labels{"trace_id": id})
			return d
		}
	}
	h.Observe(d.Seconds())
	return d
}
