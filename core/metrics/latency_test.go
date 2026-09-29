package metrics

import (
	"math"
	"testing"
	"time"

	"github.com/mft/core/testutil"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func TestValidateDurationName(t *testing.T) {
	t.Parallel()
	if err := ValidateDurationName("mft_execution_order_duration_seconds"); err != nil {
		t.Fatalf("compliant latency name rejected: %v", err)
	}
	for _, name := range []string{
		"mft_execution_order_latency",    // no unit
		"mft_execution_order_seconds_v2", // unit in the wrong place
		"execution_order_duration_seconds",
	} {
		if err := ValidateDurationName(name); err == nil {
			t.Errorf("ValidateDurationName(%q) = nil, want an error", name)
		}
	}
}

func TestDurationHistogramRejectsMissingUnit(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("a latency metric without the _seconds suffix was accepted")
		}
	}()
	DurationHistogram(reg, "mft_execution_order_latency", "Order latency.")
}

func TestDurationHistogramObserves(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	hist := DurationHistogramVec(reg, "mft_execution_order_duration_seconds",
		"Time to place an order, end to end.", "side")
	hist.WithLabelValues("BUY").Observe(0.02)

	family := gatherFamily(t, reg, "mft_execution_order_duration_seconds")
	if len(family.Metric) != 1 {
		t.Fatalf("got %d series, want 1", len(family.Metric))
	}
	m := family.Metric[0]
	if m.GetHistogram().GetSampleCount() != 1 {
		t.Errorf("sample count = %d, want 1", m.GetHistogram().GetSampleCount())
	}
	if got := m.GetHistogram().GetSampleSum(); math.Abs(got-0.02) > 1e-9 {
		t.Errorf("sample sum = %v, want 0.02", got)
	}
	// 0.02s must land in the 0.025 bucket and every bucket above it.
	counts := map[float64]uint64{}
	for _, b := range m.GetHistogram().GetBucket() {
		counts[b.GetUpperBound()] = b.GetCumulativeCount()
	}
	for _, upper := range []float64{0.025, 0.05, 0.1, 30} {
		if counts[upper] != 1 {
			t.Errorf("bucket le=%v count = %d, want 1", upper, counts[upper])
		}
	}
	if counts[0.01] != 0 {
		t.Errorf("bucket le=0.01 count = %d, want 0", counts[0.01])
	}
}

func TestObserveDurationRecordsSecondsAndExemplar(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	hist := DurationHistogram(reg, "mft_execution_order_duration_seconds", "Time to place an order.")

	const id = "9f1b0d3c2a1f4e5b8c7d6a5f4e3d2c1b"
	elapsed := ObserveDuration(hist, time.Now().Add(-250*time.Millisecond), id)
	if elapsed < 250*time.Millisecond {
		t.Fatalf("ObserveDuration returned %v, want at least 250ms", elapsed)
	}

	family := gatherFamily(t, reg, "mft_execution_order_duration_seconds")
	if got := family.Metric[0].GetHistogram().GetSampleSum(); got < 0.25 || got > 1 {
		t.Errorf("sample sum = %v, want ~0.25 (seconds, not nanoseconds)", got)
	}

	var exemplars []*dto.Exemplar
	for _, b := range family.Metric[0].GetHistogram().GetBucket() {
		if e := b.GetExemplar(); e != nil {
			exemplars = append(exemplars, e)
		}
	}
	if len(exemplars) != 1 {
		t.Fatalf("got %d exemplars, want 1", len(exemplars))
	}
	if got := exemplars[0].GetLabel()[0].GetValue(); got != id {
		t.Errorf("exemplar trace_id = %q, want %q", got, id)
	}
}

// TestObserveDurationWithoutExemplar covers the plain observer, where the
// correlation id is dropped rather than invented.
func TestObserveDurationWithoutExemplar(t *testing.T) {
	t.Parallel()
	observer := &plainObserver{}
	ObserveDuration(observer, time.Now().Add(-time.Second), "correlation-id")

	if observer.observed != 1 {
		t.Fatalf("observed %d times, want 1", observer.observed)
	}
	if observer.last < 0.9 || observer.last > 1.5 {
		t.Errorf("observed %v seconds, want ~1", observer.last)
	}
}

// plainObserver is a prometheus.Observer that is not an ExemplarObserver.
type plainObserver struct {
	observed int
	last     float64
}

func (o *plainObserver) Observe(v float64) {
	o.observed++
	o.last = v
}

// gatherFamily returns the single gathered metric family called name.
func gatherFamily(t *testing.T, reg *prometheus.Registry, name string) *dto.MetricFamily {
	t.Helper()
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() == name {
			return f
		}
	}
	t.Fatalf("%s was not gathered", name)
	return nil
}
