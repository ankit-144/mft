package metrics

import (
	"errors"
	"strings"
	"testing"

	"github.com/mft/core/testutil"
	"github.com/prometheus/client_golang/prometheus"
)

func TestValidateName(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name        string
		metric      string
		wantErr     bool
		wantPrefix  bool
		errContains string
	}{
		{name: "compliant", metric: "mft_execution_orders_placed_total"},
		{name: "prefix only", metric: "mft_"},
		{name: "no prefix", metric: "execution_orders_placed_total", wantErr: true, wantPrefix: true,
			errContains: `expected it to be "mft_execution_orders_placed_total"`},
		{name: "prefix without underscore", metric: "mftfoo_total", wantErr: true, wantPrefix: true},
		{name: "empty", metric: "", wantErr: true},
		{name: "hyphen", metric: "mft_tick-rate", wantErr: true, errContains: "invalid"},
		{name: "nested", metric: "mft_execution_orders_placed_seconds_total"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateName(tc.metric)
			if !tc.wantErr {
				if err != nil {
					t.Fatalf("ValidateName(%q) = %v, want nil", tc.metric, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("ValidateName(%q) = nil, want an error", tc.metric)
			}
			if tc.wantPrefix && !errors.Is(err, ErrMissingPrefix) {
				t.Errorf("error %v does not wrap ErrMissingPrefix", err)
			}
			if tc.errContains != "" && !strings.Contains(err.Error(), tc.errContains) {
				t.Errorf("error %q does not contain %q", err, tc.errContains)
			}
		})
	}
}

// TestConstructorsRejectMissingPrefix is the guard the whole platform depends
// on: a metric registered without the mft_ prefix must fail loudly at
// registration instead of scraping under a name nothing else will query.
func TestConstructorsRejectMissingPrefix(t *testing.T) {
	t.Parallel()
	constructors := []struct {
		name string
		call func(reg *prometheus.Registry)
	}{
		{"Counter", func(reg *prometheus.Registry) { Counter(reg, "execution_orders_placed_total", "h") }},
		{"CounterVec", func(reg *prometheus.Registry) { CounterVec(reg, "execution_orders_placed_total", "h", "side") }},
		{"Gauge", func(reg *prometheus.Registry) { Gauge(reg, "risk_drawdown_pct", "h") }},
		{"GaugeVec", func(reg *prometheus.Registry) { GaugeVec(reg, "risk_drawdown_pct", "h", "symbol") }},
		{"Histogram", func(reg *prometheus.Registry) { Histogram(reg, "ticks_batch", "h") }},
		{"HistogramVec", func(reg *prometheus.Registry) { HistogramVec(reg, "ticks_batch", "h", "symbol") }},
		{"DurationHistogram", func(reg *prometheus.Registry) { DurationHistogram(reg, "order_latency", "h") }},
		{"DurationHistogramVec", func(reg *prometheus.Registry) { DurationHistogramVec(reg, "order_latency", "h", "side") }},
	}
	for _, tc := range constructors {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			reg := testutil.NewRegistry()
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("%s registered a metric without the %q prefix", tc.name, Prefix)
				}
				err, ok := r.(error)
				if !ok || !errors.Is(err, ErrMissingPrefix) {
					t.Fatalf("panic = %v, want an error wrapping ErrMissingPrefix", r)
				}
				assertNotRegistered(t, reg, "execution_orders_placed_total")
			}()
			tc.call(reg)
		})
	}
}

func TestConstructorsRequireHelp(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Counter with an empty Help did not panic")
		}
	}()
	Counter(reg, "mft_execution_orders_placed_total", "  ")
}

func TestConstructorsRejectDuplicate(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	Counter(reg, "mft_execution_orders_placed_total", "Total orders placed.")
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("registering the same metric twice did not panic")
		}
	}()
	Counter(reg, "mft_execution_orders_placed_total", "Total orders placed.")
}

func TestConstructorsRegisterUsableMetrics(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	Counter(reg, "mft_execution_orders_placed_total", "Total orders placed.").Inc()
	CounterVec(reg, "mft_execution_orders_rejected_total", "Total orders rejected by the risk gate.", "reason").
		WithLabelValues("RISK_DEBOUNCED").Inc()
	Gauge(reg, "mft_execution_position_count", "Open positions.").Set(3)

	if got := testutil.MetricValue(t, "mft_execution_orders_placed_total", reg, nil); got != 1 {
		t.Errorf("orders placed = %d, want 1", got)
	}
	if got := testutil.MetricValue(t, "mft_execution_position_count", reg, nil); got != 3 {
		t.Errorf("position count = %d, want 3", got)
	}
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() != "mft_execution_orders_rejected_total" {
			continue
		}
		if len(f.GetMetric()) != 1 || len(f.GetMetric()[0].GetLabel()) != 1 {
			t.Fatalf("rejections = %v, want a single series with one label", f.GetMetric())
		}
		if got := f.GetMetric()[0].GetLabel()[0].GetValue(); got != "RISK_DEBOUNCED" {
			t.Errorf("reason label = %q, want RISK_DEBOUNCED", got)
		}
		return
	}
	t.Fatal("mft_execution_orders_rejected_total was not gathered")
}

func TestMustRegisterRejectsDuplicate(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	c := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "mft_execution_orders_placed_total",
		Help: "Total orders placed.",
	})
	MustRegister(reg, c)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("MustRegister accepted a metric that was already registered")
		}
	}()
	MustRegister(reg, c)
}

// assertNotRegistered fails when name is present in reg.
func assertNotRegistered(t *testing.T, reg *prometheus.Registry, name string) {
	t.Helper()
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range families {
		if f.GetName() == name {
			t.Errorf("%s is registered; the guard must stop the registration", name)
		}
	}
}
