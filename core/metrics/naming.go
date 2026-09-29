package metrics

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
)

// Prefix is the mandatory namespace for every metric this platform exposes.
// It is what lets one Prometheus instance, one Grafana dashboard and one alert
// set cover all four services without two of them shadowing each other.
const Prefix = "mft_"

// DurationSuffix is the mandatory suffix for a metric that measures elapsed
// time. The unit belongs in the name, not in a comment.
const DurationSuffix = "_seconds"

// ErrMissingPrefix is the sentinel wrapped by ValidateName when a metric name
// does not carry Prefix. Tests and linters can match it with errors.Is.
var ErrMissingPrefix = errors.New("metric name must start with " + Prefix)

// promNameRE is the Prometheus metric name grammar.
var promNameRE = regexp.MustCompile(`^[a-zA-Z_:][a-zA-Z0-9_:]*$`)

// ValidateName returns an error unless name is a syntactically valid Prometheus
// metric name that carries the mandatory Prefix. It is the single place the
// prefix rule is enforced: every constructor in this package calls it before
// touching a registry, so a violation panics at startup rather than scraping
// under a foreign name.
func ValidateName(name string) error {
	if name == "" {
		return errors.New("metric name must not be empty")
	}
	if !strings.HasPrefix(name, Prefix) {
		return fmt.Errorf("%w: %q, expected it to be %q", ErrMissingPrefix, name, Prefix+name)
	}
	if !promNameRE.MatchString(name) {
		return fmt.Errorf("metric name %q is invalid: want [a-zA-Z_:][a-zA-Z0-9_:]*", name)
	}
	return nil
}

// ValidateDurationName returns an error unless name is valid per ValidateName
// and ends in DurationSuffix.
func ValidateDurationName(name string) error {
	if err := ValidateName(name); err != nil {
		return err
	}
	if !strings.HasSuffix(name, DurationSuffix) {
		return fmt.Errorf("latency metric %q must end in %q so the unit is unambiguous", name, DurationSuffix)
	}
	return nil
}

// spec is a validated metric name and Help string.
type spec struct {
	name string
	help string
}

// newSpec validates a metric against the platform rules: the mft_ prefix, the
// Prometheus name grammar, and a non-empty Help.
func newSpec(name, help string) (spec, error) {
	if err := ValidateName(name); err != nil {
		return spec{}, err
	}
	if strings.TrimSpace(help) == "" {
		return spec{}, fmt.Errorf("metric %q needs a non-empty Help string", name)
	}
	return spec{name: name, help: help}, nil
}

// newDurationSpec additionally requires the DurationSuffix.
func newDurationSpec(name, help string) (spec, error) {
	if err := ValidateDurationName(name); err != nil {
		return spec{}, err
	}
	if strings.TrimSpace(help) == "" {
		return spec{}, fmt.Errorf("metric %q needs a non-empty Help string", name)
	}
	return spec{name: name, help: help}, nil
}

// must panics with err. Metric construction is a programmer error, not a
// runtime condition: a name without the prefix, a missing Help, or a duplicate
// registration must stop the process at startup instead of quietly dropping a
// series that someone will miss at 09:31 on a Monday.
func must(err error) {
	if err != nil {
		panic(err)
	}
}

// MustRegister registers every collector in reg, panicking on any error. It is
// the registration path for metrics not built by the constructors in this
// file, and it applies the same rules: the name must carry Prefix, the Help
// must not be empty, and the name must not already be taken.
func MustRegister(reg prometheus.Registerer, collectors ...prometheus.Collector) {
	for _, c := range collectors {
		if err := reg.Register(c); err != nil {
			must(fmt.Errorf("register metric: %w", err))
		}
	}
}

// Counter registers and returns a counter with no labels.
func Counter(reg prometheus.Registerer, name, help string) prometheus.Counter {
	s, err := newSpec(name, help)
	must(err)
	c := prometheus.NewCounter(prometheus.CounterOpts{Name: s.name, Help: s.help})
	MustRegister(reg, c)
	return c
}

// CounterVec registers and returns a counter partitioned by labels.
func CounterVec(reg prometheus.Registerer, name, help string, labels ...string) *prometheus.CounterVec {
	s, err := newSpec(name, help)
	must(err)
	c := prometheus.NewCounterVec(prometheus.CounterOpts{Name: s.name, Help: s.help}, labels)
	MustRegister(reg, c)
	return c
}

// Gauge registers and returns a gauge with no labels.
func Gauge(reg prometheus.Registerer, name, help string) prometheus.Gauge {
	s, err := newSpec(name, help)
	must(err)
	g := prometheus.NewGauge(prometheus.GaugeOpts{Name: s.name, Help: s.help})
	MustRegister(reg, g)
	return g
}

// GaugeVec registers and returns a gauge partitioned by labels.
func GaugeVec(reg prometheus.Registerer, name, help string, labels ...string) *prometheus.GaugeVec {
	s, err := newSpec(name, help)
	must(err)
	g := prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: s.name, Help: s.help}, labels)
	MustRegister(reg, g)
	return g
}

// Histogram registers and returns a histogram with DefaultBuckets.
func Histogram(reg prometheus.Registerer, name, help string) prometheus.Histogram {
	s, err := newSpec(name, help)
	must(err)
	h := prometheus.NewHistogram(prometheus.HistogramOpts{Name: s.name, Help: s.help, Buckets: DefaultBuckets})
	MustRegister(reg, h)
	return h
}

// HistogramVec registers and returns a histogram partitioned by labels, using
// DefaultBuckets.
func HistogramVec(reg prometheus.Registerer, name, help string, labels ...string) *prometheus.HistogramVec {
	s, err := newSpec(name, help)
	must(err)
	h := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{Name: s.name, Help: s.help, Buckets: DefaultBuckets}, labels)
	MustRegister(reg, h)
	return h
}
