package metrics

import (
	"strings"
	"testing"

	"github.com/mft/core/testutil"
	"github.com/prometheus/client_golang/prometheus"
)

func TestRegisterBuildInfo(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	err := RegisterBuildInfo(reg, BuildInfo{
		Service:   "execution",
		Version:   "v1.2.3",
		Commit:    "abc1234",
		Branch:    "feat/risk-engine",
		Env:       "dev",
		GoVersion: "go1.25.0",
	})
	if err != nil {
		t.Fatalf("RegisterBuildInfo: %v", err)
	}

	build := gatherFamily(t, reg, "mft_build_info")
	labels := map[string]string{}
	for _, l := range build.Metric[0].GetLabel() {
		labels[l.GetName()] = l.GetValue()
	}
	want := map[string]string{
		"service":   "execution",
		"version":   "v1.2.3",
		"commit":    "abc1234",
		"branch":    "feat/risk-engine",
		"env":       "dev",
		"goversion": "go1.25.0",
	}
	for k, v := range want {
		if labels[k] != v {
			t.Errorf("mft_build_info{%s} = %q, want %q", k, labels[k], v)
		}
	}
	if got := build.Metric[0].GetGauge().GetValue(); got != 1 {
		t.Errorf("mft_build_info = %v, want 1", got)
	}

	uptime := gatherFamily(t, reg, Prefix+"uptime_seconds")
	if len(uptime.Metric) != 1 || len(uptime.Metric[0].GetLabel()) != 0 {
		t.Fatalf("mft_uptime_seconds = %+v, want one unlabelled series", uptime.Metric)
	}
	if got := uptime.Metric[0].GetGauge().GetValue(); got < 0 {
		t.Errorf("uptime = %v, want >= 0", got)
	}
}

func TestRegisterBuildInfoDefaults(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	if err := RegisterBuildInfo(reg, BuildInfo{Service: "jobs", Env: "prod"}); err != nil {
		t.Fatalf("RegisterBuildInfo: %v", err)
	}
	build := gatherFamily(t, reg, "mft_build_info")
	labels := map[string]string{}
	for _, l := range build.Metric[0].GetLabel() {
		labels[l.GetName()] = l.GetValue()
	}
	if labels["version"] == "" || labels["commit"] == "" {
		t.Errorf("version/commit must never be empty: %+v", labels)
	}
	if !strings.HasPrefix(labels["goversion"], "go1") {
		t.Errorf("goversion = %q, want the running toolchain", labels["goversion"])
	}
}

func TestRegisterBuildInfoTwiceFails(t *testing.T) {
	t.Parallel()
	reg := testutil.NewRegistry()
	if err := RegisterBuildInfo(reg, BuildInfo{Service: "jobs"}); err != nil {
		t.Fatalf("first RegisterBuildInfo: %v", err)
	}
	if err := RegisterBuildInfo(reg, BuildInfo{Service: "jobs"}); err == nil {
		t.Fatal("registering build info twice must report an error")
	}
}

func TestRegistryHasGoCollectors(t *testing.T) {
	t.Parallel()
	reg := Registry()
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	seen := map[string]bool{}
	for _, f := range families {
		seen[f.GetName()] = true
	}
	for _, name := range []string{"go_goroutines", "process_start_time_seconds"} {
		if !seen[name] {
			t.Errorf("%s is not registered", name)
		}
	}
}

var _ prometheus.Registerer = testutil.NewRegistry()
