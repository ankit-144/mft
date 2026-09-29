package metrics

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// BuildInfo describes the binary currently running. Every value becomes a
// const label on mft_build_info, which is what lets an operator answer "which
// commit was running when this alert fired" from a dashboard instead of from
// tribal memory.
type BuildInfo struct {
	// Service is the emitting service: ingestion, execution, jobs, inference.
	Service string
	// Version is the release version, e.g. "v1.0.0".
	Version string
	// Commit is the git SHA the binary was built from.
	Commit string
	// Env is the deployment environment: dev, staging or prod.
	Env string
	// GoVersion overrides the reported Go toolchain. Leave empty to use the
	// toolchain that is running.
	GoVersion string
	// Branch is the source branch, when the build can report one.
	Branch string
}

// RegisterBuildInfo registers mft_build_info and mft_uptime_seconds against reg.
// Call it once, from the service constructor, before the metrics server starts;
// a non-nil error means a second registration in the same process and should be
// treated as fatal.
//
// Version, commit and branch come from the Go build info — the linker stamps them
// from the VCS by default, or from -X main.version=... — and fall back to
// "unknown", so a plain `go run` still produces a complete, honest series.
func RegisterBuildInfo(reg prometheus.Registerer, info BuildInfo) error {
	version, commit, branch := info.Version, info.Commit, info.Branch
	if v, c, b, ok := readBuildStamps(); ok {
		if version == "" {
			version = v
		}
		if commit == "" {
			commit = c
		}
		if branch == "" {
			branch = b
		}
	}
	if version == "" {
		version = "unknown"
	}
	if commit == "" {
		commit = "unknown"
	}
	if info.GoVersion == "" {
		info.GoVersion = runtime.Version()
	}

	name := prometheus.BuildFQName("mft", "build", "info")
	if err := ValidateName(name); err != nil {
		return err
	}

	labels := prometheus.Labels{
		"service":   info.Service,
		"version":   version,
		"commit":    commit,
		"branch":    branch,
		"goversion": info.GoVersion,
		"env":       info.Env,
	}
	build := prometheus.NewGauge(prometheus.GaugeOpts{
		Name:        name,
		Help:        "Build metadata of the running binary. Always 1; the metadata is carried in labels.",
		ConstLabels: labels,
	})
	build.Set(1)

	uptime := prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: Prefix + "uptime_seconds",
		Help: "Seconds since this process started.",
	}, func() float64 { return time.Since(processStart).Seconds() })

	if err := reg.Register(build); err != nil {
		return fmt.Errorf("register build info: %w", err)
	}
	if err := reg.Register(uptime); err != nil {
		return fmt.Errorf("register uptime: %w", err)
	}
	return nil
}

// processStart is the process start time, used for the uptime gauge and the
// health report.
var processStart = time.Now()

// readBuildStamps pulls version, commit and branch out of the Go build info,
// which the linker populates from -ldflags -X.
func readBuildStamps() (version, commit, branch string, ok bool) {
	bi, available := debug.ReadBuildInfo()
	if !available {
		return "", "", "", false
	}
	version = bi.Main.Version
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			commit = s.Value
		case "vcs.modified":
			if s.Value == "true" && commit != "" {
				commit += "-dirty"
			}
		case "vcs.ref":
			branch = strings.TrimPrefix(s.Value, "refs/heads/")
		}
	}
	return version, commit, branch, true
}
