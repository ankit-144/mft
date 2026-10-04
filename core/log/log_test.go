package log

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mft/core/config"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// loggerFor returns a logger writing to buf, for the given environment.
func loggerFor(t *testing.T, env, level string) (*zap.Logger, *bytes.Buffer) {
	t.Helper()
	buf := &bytes.Buffer{}
	logger, err := Build(Options{Env: env, Level: level, Output: zapcore.AddSync(buf)})
	if err != nil {
		t.Fatalf("Build(env=%q, level=%q): %v", env, level, err)
	}
	return logger, buf
}

func TestFormat(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"dev":         FormatConsole,
		"development": FormatConsole,
		"DEV":         FormatConsole,
		" local ":     FormatConsole,
		"test":        FormatConsole,
		"prod":        FormatJSON,
		"staging":     FormatJSON,
		"":            FormatJSON,
		"production":  FormatJSON,
		"nonsense":    FormatJSON,
	}
	for env, want := range cases {
		if got := Format(env); got != want {
			t.Errorf("Format(%q) = %q, want %q", env, got, want)
		}
	}
}

// TestBuildSelectsEncodingByEnv is the difference between a readable local
// debugging session and grep-able production logs.
func TestBuildSelectsEncodingByEnv(t *testing.T) {
	t.Parallel()

	t.Run("dev is human readable", func(t *testing.T) {
		t.Parallel()
		logger, buf := loggerFor(t, "dev", "info")
		logger.Info("candle aggregator started", zap.String("symbol", "RELIANCE"))
		out := buf.String()
		if !strings.Contains(out, "candle aggregator started") {
			t.Fatalf("message missing from console output: %q", out)
		}
		if !strings.Contains(out, "INFO") || !strings.Contains(out, "RELIANCE") {
			t.Errorf("console output is missing the level or the field: %q", out)
		}
		if strings.HasPrefix(strings.TrimSpace(out), "{") {
			t.Errorf("dev output is JSON, want the console encoder: %q", out)
		}
	})

	t.Run("prod is json", func(t *testing.T) {
		t.Parallel()
		logger, buf := loggerFor(t, "prod", "info")
		logger.Info("candle aggregator started", zap.String("symbol", "RELIANCE"))
		out := strings.TrimSpace(buf.String())
		var entry map[string]any
		if err := json.Unmarshal([]byte(out), &entry); err != nil {
			t.Fatalf("prod output is not one JSON object per line: %q", out)
		}
		if entry["level"] != "info" || entry["msg"] != "candle aggregator started" {
			t.Errorf("entry = %v", entry)
		}
		if entry["symbol"] != "RELIANCE" {
			t.Errorf("symbol field = %v, want RELIANCE", entry["symbol"])
		}
		if _, ok := entry["ts"].(string); !ok {
			t.Errorf("ts = %v, want a formatted timestamp", entry["ts"])
		}
	})
}

func TestParseLevel(t *testing.T) {
	t.Parallel()
	cases := map[string]zapcore.Level{
		"":        zapcore.InfoLevel,
		"info":    zapcore.InfoLevel,
		"INFO":    zapcore.InfoLevel,
		" debug ": zapcore.DebugLevel,
		"warn":    zapcore.WarnLevel,
		"warning": zapcore.WarnLevel,
		"error":   zapcore.ErrorLevel,
	}
	for in, want := range cases {
		got, err := ParseLevel(in)
		if err != nil {
			t.Errorf("ParseLevel(%q): %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("ParseLevel(%q) = %v, want %v", in, got, want)
		}
	}
	if _, err := ParseLevel("verbose"); err == nil {
		t.Error(`ParseLevel("verbose") = nil error, want a rejection`)
	}
}

func TestBuildRejectsUnknownLevel(t *testing.T) {
	t.Parallel()
	if _, err := Build(Options{Env: "prod", Level: "trace"}); err == nil {
		t.Fatal("Build accepted app.log_level: trace")
	}
}

func TestLevelIsHonoured(t *testing.T) {
	t.Parallel()
	cases := []struct {
		level     string
		wantDebug bool
		wantInfo  bool
	}{
		{level: "debug", wantDebug: true, wantInfo: true},
		{level: "info", wantDebug: false, wantInfo: true},
		{level: "warn", wantDebug: false, wantInfo: false},
		{level: "error", wantDebug: false, wantInfo: false},
	}
	for _, tc := range cases {
		t.Run(tc.level, func(t *testing.T) {
			t.Parallel()
			logger, buf := loggerFor(t, "prod", tc.level)
			logger.Debug("debug line")
			logger.Info("info line")
			if got := strings.Contains(buf.String(), "debug line"); got != tc.wantDebug {
				t.Errorf("debug logged = %v, want %v", got, tc.wantDebug)
			}
			if got := strings.Contains(buf.String(), "info line"); got != tc.wantInfo {
				t.Errorf("info logged = %v, want %v", got, tc.wantInfo)
			}
		})
	}
}

// TestNewHonoursConfig is the wiring the services actually use: app.env picks
// the encoding and app.log_level picks the verbosity.
func TestNewHonoursConfig(t *testing.T) {
	t.Parallel()
	cfg := &config.Config{}
	cfg.App.Name = "mft"
	cfg.App.Env = "dev"
	cfg.App.LogLevel = "debug"

	logger, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !logger.Core().Enabled(zapcore.DebugLevel) {
		t.Error("app.log_level: debug did not enable debug logging")
	}

	cfg.App.LogLevel = "chatty"
	if _, err := New(cfg); err == nil {
		t.Error("New accepted an app.log_level it cannot parse")
	}
}

func TestTimezoneIsApplied(t *testing.T) {
	t.Parallel()
	cfg := &config.Config{}
	cfg.App.Env = "prod"
	cfg.App.LogLevel = "info"
	cfg.App.Timezone = "Asia/Kolkata"

	buf := &bytes.Buffer{}
	logger, err := Build(Options{Env: cfg.App.Env, Level: cfg.App.LogLevel, Zone: cfg.App.Timezone,
		Output: zapcore.AddSync(buf)})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	logger.Info("session open")
	if !strings.Contains(buf.String(), "+05:30") {
		t.Errorf("timestamp is not in Asia/Kolkata: %q", buf.String())
	}
}

func TestUnresolvableTimezoneFallsBack(t *testing.T) {
	t.Parallel()
	logger, err := Build(Options{Env: "prod", Level: "info", Zone: "Mars/Olympus_Mons",
		Output: zapcore.AddSync(&bytes.Buffer{})})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if logger == nil {
		t.Fatal("Build returned a nil logger")
	}
}
