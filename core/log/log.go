// Package log provides the shared zap logger for every MFT service.
//
// It reads app.env and app.log_level from the frozen config schema and turns
// them into two decisions that matter in practice: how loud to be, and whether
// to write JSON for a log shipper or a colourised console for the person
// debugging on a laptop at 15:29 IST.
//
// Every logger it returns is wrapped in a redacting core, so a credential
// cannot reach stdout even if a caller hands it a whole config struct. See
// redact.go.
package log

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/mft/core/config"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Encodings for the shared logger.
const (
	// FormatConsole is the human-readable colourised encoding used in dev.
	FormatConsole = "console"
	// FormatJSON is the machine-readable encoding used everywhere else.
	FormatJSON = "json"
)

// Options configures the shared logger.
type Options struct {
	// Env is the deployment environment: dev, staging or prod. It selects the
	// encoding; anything unrecognised is treated as prod.
	Env string
	// Level is app.log_level: debug, info, warn or error. Empty means info.
	Level string
	// Zone is the app.timezone the timestamps are rendered in, e.g.
	// "Asia/Kolkata". An empty or unresolvable zone falls back to UTC.
	Zone string
	// Output receives the log lines. Defaults to os.Stderr.
	Output zapcore.WriteSyncer
}

// New builds the shared logger from the frozen config schema. It is the fx
// provider used by core.Module, and it is the only constructor that reads
// config; Build is the one to use in tests and in code that is not driven by
// config.
func New(cfg *config.Config) (*zap.Logger, error) {
	return Build(Options{
		Env:   cfg.App.Env,
		Level: cfg.App.LogLevel,
		Zone:  cfg.App.Timezone,
	})
}

// Build constructs a redacting zap logger from opts.
func Build(opts Options) (*zap.Logger, error) {
	level, err := ParseLevel(opts.Level)
	if err != nil {
		return nil, err
	}

	out := opts.Output
	if out == nil {
		out = zapcore.Lock(os.Stderr)
	}

	encCfg := encoderConfig(opts)
	enc := zapcore.NewJSONEncoder(encCfg)
	if Format(opts.Env) == FormatConsole {
		enc = zapcore.NewConsoleEncoder(encCfg)
	}
	core := zapcore.NewCore(enc, out, level)

	// Stack traces belong on errors and nowhere else. The dev encoding already
	// prints one for a Warn, which is noise on a platform that logs warnings.
	return zap.New(redactingCore{Core: core},
		zap.ErrorOutput(out),
		zap.AddCaller(),
		zap.AddStacktrace(zapcore.ErrorLevel),
	), nil
}

// encoderConfig returns the encoding for the environment, with timestamps in
// the configured trading timezone.
func encoderConfig(opts Options) zapcore.EncoderConfig {
	cfg := zap.NewProductionEncoderConfig()
	if Format(opts.Env) == FormatConsole {
		cfg = zap.NewDevelopmentEncoderConfig()
		cfg.EncodeLevel = zapcore.CapitalColorLevelEncoder
	}
	// Millisecond precision, and in the trading timezone when one is
	// configured, so a log line and a candle are read against the same clock.
	cfg.EncodeTime = zapcore.TimeEncoderOfLayout(timeLayout)
	if loc := location(opts.Zone); loc != nil {
		cfg.EncodeTime = zoneEncoder(loc)
	}
	cfg.EncodeDuration = zapcore.StringDurationEncoder
	return cfg
}

// timeLayout is the timestamp layout for both encodings.
const timeLayout = "2006-01-02T15:04:05.000Z07:00"

// zoneEncoder renders timestamps in loc.
func zoneEncoder(loc *time.Location) zapcore.TimeEncoder {
	return func(t time.Time, enc zapcore.PrimitiveArrayEncoder) {
		enc.AppendString(t.In(loc).Format(timeLayout))
	}
}

// location resolves a timezone name, returning nil when it cannot be loaded so
// that a missing tzdata on a scratch container is not a reason to refuse to
// start.
func location(zone string) *time.Location {
	if strings.TrimSpace(zone) == "" {
		return nil
	}
	loc, err := time.LoadLocation(zone)
	if err != nil {
		return nil
	}
	return loc
}

// Format returns the log encoding for env: FormatConsole in development,
// FormatJSON everywhere else. An unrecognised environment is treated as
// production, because a stack trace pasted into a chat is a far smaller problem
// than a credential printed to a terminal.
func Format(env string) string {
	switch strings.ToLower(strings.TrimSpace(env)) {
	case "dev", "development", "local", "test":
		return FormatConsole
	default:
		return FormatJSON
	}
}

// ParseLevel maps an app.log_level value onto a zap level. An empty value
// means info, which is the config default.
func ParseLevel(level string) (zapcore.Level, error) {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "", "info":
		return zapcore.InfoLevel, nil
	case "debug":
		return zapcore.DebugLevel, nil
	case "warn", "warning":
		return zapcore.WarnLevel, nil
	case "error":
		return zapcore.ErrorLevel, nil
	default:
		return zapcore.InfoLevel, fmt.Errorf("app.log_level %q must be one of debug, info, warn, error", level)
	}
}
