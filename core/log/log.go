// Package log provides the shared zap logger for every MFT service.
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
	FormatConsole = "console"

	FormatJSON = "json"
)

// Options configures the shared logger.
type Options struct {
	// Env is the deployment environment: dev, staging or prod.
	Env string
	// Level is app.log_level: debug, info, warn or error.
	Level string
	// Zone is the app.timezone the timestamps are rendered in, e.g.
	Zone string
	// Output receives the log lines.
	Output zapcore.WriteSyncer
}

// New builds the shared logger from the shared config schema.
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

	return zap.New(redactingCore{Core: core},
		zap.ErrorOutput(out),
		zap.AddCaller(),
		zap.AddStacktrace(zapcore.ErrorLevel),
	), nil
}

// encoderConfig returns the encoding for the environment, with timestamps in the
// configured trading timezone.
func encoderConfig(opts Options) zapcore.EncoderConfig {
	cfg := zap.NewProductionEncoderConfig()
	if Format(opts.Env) == FormatConsole {
		cfg = zap.NewDevelopmentEncoderConfig()
		cfg.EncodeLevel = zapcore.CapitalColorLevelEncoder
	}

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

// location resolves a timezone name, returning nil when it cannot be loaded so that a
// missing tzdata on a scratch container is not a reason to refuse to start.
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

// Format returns the log encoding for env: FormatConsole in development, FormatJSON
// everywhere else.
func Format(env string) string {
	switch strings.ToLower(strings.TrimSpace(env)) {
	case "dev", "development", "local", "test":
		return FormatConsole
	default:
		return FormatJSON
	}
}

// ParseLevel maps an app.log_level value onto a zap level.
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
