// Package log provides a shared zap logger constructor for FX.
package log

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// New builds a production or development zap logger based on the environment.
func New(env string) (*zap.Logger, error) {
	var cfg zap.Config
	if env == "development" || env == "dev" {
		cfg = zap.NewDevelopmentConfig()
	} else {
		cfg = zap.NewProductionConfig()
	}
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	return cfg.Build()
}
