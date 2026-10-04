// Risk policy interfaces.
package contracts

import (
	"context"
	"time"
)

// Checker validates a signal against risk policy.
type Checker interface {
	Check(ctx context.Context, sig Signal, portfolio Portfolio) error
}

// CheckerFunc adapts a function to the Checker interface.
type CheckerFunc func(ctx context.Context, sig Signal, portfolio Portfolio) error

// Check implements the Checker interface.
func (f CheckerFunc) Check(ctx context.Context, sig Signal, portfolio Portfolio) error {
	return f(ctx, sig, portfolio)
}

// Checks composes an ordered list of checks.
type Checks []Checker

// Check implements the Checker interface by running each check in order.
func (c Checks) Check(ctx context.Context, sig Signal, portfolio Portfolio) error {
	for _, check := range c {
		if check == nil {
			continue
		}
		if err := check.Check(ctx, sig, portfolio); err != nil {
			return err
		}
	}
	return nil
}

// Policy is the configurable risk budget, loaded from config.ExecutionConfig.
type Policy struct {
	Capital          float64
	MaxPositionPct   float64
	MaxOpenPositions int
	MaxDrawdownPct   float64
	DailyLossLimit   float64
	MaxOrderQuantity int
	DebounceTTL      time.Duration
}
