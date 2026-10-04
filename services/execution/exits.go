package execution

import (
	"context"
	"fmt"

	"github.com/mft/core/contracts"
)

// ExitPolicy derives explicit risk-reducing orders from the current portfolio.
type ExitPolicy interface {
	ExitOrders(context.Context, contracts.Portfolio) ([]contracts.OrderRequest, error)
}

// ExitPolicyFunc adapts a function to ExitPolicy.
type ExitPolicyFunc func(context.Context, contracts.Portfolio) ([]contracts.OrderRequest, error)

func (f ExitPolicyFunc) ExitOrders(ctx context.Context, portfolio contracts.Portfolio) ([]contracts.OrderRequest, error) {
	return f(ctx, portfolio)
}

// SetExitPolicy installs the strategy-owned policy run after each successful
// reconciliation.
func (e *Engine) SetExitPolicy(policy ExitPolicy) {
	e.mu.Lock()
	e.exitPolicy = policy
	e.mu.Unlock()
}

// RunExitPolicy submits explicit SELL requests through the normal durable order path.
func (e *Engine) RunExitPolicy(ctx context.Context) error {
	e.mu.Lock()
	policy := e.exitPolicy
	portfolio := e.portfolioLocked(e.clock())
	e.mu.Unlock()
	if policy == nil {
		return nil
	}
	requests, err := policy.ExitOrders(ctx, portfolio)
	if err != nil {
		return fmt.Errorf("execution exit policy: %w", err)
	}
	for _, req := range requests {
		if req.Side != contracts.SideSell {
			return fmt.Errorf("execution exit policy may only emit SELL orders")
		}
		if req.IdempotencyKey == "" {
			return fmt.Errorf("execution exit policy orders require an idempotency key")
		}
		_, err := e.ExecuteOrder(ctx, req)
		if err != nil {
			if rej, ok := err.(*contracts.Rejection); ok && rej.Code == contracts.ReasonDuplicate {
				continue
			}
			return fmt.Errorf("execution exit order %q: %w", req.IdempotencyKey, err)
		}
	}
	return nil
}
