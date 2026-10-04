package execution

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/mft/core/contracts"
	"github.com/mft/services/execution/risk"
)

const journalVersion = 1

// OrderRecord is the durable state for one caller idempotency key.
type OrderRecord struct {
	Key              string                 `json:"key"`
	Tag              string                 `json:"tag"`
	Request          contracts.OrderRequest `json:"request"`
	AsOf             time.Time              `json:"as_of"`
	OrderID          string                 `json:"order_id,omitempty"`
	Status           string                 `json:"status"`
	FilledQuantity   int                    `json:"filled_quantity"`
	AppliedNotional  float64                `json:"applied_notional"`
	RiskPrice        float64                `json:"risk_price"`
	AverageFillPrice float64                `json:"average_fill_price,omitempty"`
	CreatedAt        time.Time              `json:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at"`
	LastError        string                 `json:"last_error,omitempty"`
}

// journalState is replaced atomically so claims and accounting survive restart.
type journalState struct {
	Version     int                    `json:"version"`
	Initialized bool                   `json:"initialized"`
	Book        risk.BookState         `json:"book"`
	Orders      map[string]OrderRecord `json:"orders"`
}

type journal struct {
	path string
	lock *os.File
}

func openJournal(path string) (*journal, journalState, error) {
	if path == "" {
		return nil, journalState{}, fmt.Errorf("execution: journal path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, journalState{}, fmt.Errorf("create journal directory: %w", err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, journalState{}, fmt.Errorf("open execution journal lock: %w", err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = lock.Close()
		return nil, journalState{}, fmt.Errorf("execution journal is already in use: %w", err)
	}
	j := &journal{path: path, lock: lock}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return j, journalState{Version: journalVersion, Orders: make(map[string]OrderRecord)}, nil
	}
	if err != nil {
		_ = j.Close()
		return nil, journalState{}, fmt.Errorf("open execution journal: %w", err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var state journalState
	if err := dec.Decode(&state); err != nil {
		_ = j.Close()
		return nil, journalState{}, fmt.Errorf("decode execution journal: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		_ = j.Close()
		return nil, journalState{}, fmt.Errorf("execution journal has trailing data")
	}
	if state.Version != journalVersion {
		_ = j.Close()
		return nil, journalState{}, fmt.Errorf("unsupported execution journal version %d", state.Version)
	}
	if state.Orders == nil {
		state.Orders = make(map[string]OrderRecord)
	}
	if err := validateJournalState(state); err != nil {
		_ = j.Close()
		return nil, journalState{}, err
	}
	return j, state, nil
}

func validateJournalState(state journalState) error {
	if !finite(state.Book.Cash) || !finite(state.Book.RealisedPnL) || !finite(state.Book.PeakEquity) {
		return fmt.Errorf("execution journal contains non-finite accounting values")
	}
	for key, rec := range state.Orders {
		if key == "" || rec.Key != key || rec.Tag != orderTag(key) || rec.Request.IdempotencyKey != key ||
			rec.Request.Symbol == "" || (rec.Request.Side != contracts.SideBuy && rec.Request.Side != contracts.SideSell) ||
			rec.Request.Quantity < 1 || rec.FilledQuantity < 0 || rec.FilledQuantity > rec.Request.Quantity ||
			!finite(rec.RiskPrice) || rec.RiskPrice <= 0 || !finite(rec.AppliedNotional) || rec.AppliedNotional < 0 ||
			!finite(rec.AverageFillPrice) || rec.AverageFillPrice < 0 {
			return fmt.Errorf("execution journal contains invalid order record %q", key)
		}
		switch rec.Status {
		case contracts.OrderStatusSubmitting, contracts.OrderStatusUnknown, contracts.OrderStatusOpen,
			contracts.OrderStatusPartial, contracts.OrderStatusFilled, contracts.OrderStatusCancelled,
			contracts.OrderStatusRejected, contracts.OrderStatusPaperFilled:
		default:
			return fmt.Errorf("execution journal contains unknown order status %q", rec.Status)
		}
	}
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func (j *journal) Save(state journalState) error {
	state.Version = journalVersion
	body, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("encode execution journal: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(j.path), ".execution-state-*.tmp")
	if err != nil {
		return fmt.Errorf("create execution journal temp: %w", err)
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("secure execution journal: %w", err)
	}
	if _, err := tmp.Write(body); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write execution journal: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync execution journal: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close execution journal temp: %w", err)
	}
	if err := os.Rename(name, j.path); err != nil {
		return fmt.Errorf("replace execution journal: %w", err)
	}
	dir, err := os.Open(filepath.Dir(j.path))
	if err != nil {
		return fmt.Errorf("open execution journal directory: %w", err)
	}
	defer dir.Close()
	if err := dir.Sync(); err != nil {
		return fmt.Errorf("sync execution journal directory: %w", err)
	}
	return nil
}

func (j *journal) Close() error {
	if j == nil || j.lock == nil {
		return nil
	}
	unlockErr := syscall.Flock(int(j.lock.Fd()), syscall.LOCK_UN)
	closeErr := j.lock.Close()
	if unlockErr != nil {
		return unlockErr
	}
	return closeErr
}
