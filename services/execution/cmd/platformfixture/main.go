// Command platformfixture exposes production components over test-only JSON lines.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http/httptest"
	"os"
	"path/filepath"
	"time"

	"github.com/mft/core/config"
	"github.com/mft/core/contracts"
	"github.com/mft/core/features"
	"github.com/mft/core/fluxkv"
	"github.com/mft/core/storage"
	"github.com/mft/services/execution"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

type forbiddenBroker struct{}

func (forbiddenBroker) PlaceOrder(context.Context, contracts.OrderRequest) (string, error) {
	panic("fixture attempted a live order")
}
func (forbiddenBroker) CancelOrder(context.Context, string) error {
	panic("fixture attempted broker cancellation")
}
func (forbiddenBroker) GetPositions(context.Context) ([]contracts.Position, error) {
	panic("fixture attempted broker I/O")
}

type request struct {
	Method string          `json:"method"`
	Path   string          `json:"path"`
	Body   json.RawMessage `json:"body"`
	Token  *string         `json:"token,omitempty"`
}

type response struct {
	Status int             `json:"status"`
	Body   json.RawMessage `json:"body"`
}

func candles() []contracts.Candle {
	start := time.Date(2026, 8, 3, 4, 0, 0, 0, time.UTC)
	rows := make([]contracts.Candle, 200)
	for index := range rows {
		close := 100 + float64(index)*0.01 + math.Sin(float64(index)*0.2)*0.1
		rows[index] = contracts.Candle{Symbol: "RELIANCE", Timestamp: start.Add(time.Duration(index) * time.Minute),
			Open: close - 0.02, High: close + 0.1, Low: close - 0.1, Close: close, Volume: int64(100 + index)}
	}
	return rows
}

func run(dir string) error {
	rows := candles()
	writer := storage.NewCandleWriter(filepath.Join(dir, "candles"), 10000)
	defer writer.Close()
	for _, row := range rows {
		if err := writer.Append(storage.NewCandle(row)); err != nil {
			return err
		}
	}
	if err := writer.Flush(context.Background()); err != nil {
		return err
	}
	zone, err := time.LoadLocation("Asia/Kolkata")
	if err != nil {
		return err
	}
	table, err := features.NewBuilder(zone).Build(rows)
	if err != nil {
		return err
	}
	cfg := &config.Config{Execution: config.ExecutionConfig{PaperTrading: true, Capital: 1_000_000,
		JournalPath: filepath.Join(dir, "execution", "state.json"), MaxSignalAgeSeconds: 120,
		MaxPositionPct: 10, MaxOpenPositions: 10, MaxDrawdownPct: 5, DailyLossLimit: 25_000,
		MaxOrderQuantity: 500, MaxPendingOrders: 64, DebounceTTLSeconds: 1, APIToken: "fixture-token"}}
	if err := cfg.Validate(); err != nil {
		return err
	}
	reg := prometheus.NewRegistry()
	clock := func() time.Time { return rows[len(rows)-1].Timestamp.Add(time.Minute) }
	engine, err := execution.NewEngineWithClock(forbiddenBroker{}, fluxkv.New(), cfg, reg, zap.NewNop(), clock)
	if err != nil {
		return err
	}
	defer engine.CloseContext(context.Background())
	if err := engine.Initialize(context.Background()); err != nil {
		return err
	}
	handler := execution.NewHandler(engine, forbiddenBroker{}, reg, zap.NewNop(), &cfg.Execution)
	encoder := json.NewEncoder(os.Stdout)
	if err := encoder.Encode(map[string]any{"ready": true, "columns": table.Columns, "features": table.Rows,
		"as_of": rows[len(rows)-1].Timestamp, "price": rows[len(rows)-1].Close}); err != nil {
		return err
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4096), 65536)
	for scanner.Scan() {
		var input request
		if err := json.Unmarshal(scanner.Bytes(), &input); err != nil {
			return err
		}
		req := httptest.NewRequest(input.Method, input.Path, bytes.NewReader(input.Body))
		token := "fixture-token"
		if input.Token != nil {
			token = *input.Token
		}
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, req)
		if err := encoder.Encode(response{Status: recorder.Code, Body: recorder.Body.Bytes()}); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func main() {
	dir := flag.String("data-dir", "", "required temporary fixture directory")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "--data-dir is required")
		os.Exit(2)
	}
	if err := run(*dir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
