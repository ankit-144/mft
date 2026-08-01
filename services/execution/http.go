package execution

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/mft/core/config"
	"go.uber.org/fx"
	"go.uber.org/zap"
)

// StartHTTPServer exposes the execution engine over HTTP so the inference
// engine (Python) can submit trade signals.
func StartHTTPServer(lc fx.Lifecycle, engine *Engine, cfg *config.Config, log *zap.Logger) {
	srv := &http.Server{
		Addr:              cfg.Execution.Addr,
		ReadHeaderTimeout: 5 * time.Second,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/orders", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Symbol   string  `json:"symbol"`
			Side     string  `json:"side"`
			Quantity int     `json:"quantity"`
			Price    float64 `json:"price"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		orderID, err := engine.Execute(r.Context(), req.Symbol, req.Side, req.Quantity, req.Price)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"order_id": orderID})
	})
	srv.Handler = mux

	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			go func() {
				log.Info("execution HTTP server listening", zap.String("addr", cfg.Execution.Addr))
				if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
					log.Error("execution HTTP server error", zap.Error(err))
				}
			}()
			return nil
		},
		OnStop: func(ctx context.Context) error {
			return srv.Shutdown(context.Background())
		},
	})
}

// RegisterEngine ensures the engine is constructed at startup.
func RegisterEngine(*Engine) {}
