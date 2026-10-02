// Command server runs the MFT ingestion service.
//
// The graph is core.Module plus ingestion.Module. core.Module supplies the
// broker stream, the fluxKV cache, the tick *storage.Writer and the metrics
// server; ingestion.Module supplies the *storage.CandleWriter the pipeline
// needs to persist completed bars, along with the pipeline's own lifecycle
// hooks. Both writers are therefore present without core/fx.go — which is not
// this component's to edit — knowing anything about candles.
package main

import (
	"go.uber.org/fx"

	"github.com/mft/core"
	"github.com/mft/services/ingestion"
)

func main() {
	fx.New(
		core.Module,
		ingestion.Module,
	).Run()
}
