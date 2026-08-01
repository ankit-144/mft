// Command server runs the MFT ingestion service.
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
