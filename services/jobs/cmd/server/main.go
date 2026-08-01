// Command server runs the MFT background jobs service.
package main

import (
	"go.uber.org/fx"

	"github.com/mft/core"
	"github.com/mft/services/jobs"
)

func main() {
	fx.New(
		core.Module,
		jobs.Module,
	).Run()
}
