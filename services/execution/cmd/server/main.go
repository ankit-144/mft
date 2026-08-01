// Command server runs the MFT execution & risk service.
package main

import (
	"go.uber.org/fx"

	"github.com/mft/core"
	"github.com/mft/services/execution"
)

func main() {
	fx.New(
		core.Module,
		execution.Module,
	).Run()
}
