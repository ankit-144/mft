//go:build tools

// Package tools pins dependencies that are not yet imported by any package,
// so that `go mod tidy` does not remove them.
//
// This exists so that the parallel component branches do not each have to
// edit core/go.mod to add a shared dependency, which would produce a merge
// conflict in every one of them.
package tools

import (
	_ "github.com/gorilla/websocket"
)
