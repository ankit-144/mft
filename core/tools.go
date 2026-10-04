//go:build tools

// Package tools pins dependencies that are not yet imported by any package, so that `go
// mod tidy` does not remove them.
package tools

import (
	_ "github.com/gorilla/websocket"
)
