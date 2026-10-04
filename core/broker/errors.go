package broker

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Sentinel errors returned by the broker connectors.
var (
	ErrAuth = errors.New("broker: authentication failed")

	ErrRateLimit = errors.New("broker: rate limited")

	ErrInstrumentNotFound = errors.New("broker: instrument not found")

	ErrInvalidOrder = errors.New("broker: invalid order")

	ErrOrderNotFound = errors.New("broker: order not found")

	ErrUnavailable = errors.New("broker: unavailable")
)

// kiteError is the Kite Connect error envelope:
type kiteError struct {
	Status  int
	Code    string
	Message string
}

// kiteErrorEnvelope is the primary Kite error shape.
type kiteErrorEnvelope struct {
	Status string      `json:"status"`
	Errors []kiteError `json:"errors"`
}

// kiteErrorLegacy is the alternative single-error shape.
type kiteErrorLegacy struct {
	Error kiteError `json:"error"`
}

// Error implements the error interface.
func (e *kiteError) Error() string {
	var b strings.Builder
	b.WriteString("kite: http ")
	fmt.Fprintf(&b, "%d", e.Status)
	if e.Code != "" {
		b.WriteString(" ")
		b.WriteString(e.Code)
	}
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	return b.String()
}

// Unwrap maps the HTTP status and Kite error code onto a sentinel error so errors.Is
// works across the transport boundary.
func (e *kiteError) Unwrap() error { return classifyStatus(e.Status, e.Code, e.Message) }

// classifyStatus picks the sentinel error for a Kite failure.
func classifyStatus(status int, code, message string) error {
	lower := strings.ToLower(code + " " + message)

	switch {
	case status == 401 || status == 403:
		return ErrAuth
	case status == 429 || code == "429" || strings.Contains(lower, "too many request"):
		return ErrRateLimit
	case status == 404:
		return ErrOrderNotFound
	}

	switch {
	case strings.Contains(lower, "invalid_token"),
		strings.Contains(lower, "invalid_api_key"),
		strings.Contains(lower, "invalid access token"),
		strings.Contains(lower, "incorrect api_key"),
		strings.Contains(lower, "session expired"):
		return ErrAuth
	case strings.Contains(lower, "instrument"),
		strings.Contains(lower, "tradingsymbol"),
		strings.Contains(lower, "unknown symbol"):
		return ErrInstrumentNotFound
	}

	switch {
	case status >= 500:
		return ErrUnavailable
	case status == 400 || status == 405 || status == 409:
		return ErrInvalidOrder
	}
	return ErrUnavailable
}

// parseKiteError decodes a non-2xx Kite response body.
func parseKiteError(status int, body []byte) *kiteError {
	var env kiteErrorEnvelope
	if err := json.Unmarshal(body, &env); err == nil && len(env.Errors) > 0 {
		first := env.Errors[0]
		return &kiteError{Status: status, Code: first.Code, Message: first.Message}
	}

	var legacy kiteErrorLegacy
	if err := json.Unmarshal(body, &legacy); err == nil && (legacy.Error.Code != "" || legacy.Error.Message != "") {
		return &kiteError{Status: status, Code: legacy.Error.Code, Message: legacy.Error.Message}
	}

	return &kiteError{Status: status, Message: strings.TrimSpace(truncate(string(body), 200))}
}

// truncate shortens s to at most n bytes.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
