package broker

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Sentinel errors returned by the broker connectors. Callers distinguish
// failure modes with errors.Is; every error the connectors return wraps one
// of these, so a switch on the sentinel is always exhaustive.
var (
	// ErrAuth means the credentials were missing, expired or rejected. The
	// access token is short lived and must be refreshed out of band.
	ErrAuth = errors.New("broker: authentication failed")
	// ErrRateLimit means Kite throttled the request. Retry after a backoff;
	// the REST API allows roughly three requests per second.
	ErrRateLimit = errors.New("broker: rate limited")
	// ErrInstrumentNotFound means a symbol could not be resolved to an
	// instrument token, either from the instrument dump or from a live
	// subscription.
	ErrInstrumentNotFound = errors.New("broker: instrument not found")
	// ErrInvalidOrder means the request was rejected by local validation or
	// by Kite as malformed (bad side, non-positive quantity, bad lot size,
	// limit order without a price).
	ErrInvalidOrder = errors.New("broker: invalid order")
	// ErrOrderNotFound means Kite does not know the referenced order id.
	ErrOrderNotFound = errors.New("broker: order not found")
	// ErrUnavailable means the transport failed or Kite returned a server
	// side error. Retrying is reasonable.
	ErrUnavailable = errors.New("broker: unavailable")
)

// kiteError is the Kite Connect error envelope:
//
//	{"status":"error","errors":[{"error_code":"invalid_token","message":"..."}]}
//
// Kite also sometimes answers with {"error":{"code":...,"message":...}}, so
// both shapes are decoded. Unwrap maps the response onto one of the sentinel
// errors, which is how callers tell a dead token from a throttled one.
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

// Unwrap maps the HTTP status and Kite error code onto a sentinel error so
// errors.Is works across the transport boundary.
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
