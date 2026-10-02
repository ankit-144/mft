package log

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// RequestIDHeader carries the correlation id. Inference POSTs a signal to
// execution and the risk engine rejects it; without a shared id you are left
// grepping two log streams for the same minute close. The header is read from
// an inbound request when present and always echoed on the response.
const RequestIDHeader = "X-Request-Id"

// correlationKey is the private context key for the correlation id.
type correlationKey struct{}

// WithCorrelationID returns a copy of ctx carrying id as the correlation id.
func WithCorrelationID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, correlationKey{}, id)
}

// CorrelationIDFromContext returns the correlation id in ctx, or "" when the
// request did not go through Middleware.
func CorrelationIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(correlationKey{}).(string)
	return id
}

// NewCorrelationID returns a fresh 128-bit random correlation id. A request id
// that is guessable is a request id that can be forged in a log line.
func NewCorrelationID() string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		// crypto/rand does not fail on any platform we support; if it ever
		// does, an id is still better than a panic inside a request handler.
		return "corr-" + time.Now().UTC().Format("20060102T150405.000000000")
	}
	return hex.EncodeToString(buf[:])
}

// With returns a logger that stamps every entry with the correlation id. It
// keeps the id attached through a call chain without threading it by hand.
func With(logger *zap.Logger, id string) *zap.Logger {
	if id == "" {
		return logger
	}
	return logger.With(zap.String("correlation_id", id))
}

// WithContext returns a logger stamped with the correlation id in ctx.
func WithContext(ctx context.Context, logger *zap.Logger) *zap.Logger {
	return With(logger, CorrelationIDFromContext(ctx))
}

// Middleware returns HTTP middleware that assigns or propagates a correlation
// id, attaches it to the request context, echoes it in the response header, and
// logs exactly one line per request at info (4xx, 5xx) or debug (the rest).
//
// The URI is redacted, so a token in a query string never reaches the log.
func Middleware(logger *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := r.Header.Get(RequestIDHeader)
			if id == "" {
				id = NewCorrelationID()
			}
			w.Header().Set(RequestIDHeader, id)
			ctx := WithCorrelationID(r.Context(), id)
			r = r.WithContext(ctx)

			start := time.Now()
			rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
			next.ServeHTTP(rec, r)
			elapsed := time.Since(start)

			fields := []zap.Field{
				zap.String("correlation_id", id),
				zap.String("method", r.Method),
				zap.String("uri", redactString(r.URL.RequestURI())),
				zap.Int("status", rec.status),
				zap.Int("bytes", rec.written),
				zap.Duration("duration", elapsed),
				zap.String("remote_addr", clientIP(r.RemoteAddr)),
			}
			switch {
			case rec.status >= 500:
				With(logger, id).Error("http request", fields...)
			case rec.status >= 400:
				With(logger, id).Warn("http request", fields...)
			default:
				With(logger, id).Debug("http request", fields...)
			}
		})
	}
}

// statusRecorder remembers the status code and size of a response.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written int
}

// WriteHeader records the status code of the first write.
func (s *statusRecorder) WriteHeader(status int) {
	s.status = status
	s.ResponseWriter.WriteHeader(status)
}

// Write records the size of the response body.
func (s *statusRecorder) Write(b []byte) (int, error) {
	n, err := s.ResponseWriter.Write(b)
	s.written += n
	return n, err
}

// clientIP trims the port from a remote address, tolerating the empty and
// malformed values a test server produces.
func clientIP(remote string) string {
	host, _, err := net.SplitHostPort(remote)
	if err != nil {
		return remote
	}
	return host
}
