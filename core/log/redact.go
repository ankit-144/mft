package log

import (
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Redacted replaces every value that looks like a credential.
const Redacted = "[REDACTED]"

// maxDepth bounds the walk over a redacted value. Config is two levels deep;
// this only exists so that a pathological value cannot turn a log call into a
// hang, because a log call must never be able to stop a trading process.
const maxDepth = 8

// sensitiveSubstrings are matched against a normalised key: lowercase with
// separators removed, so api_secret, apiSecret and API-SECRET all match.
var sensitiveSubstrings = []string{
	"secret", "password", "passwd", "passphrase", "credential",
	"accesstoken", "refreshtoken", "idtoken", "authtoken", "bearer",
	"apikey", "privatekey", "authorization", "cookie", "sessionkey", "otp",
}

// sensitiveExact are keys that carry a credential without containing any of the
// substrings above.
var sensitiveExact = map[string]bool{
	"token":     true,
	"auth":      true,
	"pass":      true,
	"signature": true,
	"pin":       true,
}

// notSensitive wins over every rule above. These names contain a sensitive
// word without carrying a credential, and redacting them costs more than it
// protects: instrument_token is market data, idempotency_key is the thing that
// stops a retried signal placing a second order.
var notSensitive = map[string]bool{
	"instrumenttoken":      true,
	"tokenbucket":          true,
	"tokenbucketremaining": true,
	"idempotencykey":       true,
	"publickey":            true,
	"keyid":                true,
	"cachekey":             true,
	"fluxkvkey":            true,
	"debouncekey":          true,
	"metricname":           true,
	"partitionkey":         true,
	"partitionby":          true,
}

// IsSensitiveKey reports whether a struct field, map key or label name holds a
// credential. Comparison ignores case, underscores, hyphens and spaces.
func IsSensitiveKey(key string) bool {
	normalised := normaliseKey(key)
	if normalised == "" {
		return false
	}
	if notSensitive[normalised] {
		return false
	}
	if sensitiveExact[normalised] {
		return true
	}
	for _, fragment := range sensitiveSubstrings {
		if strings.Contains(normalised, fragment) {
			return true
		}
	}
	return false
}

// normaliseKey lowercases key and drops separators, so every spelling of
// access_token collapses onto one key.
func normaliseKey(key string) string {
	var b strings.Builder
	b.Grow(len(key))
	for _, r := range key {
		switch {
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r + ('a' - 'A'))
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Redact returns a sanitised copy of v: every value under a credential-looking
// key becomes Redacted, and every string is scrubbed for embedded credentials.
//
// It is safe to call on a *config.Config — that is the point. A whole config
// struct can be logged and the api_secret and access_token will not appear:
//
//	logger.Info("starting", zap.Any("config", cfg))
//
// Types that know how to describe themselves (fmt.Stringer, json.Marshaler,
// zapcore.ObjectMarshaler, ...) are passed through untouched, so timestamps,
// durations and hand-written marshalers keep their normal rendering.
//
// The walk uses reflection, so it is meant for configuration and
// request-scoped values, not for a per-tick hot loop.
func Redact(v any) any {
	return redactValue(reflect.ValueOf(v), map[uintptr]bool{}, 0)
}

// redactValue walks v, replacing credential-looking values with Redacted.
func redactValue(v reflect.Value, seen map[uintptr]bool, depth int) any {
	if !v.IsValid() {
		return nil
	}
	if depth > maxDepth {
		return Redacted
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return nil
		}
		if v.Kind() == reflect.Pointer {
			addr := v.Pointer()
			if seen[addr] {
				return "[cycle]"
			}
			seen[addr] = true
			defer delete(seen, addr)
		}
		return redactValue(v.Elem(), seen, depth)
	case reflect.Slice, reflect.Map:
		if v.IsNil() {
			return nil
		}
		if v.Kind() == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8 {
			// A []byte is data, not a list of credentials.
			return v.Interface()
		}
	}

	if isOpaque(v.Type()) {
		return v.Interface()
	}

	switch v.Kind() {
	case reflect.Struct:
		return redactStruct(v, seen, depth)
	case reflect.Map:
		return redactMap(v, seen, depth)
	case reflect.Slice, reflect.Array:
		return redactSlice(v, seen, depth)
	case reflect.String:
		return redactString(v.String())
	default:
		return v.Interface()
	}
}

// redactStruct converts a struct to a map keyed by its wire name, so what
// reaches the log matches what reaches the YAML file.
func redactStruct(v reflect.Value, seen map[uintptr]bool, depth int) map[string]any {
	typ := v.Type()
	out := make(map[string]any, typ.NumField())
	for i := range typ.NumField() {
		field := typ.Field(i)
		if field.PkgPath != "" {
			continue // unexported
		}
		name := fieldName(field)
		if name == "-" {
			continue
		}
		if field.Anonymous && field.Type.Kind() == reflect.Struct {
			for k, val := range redactStruct(v.Field(i), seen, depth+1) {
				out[k] = val
			}
			continue
		}
		if IsSensitiveKey(name) {
			out[name] = Redacted
			continue
		}
		out[name] = redactValue(v.Field(i), seen, depth+1)
	}
	return out
}

// fieldName returns the wire name of a struct field: the json tag, then the
// yaml tag, then the Go name.
func fieldName(field reflect.StructField) string {
	for _, key := range []string{"json", "yaml"} {
		tag := field.Tag.Get(key)
		if tag == "" {
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if name == "" {
			return field.Name
		}
		return name
	}
	return field.Name
}

// redactMap converts a map to a map[string]any, redacting by key.
func redactMap(v reflect.Value, seen map[uintptr]bool, depth int) map[string]any {
	iter := v.MapRange()
	out := make(map[string]any, v.Len())
	for iter.Next() {
		key := fmt.Sprint(iter.Key().Interface())
		if IsSensitiveKey(key) {
			out[key] = Redacted
			continue
		}
		out[key] = redactValue(iter.Value(), seen, depth+1)
	}
	return out
}

// redactSlice converts a slice or array to a slice of redacted values.
func redactSlice(v reflect.Value, seen map[uintptr]bool, depth int) []any {
	out := make([]any, v.Len())
	for i := range v.Len() {
		out[i] = redactValue(v.Index(i), seen, depth+1)
	}
	return out
}

// type assertions for the interfaces redactValue treats as opaque.
var (
	stringerType        = reflect.TypeFor[fmt.Stringer]()
	jsonMarshalerType   = reflect.TypeFor[json.Marshaler]()
	textMarshalerType   = reflect.TypeFor[encoding.TextMarshaler]()
	errorType           = reflect.TypeFor[error]()
	objectMarshalerType = reflect.TypeFor[zapcore.ObjectMarshaler]()
	arrayMarshalerType  = reflect.TypeFor[zapcore.ArrayMarshaler]()
)

// isOpaque reports whether t renders itself, and must therefore not be
// walked into.
func isOpaque(t reflect.Type) bool {
	return t.Implements(stringerType) ||
		t.Implements(jsonMarshalerType) ||
		t.Implements(textMarshalerType) ||
		t.Implements(errorType) ||
		t.Implements(objectMarshalerType) ||
		t.Implements(arrayMarshalerType)
}

// secretPatterns are scrubbed out of every string that reaches the encoder.
// Each pattern captures the context to keep in group 1 and leaves the
// credential as the final, ungrouped part, so one replacement template —
// "${1}" + Redacted — covers all of them.
var secretPatterns = []*regexp.Regexp{
	// Authorization: Bearer …, Proxy-Authorization: Basic …
	regexp.MustCompile(`(?i)((?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]+`),
	// api_secret=…, "access_token":"…", password: …
	regexp.MustCompile(`(?i)([a-z0-9_-]*(?:secret|password|passwd|passphrase|credential|token|apikey|api_key|private_key|privatekey|authorization|cookie|otp)[a-z0-9_-]*["']?\s*[:=]\s*["']?)[^"'\s,;&}\[\])]+`),
	// ?access_token=…&next=value
	regexp.MustCompile(`(?i)([?&][a-z0-9_-]*(?:token|secret|apikey|api_key|auth|key)[a-z0-9_-]*=)[^&\s\[]+`),
}

// tokenPattern matches a bare JWT, which is a credential in any context.
var tokenPattern = regexp.MustCompile(`\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*`)

// redactString removes credential-looking substrings from s.
func redactString(s string) string {
	if !strings.ContainsAny(s, ":=&") && !strings.Contains(s, "eyJ") {
		return s
	}
	for _, re := range secretPatterns {
		// The credential class excludes "[", so an already redacted value
		// cannot be matched again: redaction is idempotent by construction.
		s = re.ReplaceAllString(s, "${1}"+Redacted)
	}
	return tokenPattern.ReplaceAllString(s, Redacted)
}

// redactingCore wraps a zapcore.Core and scrubs everything on its way to the
// encoder. It is the last line of defence: even a caller that hands it a whole
// config struct, or a broker error that echoes the request URL, cannot put a
// credential on stdout.
type redactingCore struct {
	zapcore.Core
}

// With returns a core with the accumulated fields already redacted.
func (c redactingCore) With(fields []zapcore.Field) zapcore.Core {
	return redactingCore{Core: c.Core.With(redactFields(fields))}
}

// Check is overridden so that entries reaching the encoder go through Write
// below; the embedded implementation would hand the original fields straight to
// the wrapped core and skip redaction.
func (c redactingCore) Check(ent zapcore.Entry, ce *zapcore.CheckedEntry) *zapcore.CheckedEntry {
	if c.Enabled(ent.Level) {
		return ce.AddCore(ent, c)
	}
	return ce
}

// Write redacts the message and every field before delegating.
func (c redactingCore) Write(ent zapcore.Entry, fields []zapcore.Field) error {
	ent.Message = redactString(ent.Message)
	return c.Core.Write(ent, redactFields(fields))
}

// redactFields returns a copy of fields with credentials removed.
func redactFields(fields []zapcore.Field) []zapcore.Field {
	if len(fields) == 0 {
		return fields
	}
	out := make([]zapcore.Field, len(fields))
	for i, f := range fields {
		out[i] = redactField(f)
	}
	return out
}

// redactField redacts one field, by key first and then by content.
func redactField(f zapcore.Field) zapcore.Field {
	if IsSensitiveKey(f.Key) {
		// Whatever the type, the value is gone.
		return zap.String(f.Key, Redacted)
	}
	switch f.Type {
	case zapcore.StringType, zapcore.ByteStringType:
		return zap.String(f.Key, redactString(f.String))
	case zapcore.ErrorType:
		// A broker error can carry the request URL, which carries the token.
		if err, ok := f.Interface.(error); ok {
			return zap.String(f.Key, redactString(err.Error()))
		}
		return f
	case zapcore.ReflectType, zapcore.ObjectMarshalerType, zapcore.ArrayMarshalerType:
		if f.Interface == nil || isOpaque(reflect.TypeOf(f.Interface)) {
			return f
		}
		return zap.Any(f.Key, Redact(f.Interface))
	default:
		return f
	}
}
