package log

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/config"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// The secrets below are fakes. The point of the test is that they never appear
// in output, whatever the caller does.
const (
	apiSecret   = "kitesecret-7f3a9c1e-do-not-log"
	accessToken = "kiteaccesstoken-2b8d4f6a-do-not-log"
	apiKey      = "kiteapikey-1a2b3c4d-do-not-log"
)

func configWithCredentials() *config.Config {
	cfg := &config.Config{}
	cfg.App.Name = "mft"
	cfg.App.Env = "dev"
	cfg.Broker.APIKey = apiKey
	cfg.Broker.APISecret = apiSecret
	cfg.Broker.AccessToken = accessToken
	cfg.Broker.Instruments = []string{"RELIANCE", "TCS"}
	cfg.Storage.DataDir = "data"
	cfg.Execution.Capital = 1000000
	return cfg
}

// assertNoSecrets fails the test if any fake credential survived.
func assertNoSecrets(t *testing.T, out string) {
	t.Helper()
	for _, secret := range []string{apiSecret, accessToken, apiKey} {
		if strings.Contains(out, secret) {
			t.Errorf("output contains a credential in the clear:\n%s", out)
		}
	}
}

// TestRedactConfigStruct is the requirement: a config struct can be handed to
// zap.Any and the api_secret and access_token still never reach the log.
func TestRedactConfigStruct(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "dev", "debug")

	logger.Info("broker configured", zap.Any("config", configWithCredentials()))

	out := buf.String()
	assertNoSecrets(t, out)
	if !strings.Contains(out, Redacted) {
		t.Errorf("expected %s in the output:\n%s", Redacted, out)
	}
	// Non-sensitive values must survive, or the log stops being useful.
	for _, keep := range []string{"RELIANCE", "data", "1000000", "api_secret", "access_token"} {
		if !strings.Contains(out, keep) {
			t.Errorf("output lost the non-sensitive value %q:\n%s", keep, out)
		}
	}
}

func TestRedactSensitiveKeys(t *testing.T) {
	t.Parallel()
	sensitive := []string{
		"api_secret", "API-SECRET", "apiSecret", "access_token", "AccessToken",
		"password", "passwd", "passphrase", "credentials", "Authorization",
		"set-cookie", "private_key", "apikey", "otp", "token", "auth", "pin",
	}
	for _, key := range sensitive {
		if !IsSensitiveKey(key) {
			t.Errorf("IsSensitiveKey(%q) = false, want true", key)
		}
	}
	harmless := []string{
		"symbol", "side", "quantity", "price", "capital", "data_dir", "env",
		"instrument_token", "InstrumentToken", "idempotency_key", "token_bucket_remaining",
		"public_key", "key_id", "metric_name", "partition_by", "score",
	}
	for _, key := range harmless {
		if IsSensitiveKey(key) {
			t.Errorf("IsSensitiveKey(%q) = true, want false", key)
		}
	}
}

func TestRedactFieldTypes(t *testing.T) {
	t.Parallel()
	type broker struct {
		APIKey      string `yaml:"api_key"`
		AccessToken string `yaml:"access_token"`
		Reconnects  int    `yaml:"reconnects"`
	}
	type order struct {
		Symbol string `json:"symbol"`
		Token  string `json:"token"`
	}

	logger, buf := loggerFor(t, "prod", "debug")
	logger.Info("everything at once",
		zap.String("access_token", accessToken),
		zap.String("endpoint", "https://api.kite.trade/session"),
		zap.Any("headers", map[string]string{"Authorization": "Bearer " + accessToken, "X-App": "mft"}),
		zap.Any("brokers", []broker{{APIKey: apiKey, AccessToken: accessToken, Reconnects: 3}}),
		zap.Any("order", order{Symbol: "RELIANCE", Token: accessToken}),
		zap.Error(fmt.Errorf("POST /orders/4412?access_token=%s: 403", accessToken)),
		zap.Int("retries", 2),
	)

	out := buf.String()
	assertNoSecrets(t, out)
	for _, keep := range []string{"https://api.kite.trade/session", "X-App", "RELIANCE", "403", `"retries":2`} {
		if !strings.Contains(out, keep) {
			t.Errorf("output lost %q:\n%s", keep, out)
		}
	}
}

func TestRedactMessageText(t *testing.T) {
	t.Parallel()
	cases := []string{
		"dial failed with api_secret=" + apiSecret,
		`body {"access_token":"` + accessToken + `","expires_in":86400}`,
		"GET /session?access_token=" + accessToken + "&x=1: unauthorized",
		"Authorization: Bearer " + accessToken,
		"authorization=Basic dXNlcjpwYXNz",
		"jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl rejected",
		"connecting to kite with password: hunter2",
	}
	logger, buf := loggerFor(t, "prod", "debug")
	for _, msg := range cases {
		logger.Error(msg)
	}
	out := buf.String()
	for _, secret := range []string{apiSecret, accessToken, "hunter2", "dXNlcjpwYXNz", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl"} {
		if strings.Contains(out, secret) {
			t.Errorf("message credential %q survived:\n%s", secret, out)
		}
	}
	if !strings.Contains(out, Redacted) {
		t.Errorf("expected %s in the output:\n%s", Redacted, out)
	}
}

// TestRedactIsIdempotent matters because a message can be scrubbed twice, once
// on the way to the encoder and once inside a field.
func TestRedactIsIdempotent(t *testing.T) {
	t.Parallel()
	inputs := []string{
		"token=" + accessToken,
		`{"access_token":"` + accessToken + `"}`,
		"Authorization: Bearer " + accessToken,
		"/session?access_token=" + accessToken + "&x=1",
		"api_secret=" + apiSecret,
		"nothing to redact here",
	}
	for _, in := range inputs {
		once := redactString(in)
		if twice := redactString(once); once != twice {
			t.Errorf("redaction is not idempotent for %q:\nfirst:  %q\nsecond: %q", in, once, twice)
		}
		if strings.Contains(once, "]]") {
			t.Errorf("redaction produced a malformed marker: %q", once)
		}
	}
}

func TestRedactLeavesUsefulValuesAlone(t *testing.T) {
	t.Parallel()
	type tick struct {
		Symbol          string    `json:"symbol"`
		InstrumentToken int64     `json:"instrument_token"`
		Timestamp       time.Time `json:"timestamp"`
	}
	logger, buf := loggerFor(t, "prod", "debug")
	logger.Info("tick",
		zap.Any("tick", tick{
			Symbol:          "RELIANCE",
			InstrumentToken: 256265,
			Timestamp:       time.Date(2026, 9, 29, 10, 31, 0, 0, time.UTC),
		}),
		zap.Duration("aggregator_latency", 1500*time.Microsecond),
	)

	out := buf.String()
	for _, keep := range []string{"RELIANCE", "256265", "2026-09-29T10:31:00Z", "1.5ms"} {
		if !strings.Contains(out, keep) {
			t.Errorf("output lost or mangled %q:\n%s", keep, out)
		}
	}
	if strings.Contains(out, Redacted) {
		t.Errorf("nothing here is a credential, yet the output is redacted:\n%s", out)
	}
}

func TestRedactSurvivesCycles(t *testing.T) {
	t.Parallel()
	type node struct {
		Name  string `json:"name"`
		Self  *node  `json:"self,omitempty"`
		Extra string `json:"extra"`
	}
	n := &node{Name: "root", Extra: "plain"}
	n.Self = n

	logger, buf := loggerFor(t, "prod", "debug")
	logger.Info("cyclic", zap.Any("node", n))

	if !strings.Contains(buf.String(), "root") {
		t.Errorf("cyclic value was dropped entirely:\n%s", buf.String())
	}
}

func TestRedactWithFields(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")
	logger.With(zap.String("access_token", accessToken), zap.String("symbol", "TCS")).
		Info("bound context")

	out := buf.String()
	assertNoSecrets(t, out)
	if !strings.Contains(out, "TCS") {
		t.Errorf("output lost the non-sensitive bound field:\n%s", out)
	}
}

func TestRedactKeepsErrorsUseful(t *testing.T) {
	t.Parallel()
	logger, buf := loggerFor(t, "prod", "debug")
	logger.Error("order failed", zap.Error(errors.New("broker returned 403 for order 4412")))

	out := buf.String()
	if !strings.Contains(out, "403 for order 4412") {
		t.Errorf("error text was lost:\n%s", out)
	}
}

func TestRedactNilAndScalars(t *testing.T) {
	t.Parallel()
	if got := Redact(nil); got != nil {
		t.Errorf("Redact(nil) = %v, want nil", got)
	}
	if got := Redact(42); got != 42 {
		t.Errorf("Redact(42) = %v, want 42", got)
	}
	if got := Redact("plain"); got != "plain" {
		t.Errorf("Redact(plain) = %v, want plain", got)
	}
	if got, ok := Redact([]byte("raw")).([]byte); !ok || string(got) != "raw" {
		t.Errorf("Redact([]byte) = %v, want the bytes untouched", got)
	}
}

// TestRedactingCoreIsInTheChain guards the wiring: a logger built by Build
// always redacts, whatever the caller does.
func TestRedactingCoreIsInTheChain(t *testing.T) {
	t.Parallel()
	buf := &bytes.Buffer{}
	logger, err := Build(Options{Env: "prod", Level: "debug", Output: zapcore.AddSync(buf)})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if _, ok := logger.Core().(redactingCore); !ok {
		t.Fatalf("core is %T, want the redacting core", logger.Core())
	}
	logger.Info("leaking", zap.Any("config", configWithCredentials()))
	assertNoSecrets(t, buf.String())
}
