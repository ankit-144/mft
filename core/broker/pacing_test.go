package broker

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/mft/core/config"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return fn(req) }

func TestKitePacesEachRESTCategoryIndependently(t *testing.T) {
	k, err := NewKiteFromConfig(configForPacerTest())
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	var delays []time.Duration
	k.pacer.now = func() time.Time { return now }
	k.pacer.sleep = func(ctx context.Context, delay time.Duration) error {
		delays = append(delays, delay)
		now = now.Add(delay)
		return ctx.Err()
	}
	k.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}")), Header: make(http.Header)}, nil
	})}
	k.endpoints.httpBase = "https://kite.test"

	for _, endpoint := range []string{
		"/quote/ltp", "/quote/ltp",
		"/instruments/historical/123/minute", "/instruments/historical/123/minute",
		"/orders", "/orders",
		"/portfolio/positions", "/portfolio/positions",
	} {
		if _, err := k.do(context.Background(), http.MethodGet, k.endpoints.httpBase+endpoint, nil, 1024); err != nil {
			t.Fatalf("do %s: %v", endpoint, err)
		}
	}
	want := []time.Duration{time.Second, time.Second / 3, 100 * time.Millisecond, 100 * time.Millisecond}
	if len(delays) != len(want) {
		t.Fatalf("pacing delays = %v, want %v", delays, want)
	}
	for i := range want {
		if delays[i] != want[i] {
			t.Fatalf("pacing delays = %v, want %v", delays, want)
		}
	}
}

func TestRESTPacerCancellationDoesNotSendRequest(t *testing.T) {
	k, err := NewKiteFromConfig(configForPacerTest())
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	k.pacer.sleep = func(ctx context.Context, _ time.Duration) error {
		close(entered)
		<-ctx.Done()
		return ctx.Err()
	}
	k.pacer.next[RESTQuote] = time.Now().Add(time.Hour)
	requests := 0
	k.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		requests++
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}")), Header: make(http.Header)}, nil
	})}
	k.endpoints.httpBase = "https://kite.test"
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := k.do(ctx, http.MethodGet, k.endpoints.quoteURL(), nil, 1024)
		done <- err
	}()
	<-entered
	cancel()
	if err := <-done; err != context.Canceled {
		t.Fatalf("do error = %v, want context canceled", err)
	}
	if requests != 0 {
		t.Fatalf("requests sent = %d, want 0", requests)
	}
}

func configForPacerTest() config.BrokerConfig {
	return config.BrokerConfig{APIKey: "key", AccessToken: "token"}
}
