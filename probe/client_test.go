package probe

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestSendParsesUsage(t *testing.T) {
	tests := []struct {
		name       string
		protocol   Protocol
		baseSuffix string
		wantPath   string
		body       string
		want       Response
	}{
		{
			name:     "openai counts cached tokens inside prompt_tokens",
			protocol: ProtocolOpenAI,
			wantPath: "/v1/chat/completions",
			body:     `{"id":"chatcmpl-1","model":"m","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":4515,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":3611}}}`,
			want:     Response{HTTPStatus: 200, ID: "chatcmpl-1", Model: "m", Text: "ok", Usage: Usage{InputTokens: 4515, OutputTokens: 2, CachedTokens: 3611}},
		},
		{
			name:       "base URL that already ends in v1 is not doubled",
			protocol:   ProtocolOpenAI,
			baseSuffix: "/v1",
			wantPath:   "/v1/chat/completions",
			body:       `{"id":"x","model":"m","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":1}}`,
			want:       Response{HTTPStatus: 200, ID: "x", Model: "m", Text: "ok", Usage: Usage{InputTokens: 9, OutputTokens: 1}},
		},
		{
			name:     "anthropic input total includes cache read and creation",
			protocol: ProtocolAnthropic,
			wantPath: "/v1/messages",
			body:     `{"id":"msg_01MzmDFUhapVJeHMhAUX2ZwC","model":"claude","content":[{"type":"thinking","thinking":"t"},{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":3,"cache_creation_input_tokens":100,"cache_read_input_tokens":4000}}`,
			want:     Response{HTTPStatus: 200, ID: "msg_01MzmDFUhapVJeHMhAUX2ZwC", Model: "claude", Text: "ok", Usage: Usage{InputTokens: 4110, OutputTokens: 3, CachedTokens: 4000}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotPath string
			var gotBody map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				raw, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(raw, &gotBody)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()
			ep := Endpoint{BaseURL: srv.URL + tt.baseSuffix, APIKey: "k", Model: "m", Protocol: tt.protocol}
			got, err := ep.Send(context.Background(), Request{System: "s", User: "u", MaxTokens: 8})
			if err != nil {
				t.Fatalf("Send: %v", err)
			}
			if gotPath != tt.wantPath {
				t.Errorf("path = %q, want %q", gotPath, tt.wantPath)
			}
			if gotBody["model"] != "m" {
				t.Errorf("request model = %v, want m", gotBody["model"])
			}
			got.Latency = 0
			if got != tt.want {
				t.Errorf("response = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestSendReturnsHTTPErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate"}`))
	}))
	defer srv.Close()
	got, err := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m"}.Send(context.Background(), Request{User: "u"})
	if err == nil {
		t.Fatal("expected error for HTTP 429")
	}
	if got.HTTPStatus != http.StatusTooManyRequests {
		t.Errorf("HTTPStatus = %d, want 429", got.HTTPStatus)
	}
}

// statusServer answers each request with the next status from statuses (the
// last one repeats) and a fixed Retry-After, and counts the requests.
func statusServer(t *testing.T, retryAfter string, statuses ...int) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := int(calls.Add(1)) - 1
		status := statuses[min(i, len(statuses)-1)]
		if retryAfter != "" {
			w.Header().Set("Retry-After", retryAfter)
		}
		w.WriteHeader(status)
		if status == http.StatusOK {
			_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":1}}`))
			return
		}
		_, _ = w.Write([]byte(`{"error":"busy"}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestSendRetries(t *testing.T) {
	tests := []struct {
		name        string
		retries     int
		retryAfter  string
		statuses    []int
		wantCalls   int64
		wantStatus  int
		wantRetries int
		wantErr     string
	}{
		{"429 then success yields a normal response", 2, "0", []int{429, 200}, 2, 200, 1, ""},
		{"every retryable 5xx is retried", 4, "0", []int{500, 502, 503, 504, 200}, 5, 200, 4, ""},
		{"exhausted retries keep the last status", 2, "0", []int{503, 502, 429}, 3, 429, 2, "probe: HTTP 429"},
		{"zero retries sends once", 0, "0", []int{503, 200}, 1, 503, 0, "probe: HTTP 503"},
		{"400 is not retried", 2, "0", []int{400, 200}, 1, 400, 0, "probe: HTTP 400"},
		{"401 is not retried", 2, "0", []int{401, 200}, 1, 401, 0, "probe: HTTP 401"},
		{"404 is not retried", 2, "0", []int{404, 200}, 1, 404, 0, "probe: HTTP 404"},
		{"501 is not retried", 2, "0", []int{501, 200}, 1, 501, 0, "probe: HTTP 501"},
		{"no Retry-After backs off up to max wait", 1, "", []int{503, 200}, 2, 200, 1, ""},
		{"Retry-After beyond max wait stops retrying", 2, "120", []int{429, 200}, 1, 429, 0, "Retry-After 2m0s exceeds max wait"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, calls := statusServer(t, tt.retryAfter, tt.statuses...)
			ep := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Retries: tt.retries, MaxWait: 20 * time.Millisecond}
			got, err := ep.Send(context.Background(), Request{User: "u"})
			if tt.wantErr == "" && err != nil {
				t.Fatalf("Send: %v", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("error = %v, want it to contain %q", err, tt.wantErr)
			}
			if calls.Load() != tt.wantCalls {
				t.Errorf("requests = %d, want %d", calls.Load(), tt.wantCalls)
			}
			if got.HTTPStatus != tt.wantStatus || got.Retries != tt.wantRetries {
				t.Errorf("status %d after %d retries, want %d after %d", got.HTTPStatus, got.Retries, tt.wantStatus, tt.wantRetries)
			}
			if tt.wantErr == "" && got.Text != "ok" {
				t.Errorf("text = %q, want ok", got.Text)
			}
		})
	}
}

func TestSendWaitsForRetryAfter(t *testing.T) {
	srv, calls := statusServer(t, "1", 429, 200)
	start := time.Now()
	got, err := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Retries: 1}.Send(context.Background(), Request{User: "u"})
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if elapsed := time.Since(start); elapsed < time.Second {
		t.Errorf("retried after %s, want at least the 1s Retry-After", elapsed)
	}
	if calls.Load() != 2 || got.Retries != 1 {
		t.Errorf("requests = %d, retries = %d, want 2 and 1", calls.Load(), got.Retries)
	}
}

func TestSendRetriesTimeouts(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			_, _ = io.Copy(io.Discard, r.Body)
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
			return
		}
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":1}}`))
	}))
	defer srv.Close()
	ep := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Retries: 1, MaxWait: time.Millisecond, HTTP: &http.Client{Timeout: 200 * time.Millisecond}}
	got, err := ep.Send(context.Background(), Request{User: "u"})
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if calls.Load() != 2 || got.Retries != 1 || got.HTTPStatus != 200 {
		t.Errorf("requests = %d, retries = %d, status = %d, want 2, 1, 200", calls.Load(), got.Retries, got.HTTPStatus)
	}
}

func TestSendStopsRetryingWhenCancelled(t *testing.T) {
	srv, calls := statusServer(t, "10", 503)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	got, err := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Retries: 2}.Send(ctx, Request{User: "u"})
	if err == nil || !strings.Contains(err.Error(), "retry interrupted") {
		t.Fatalf("error = %v, want an interrupted retry", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("cancel took %s to stop the wait", elapsed)
	}
	if calls.Load() != 1 || got.HTTPStatus != 503 {
		t.Errorf("requests = %d, status = %d, want 1 and 503", calls.Load(), got.HTTPStatus)
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 11, 21, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		in   string
		want time.Duration
	}{
		{"", -1},
		{"0", 0},
		{" 7 ", 7 * time.Second},
		{"-3", -1},
		{"soon", -1},
		{"99999999999", 24 * time.Hour},
		{now.Add(90 * time.Second).Format(http.TimeFormat), 90 * time.Second},
		{now.Add(-time.Minute).Format(http.TimeFormat), 0},
	}
	for _, tt := range tests {
		if got := parseRetryAfter(tt.in, now); got != tt.want {
			t.Errorf("parseRetryAfter(%q) = %s, want %s", tt.in, got, tt.want)
		}
	}
}

// TestSendBodyTimeout covers responses whose headers arrive but whose body
// stalls past the client timeout.
func TestSendBodyTimeout(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		retryAfter string
		wantCalls  int64
		wantErr    string
	}{
		{"permanent 4xx is not retried", 400, "", 1, ""},
		{"permanent 501 is not retried", 501, "", 1, ""},
		{"Retry-After beyond max wait still stops retrying", 503, "120", 1, "exceeds max wait"},
		{"retryable status is retried", 503, "0", 2, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int64
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				if tt.retryAfter != "" {
					w.Header().Set("Retry-After", tt.retryAfter)
				}
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(`{"error":`))
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
				case <-time.After(5 * time.Second):
				}
			}))
			defer srv.Close()
			ep := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Retries: 1, MaxWait: 20 * time.Millisecond, HTTP: &http.Client{Timeout: 100 * time.Millisecond}}
			got, err := ep.Send(context.Background(), Request{User: "u"})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want one containing %q", err, tt.wantErr)
			}
			if calls.Load() != tt.wantCalls || got.HTTPStatus != tt.status {
				t.Errorf("requests = %d, status = %d, want %d and %d", calls.Load(), got.HTTPStatus, tt.wantCalls, tt.status)
			}
		})
	}
}
