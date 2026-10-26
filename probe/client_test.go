package probe

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
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
