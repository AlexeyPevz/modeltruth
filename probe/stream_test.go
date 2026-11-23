package probe

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

const openAIStream = `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-6-sol","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-6-sol","choices":[{"index":0,"delta":{"content":"rea"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-6-sol","choices":[{"index":0,"delta":{"content":"dy"},"finish_reason":"stop"}],"usage":null}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-6-sol","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":0}}}

data: [DONE]

`

const anthropicStream = "event: message_start\r\n" +
	`data: {"type":"message_start","message":{"id":"msg_01MzmDFUhapVJeHMhAUX2ZwC","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"usage":{"input_tokens":10,"cache_read_input_tokens":4,"output_tokens":1}}}` + "\r\n\r\n" +
	"event: content_block_start\r\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\r\n\r\n" +
	"event: ping\r\n" +
	`data: {"type": "ping"}` + "\r\n\r\n" +
	": keep-alive comment\r\n\r\n" +
	"event: content_block_delta\r\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"rea"}}` + "\r\n\r\n" +
	"event: content_block_delta\r\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"dy"}}` + "\r\n\r\n" +
	"event: content_block_stop\r\n" +
	`data: {"type":"content_block_stop","index":0}` + "\r\n\r\n" +
	"event: message_delta\r\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}` + "\r\n\r\n" +
	"event: message_stop\r\n" +
	`data: {"type":"message_stop"}` + "\r\n\r\n"

func TestSendStream(t *testing.T) {
	tests := []struct {
		name        string
		protocol    Protocol
		contentType string
		body        string
		want        Response
		wantEvents  []string
		wantUsage   bool
		wantJSON    bool
		wantErr     string
	}{
		{
			name:       "openai stream with usage in the final chunk",
			protocol:   ProtocolOpenAI,
			body:       openAIStream,
			want:       Response{ID: "chatcmpl-1", Model: "gpt-6-sol", Text: "ready", Usage: Usage{InputTokens: 14, OutputTokens: 2}},
			wantEvents: []string{"chunk", "chunk", "chunk", "usage", "[DONE]"},
			wantUsage:  true,
		},
		{
			name:       "openai stream without usage",
			protocol:   ProtocolOpenAI,
			body:       "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
			want:       Response{ID: "c", Model: "m", Text: "ok"},
			wantEvents: []string{"chunk", "[DONE]"},
		},
		{
			name:       "anthropic stream merges message_start and message_delta usage",
			protocol:   ProtocolAnthropic,
			body:       anthropicStream,
			want:       Response{ID: "msg_01MzmDFUhapVJeHMhAUX2ZwC", Model: "claude-sonnet-5", Text: "ready", Usage: Usage{InputTokens: 14, OutputTokens: 2, CachedTokens: 4}},
			wantEvents: []string{"message_start", "content_block_start", "ping", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"},
			wantUsage:  true,
		},
		{
			name:        "JSON answer to a stream request is marked",
			protocol:    ProtocolOpenAI,
			contentType: "application/json",
			body:        `{"id":"x","model":"m","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":1}}`,
			want:        Response{ID: "x", Model: "m", Text: "ok", Usage: Usage{InputTokens: 9, OutputTokens: 1}},
			wantUsage:   true,
			wantJSON:    true,
		},
		{
			name:     "anthropic error event is an error",
			protocol: ProtocolAnthropic,
			body:     "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\"}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
			wantErr:  "probe: stream error: {\"type\":\"overloaded_error\"",
		},
		{
			name:     "openai error chunk is an error",
			protocol: ProtocolOpenAI,
			body:     "data: {\"error\":{\"message\":\"upstream timeout\"}}\n\n",
			wantErr:  "probe: stream error",
		},
		{
			name:       "openai frames without choices or usage are not chunks",
			protocol:   ProtocolOpenAI,
			body:       "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[]}\n\ndata: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":14}}\n\ndata: [DONE]\n\n",
			want:       Response{ID: "c", Model: "m", Usage: Usage{InputTokens: 14}},
			wantEvents: []string{"usage", "[DONE]"},
			wantUsage:  true,
		},
		{
			name:     "malformed chunk is an error",
			protocol: ProtocolOpenAI,
			body:     "data: {\"id\":\n\n",
			wantErr:  "decode OpenAI stream chunk",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(raw, &gotBody)
				ct := tt.contentType
				if ct == "" {
					ct = "text/event-stream"
				}
				w.Header().Set("Content-Type", ct)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()
			ep := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Protocol: tt.protocol}
			got, err := ep.Send(context.Background(), Request{User: "u", Stream: true})
			if gotBody["stream"] != true {
				t.Errorf("request stream = %v, want true", gotBody["stream"])
			}
			opts, _ := gotBody["stream_options"].(map[string]any)
			if wantOpts := tt.protocol == ProtocolOpenAI; (opts["include_usage"] == true) != wantOpts {
				t.Errorf("stream_options = %v, want include_usage only for OpenAI", gotBody["stream_options"])
			}
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want one containing %q", err, tt.wantErr)
				}
				if got.HTTPStatus != http.StatusOK {
					t.Errorf("HTTPStatus = %d, want 200", got.HTTPStatus)
				}
				return
			}
			if err != nil {
				t.Fatalf("Send: %v", err)
			}
			st := got.Stream
			if st == nil {
				t.Fatal("Stream is nil for a stream request")
			}
			if st.Protocol != tt.protocol || st.Usage != tt.wantUsage || st.JSON != tt.wantJSON || !reflect.DeepEqual(st.Events, tt.wantEvents) {
				t.Errorf("stream = %+v, want protocol %s, events %v, usage %v, json %v", *st, tt.protocol, tt.wantEvents, tt.wantUsage, tt.wantJSON)
			}
			got.Stream, got.Latency, tt.want.HTTPStatus = nil, 0, http.StatusOK
			if got != tt.want {
				t.Errorf("response = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestSendWithoutStreamOmitsStreamFields(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":1}}`))
	}))
	defer srv.Close()
	got, err := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m"}.Send(context.Background(), Request{User: "u"})
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if _, ok := gotBody["stream"]; ok || got.Stream != nil {
		t.Errorf("non-stream request sent stream=%v and got Stream %+v", gotBody["stream"], got.Stream)
	}
}

func TestSendStreamStallIsATimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.SplitAfter(openAIStream, "\n\n")[0]))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()
	ep := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", HTTP: &http.Client{Timeout: 100 * time.Millisecond}}
	got, err := ep.Send(context.Background(), Request{User: "u", Stream: true})
	if err == nil {
		t.Fatal("expected an error for a stalled stream")
	}
	if got.Stream != nil {
		t.Errorf("a stalled stream must not produce a stream trace: %+v", got.Stream)
	}
}
