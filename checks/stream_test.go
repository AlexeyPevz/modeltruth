package checks

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// anthropicEvents are the events of a well-formed Anthropic stream for the
// tiny prompt, in order; input is billed as 14 tokens.
var anthropicEvents = []string{
	"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01MzmDFUhapVJeHMhAUX2ZwC\",\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":14,\"output_tokens\":1}}}\n\n",
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
	"event: ping\ndata: {\"type\":\"ping\"}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"o\"}}\n\n",
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"k\"}}\n\n",
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
}

// openAIEvents are the chunks of a well-formed OpenAI stream with
// stream_options.include_usage; input is billed as 14 tokens.
var openAIEvents = []string{
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-6-sol\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}],\"usage\":null}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-6-sol\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-6-sol\",\"choices\":[],\"usage\":{\"prompt_tokens\":14,\"completion_tokens\":1}}\n\n",
	"data: [DONE]\n\n",
}

// fixture builds a stream body from events, applying edit to a copy.
func fixture(events []string, edit func([]string) []string) string {
	ev := append([]string(nil), events...)
	if edit != nil {
		ev = edit(ev)
	}
	return strings.Join(ev, "")
}

func without(i int) func([]string) []string {
	return func(ev []string) []string { return append(ev[:i], ev[i+1:]...) }
}

func swap(i, j int) func([]string) []string {
	return func(ev []string) []string { ev[i], ev[j] = ev[j], ev[i]; return ev }
}

func replace(old, new string) func([]string) []string {
	return func(ev []string) []string {
		for i := range ev {
			ev[i] = strings.ReplaceAll(ev[i], old, new)
		}
		return ev
	}
}

// streamSamples sends the tiny prompt once per body through probe.Endpoint
// to a server that answers with that body, and returns the stream samples
// next to two non-streamed tiny samples billed as 14 input tokens.
func streamSamples(t *testing.T, protocol probe.Protocol, bodies ...string) []Sample {
	t.Helper()
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		body := bodies[calls.Add(1)-1]
		switch body {
		case "503":
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		case "stall":
			w.Header().Set("Content-Type", "text/event-stream")
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()
	ep := probe.Endpoint{BaseURL: srv.URL, APIKey: "test-key", Model: "m", Protocol: protocol, HTTP: &http.Client{Timeout: 200 * time.Millisecond}}
	samples := []Sample{tiny(14), tiny(14)}
	for range bodies {
		resp, err := ep.Send(context.Background(), probe.Request{User: tinyPrompt("ok"), Stream: true})
		samples = append(samples, Sample{Probe: ProbeStream, PromptChars: len(tinyPrompt("ok")), Response: resp, Err: err})
	}
	return samples
}

func TestStreamShape(t *testing.T) {
	okA := fixture(anthropicEvents, nil)
	okO := fixture(openAIEvents, nil)
	tests := []struct {
		name     string
		protocol probe.Protocol
		bodies   []string
		want     Status
	}{
		{"anthropic: well-formed streams pass", probe.ProtocolAnthropic, []string{okA, okA}, StatusPass},
		{"anthropic: ping and unknown events anywhere pass", probe.ProtocolAnthropic, []string{
			fixture(anthropicEvents, func(ev []string) []string {
				odd := "event: \x1b[2Jvendor_note\ndata: {\"type\":\"\x1b[2Jvendor_note\"}\n\n"
				return append(append([]string{odd, ev[2]}, ev...), ev[2], odd)
			}), okA,
		}, StatusPass},
		{"anthropic: missing message_stop once warns", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, without(7)), okA}, StatusWarn},
		{"anthropic: missing message_stop twice fails", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, without(7)), fixture(anthropicEvents, without(7))}, StatusFail},
		{"anthropic: missing message_start twice fails", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, without(0)), fixture(anthropicEvents, without(0))}, StatusFail},
		{"anthropic: missing content_block_start twice fails", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, without(1)), fixture(anthropicEvents, without(1))}, StatusFail},
		{"anthropic: message_delta before content_block_stop twice fails", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, swap(5, 6)), fixture(anthropicEvents, swap(5, 6))}, StatusFail},
		{"anthropic: reordered once warns", probe.ProtocolAnthropic, []string{okA, fixture(anthropicEvents, swap(5, 6))}, StatusWarn},
		{"anthropic: stream without usage warns", probe.ProtocolAnthropic, []string{
			fixture(anthropicEvents, replace(`,"usage":{"input_tokens":14,"output_tokens":1}`, "")),
			fixture(anthropicEvents, replace(`,"usage":{"input_tokens":14,"output_tokens":1}`, "")),
		}, StatusWarn},
		{"anthropic: streams billed unlike non-streamed requests fail", probe.ProtocolAnthropic, []string{
			fixture(anthropicEvents, replace(`"input_tokens":14`, `"input_tokens":4400`)),
			fixture(anthropicEvents, replace(`"input_tokens":14`, `"input_tokens":4400`)),
		}, StatusFail},
		{"anthropic: one stream billed unlike non-streamed requests warns", probe.ProtocolAnthropic, []string{
			okA, fixture(anthropicEvents, replace(`"input_tokens":14`, `"input_tokens":4400`)),
		}, StatusWarn},
		{"anthropic: small usage jitter passes", probe.ProtocolAnthropic, []string{
			fixture(anthropicEvents, replace(`"input_tokens":14`, `"input_tokens":30`)), okA,
		}, StatusPass},
		{"openai: well-formed streams pass", probe.ProtocolOpenAI, []string{okO, okO}, StatusPass},
		{"openai: usage on the last content chunk passes", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, func(ev []string) []string {
				ev[1] = strings.Replace(ev[1], `"usage":null`, `"usage":{"prompt_tokens":14,"completion_tokens":1}`, 1)
				return append(ev[:2], ev[3])
			}), okO,
		}, StatusPass},
		{"openai: missing [DONE] once warns", probe.ProtocolOpenAI, []string{fixture(openAIEvents, without(3)), okO}, StatusWarn},
		{"openai: missing [DONE] twice fails", probe.ProtocolOpenAI, []string{fixture(openAIEvents, without(3)), fixture(openAIEvents, without(3))}, StatusFail},
		{"openai: usage before content twice fails", probe.ProtocolOpenAI, []string{fixture(openAIEvents, swap(1, 2)), fixture(openAIEvents, swap(1, 2))}, StatusFail},
		{"openai: chunks after [DONE] twice fails", probe.ProtocolOpenAI, []string{fixture(openAIEvents, swap(2, 3)), fixture(openAIEvents, swap(2, 3))}, StatusFail},
		{"openai: frames without choices are not content", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, func(ev []string) []string {
				return []string{"data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n", ev[2], ev[3]}
			}),
			fixture(openAIEvents, func(ev []string) []string {
				return []string{"data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n", ev[2], ev[3]}
			}),
		}, StatusFail},
		{"openai: usage before content and a trailing empty frame twice fails", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, func(ev []string) []string {
				return []string{ev[0], ev[2], ev[1], "data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n", ev[3]}
			}),
			fixture(openAIEvents, func(ev []string) []string {
				return []string{ev[0], ev[2], ev[1], "data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n", ev[3]}
			}),
		}, StatusFail},
		{"openai: empty frame after usage passes", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, func(ev []string) []string {
				return []string{ev[0], ev[1], ev[2], "data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n", ev[3]}
			}), okO,
		}, StatusPass},
		{"openai: empty frames after [DONE] twice fail", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, func(ev []string) []string { return append(ev, "data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n") }),
			fixture(openAIEvents, func(ev []string) []string { return append(ev, "data: {\"id\":\"chatcmpl-1\",\"choices\":[]}\n\n") }),
		}, StatusFail},
		{"openai: stream without usage warns", probe.ProtocolOpenAI, []string{fixture(openAIEvents, without(2)), fixture(openAIEvents, without(2))}, StatusWarn},
		{"openai: streams billed unlike non-streamed requests fail", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, replace(`"prompt_tokens":14`, `"prompt_tokens":420`)),
			fixture(openAIEvents, replace(`"prompt_tokens":14`, `"prompt_tokens":420`)),
		}, StatusFail},
		{"anthropic: final message_stop with no line ending", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, func(ev []string) []string { ev[7] = strings.TrimSuffix(ev[7], "\n\n") + ""; return ev }), fixture(anthropicEvents, func(ev []string) []string { ev[7] = strings.TrimSuffix(ev[7], "\n\n") + ""; return ev })}, StatusFail},
		{"anthropic: final message_stop with one line ending", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, func(ev []string) []string { ev[7] = strings.TrimSuffix(ev[7], "\n\n") + "\n"; return ev }), fixture(anthropicEvents, func(ev []string) []string { ev[7] = strings.TrimSuffix(ev[7], "\n\n") + "\n"; return ev })}, StatusFail},
		{"anthropic: final message_stop with a blank line", probe.ProtocolAnthropic, []string{fixture(anthropicEvents, func(ev []string) []string { ev[7] = strings.TrimSuffix(ev[7], "\n\n") + "\n\n"; return ev }), fixture(anthropicEvents, func(ev []string) []string { ev[7] = strings.TrimSuffix(ev[7], "\n\n") + "\n\n"; return ev })}, StatusPass},
		{"openai: final [DONE] with no line ending", probe.ProtocolOpenAI, []string{fixture(openAIEvents, func(ev []string) []string { ev[3] = strings.TrimSuffix(ev[3], "\n\n") + ""; return ev }), fixture(openAIEvents, func(ev []string) []string { ev[3] = strings.TrimSuffix(ev[3], "\n\n") + ""; return ev })}, StatusFail},
		{"openai: final [DONE] with one line ending", probe.ProtocolOpenAI, []string{fixture(openAIEvents, func(ev []string) []string { ev[3] = strings.TrimSuffix(ev[3], "\n\n") + "\n"; return ev }), fixture(openAIEvents, func(ev []string) []string { ev[3] = strings.TrimSuffix(ev[3], "\n\n") + "\n"; return ev })}, StatusFail},
		{"openai: final [DONE] with a blank line", probe.ProtocolOpenAI, []string{fixture(openAIEvents, func(ev []string) []string { ev[3] = strings.TrimSuffix(ev[3], "\n\n") + "\n\n"; return ev }), fixture(openAIEvents, func(ev []string) []string { ev[3] = strings.TrimSuffix(ev[3], "\n\n") + "\n\n"; return ev })}, StatusPass},
		{"anthropic: streams cut off mid-JSON are transport errors", probe.ProtocolAnthropic, []string{
			fixture(anthropicEvents, func(ev []string) []string { ev[7] = ev[7][:len(ev[7])-6]; return ev }),
			fixture(anthropicEvents, func(ev []string) []string { ev[7] = ev[7][:len(ev[7])-6]; return ev }),
		}, StatusSkip},
		{"openai: streams cut off mid-JSON are transport errors", probe.ProtocolOpenAI, []string{
			fixture(openAIEvents, func(ev []string) []string { ev[2] = ev[2][:len(ev[2])-6]; return ev[:3] }),
			fixture(openAIEvents, func(ev []string) []string { ev[2] = ev[2][:len(ev[2])-6]; return ev[:3] }),
		}, StatusSkip},
		{"anthropic: transport errors skip", probe.ProtocolAnthropic, []string{"503", "stall"}, StatusSkip},
		{"openai: a transport error does not turn a broken stream into a fail", probe.ProtocolOpenAI, []string{"503", fixture(openAIEvents, without(3))}, StatusWarn},
		{"anthropic: upstream error event is a transport error", probe.ProtocolAnthropic, []string{
			fixture(anthropicEvents, func(ev []string) []string {
				return []string{ev[0], "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"}
			}), okA,
		}, StatusPass},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StreamShape(streamSamples(t, tt.protocol, tt.bodies...))
			if got.Status != tt.want {
				t.Errorf("status = %q, want %q (%s; evidence %v)", got.Status, tt.want, got.Summary, got.Evidence)
			}
			if got.Check != "stream_shape" {
				t.Errorf("check = %q, want stream_shape", got.Check)
			}
		})
	}
}

func TestStreamShapeJSONAnswer(t *testing.T) {
	json := probe.Response{HTTPStatus: 200, Usage: probe.Usage{InputTokens: 14, OutputTokens: 1}, Stream: &probe.Stream{Protocol: probe.ProtocolOpenAI, JSON: true, Usage: true}}
	samples := []Sample{tiny(14), {Probe: ProbeStream, Response: json}, {Probe: ProbeStream, Response: json}}
	if got := StreamShape(samples); got.Status != StatusFail || !strings.Contains(got.Summary, "JSON body") {
		t.Errorf("status = %q (%s), want fail for JSON answers to stream requests", got.Status, got.Summary)
	}
}

func TestCompactEvents(t *testing.T) {
	got := compactEvents([]string{"message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_delta", "content_block_stop"})
	if want := "message_start, content_block_start, content_block_delta×3, content_block_stop"; got != want {
		t.Errorf("compactEvents = %q, want %q", got, want)
	}
}

func TestChannelConsistencyLeavesStreamsToStreamShape(t *testing.T) {
	samples := streamSamples(t, probe.ProtocolAnthropic,
		fixture(anthropicEvents, nil), fixture(anthropicEvents, replace(`"input_tokens":14`, `"input_tokens":4400`)))
	if got := ChannelConsistency(samples); got.Status != StatusPass {
		t.Errorf("channel_consistency = %q, want pass: one divergent stream only warns in stream_shape (%s)", got.Status, got.Summary)
	}
}
