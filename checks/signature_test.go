package checks

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// vendor modes of signatureServer.
const (
	honest         = "honest"          // accepts the signatures it issued, rejects the rest
	acceptsAny     = "accepts_any"     // accepts every sent-back turn
	rejectsAll     = "rejects_all"     // rejects every sent-back turn
	unsigned       = "unsigned"        // answers with a tool call but no signature
	noThinking     = "no_thinking"     // refuses thinking and tools with HTTP 400
	rateLimited    = "rate_limited"    // answers sent-back turns with HTTP 429
	brokenFirst    = "broken_first"    // answers signed requests with HTTP 503
	honestOnce     = "honest_once"     // accepts the forgery in the second round only
	noToolCall     = "no_tool_call"    // signs but answers in text without a tool call
	acceptsAnyOnce = "accepts_any_1x"  // accepts the forgery in the first round only
	errorBody      = "error_body"      // answers sent-back turns with HTTP 200 and an error body
	manualOnly     = "manual_only"     // refuses adaptive thinking like Claude 4.5, otherwise honest
	manualUnsigned = "manual_unsigned" // refuses adaptive thinking and answers without a signature
)

// signatureServer simulates the signing side of an Anthropic or a Gemini
// OpenAI-compatible endpoint. Signatures are synthetic base64 values; the
// server records which ones it issued and which came back.
type signatureServer struct {
	t        *testing.T
	protocol probe.Protocol
	mode     string

	mu       sync.Mutex
	issued   map[string]bool
	returned []string
	bodies   []map[string]any
	replies  int
}

func newSignatureServer(t *testing.T, protocol probe.Protocol, mode string) (*signatureServer, probe.Endpoint) {
	s := &signatureServer{t: t, protocol: protocol, mode: mode, issued: map[string]bool{}}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	return s, probe.Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Protocol: protocol}
}

func (s *signatureServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		s.t.Errorf("request body: %v", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bodies = append(s.bodies, body)
	messages, _ := body["messages"].([]any)
	thinking, _ := body["thinking"].(map[string]any)
	if s.mode == noThinking || (s.mode == manualOnly || s.mode == manualUnsigned) && thinking["type"] == "adaptive" {
		http.Error(w, `{"error":{"message":"thinking type is not supported for this model"}}`, http.StatusBadRequest)
		return
	}
	if len(messages) == 1 {
		if s.mode == brokenFirst {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		sig := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("synthetic-signature-%02d-0123456789abcdef", len(s.issued))))
		s.issued[sig] = true
		w.Write(s.signedAnswer(sig, len(s.issued)))
		return
	}
	sig := s.sentBack(messages)
	s.returned = append(s.returned, sig)
	s.replies++
	round := (s.replies + 1) / 2
	accept := s.issued[sig]
	switch {
	case s.mode == acceptsAny, s.mode == honestOnce && round == 2, s.mode == acceptsAnyOnce && round == 1:
		accept = true
	case s.mode == rejectsAll:
		accept = false
	case s.mode == errorBody:
		w.Write([]byte(`{"error":{"message":"upstream failed"}}`))
		return
	case s.mode == rateLimited:
		http.Error(w, "slow down", http.StatusTooManyRequests)
		return
	}
	if !accept {
		http.Error(w, `{"error":{"message":"Invalid signature"}}`, http.StatusBadRequest)
		return
	}
	if s.protocol == probe.ProtocolAnthropic {
		w.Write([]byte(`{"id":"msg_01MzmDFUhapVJeHMhAUX2ZwC","model":"claude-sonnet-5","content":[{"type":"text","text":"Sunny, 18 C."}],"usage":{"input_tokens":700,"output_tokens":8}}`))
	} else {
		w.Write([]byte(`{"id":"chatcmpl-synth","model":"gemini-3-pro","choices":[{"message":{"role":"assistant","content":"Sunny, 18 C."}}],"usage":{"prompt_tokens":120,"completion_tokens":8}}`))
	}
}

func (s *signatureServer) signedAnswer(sig string, n int) []byte {
	id := fmt.Sprintf("call_%02d", n)
	if s.protocol == probe.ProtocolAnthropic {
		thinking := fmt.Sprintf(`{"type":"thinking","thinking":"The user wants the weather; call the tool.","signature":%q},`, sig)
		if s.mode == unsigned || s.mode == manualUnsigned {
			thinking = ""
		}
		tool := fmt.Sprintf(`{"type":"tool_use","id":%q,"name":"get_weather","input":{"city":"Oslo"}}`, id)
		if s.mode == noToolCall {
			tool = `{"type":"text","text":"I cannot check the weather."}`
		}
		return []byte(`{"id":"msg_01MzmDFUhapVJeHMhAUX2ZwC","model":"claude-sonnet-5","content":[` + thinking + tool + `],"usage":{"input_tokens":600,"output_tokens":60}}`)
	}
	extra := fmt.Sprintf(`,"extra_content":{"google":{"thought_signature":%q}}`, sig)
	if s.mode == unsigned || s.mode == manualUnsigned {
		extra = ""
	}
	message := fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":%q,"type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}%s}]}`, id, extra)
	if s.mode == noToolCall {
		message = `{"role":"assistant","content":"I cannot check the weather."}`
	}
	return []byte(`{"id":"chatcmpl-synth","model":"gemini-3-pro","choices":[{"message":` + message + `}],"usage":{"prompt_tokens":100,"completion_tokens":20}}`)
}

// sentBack returns the signature in the assistant turn of a sent-back
// conversation and checks the tool results that follow it.
func (s *signatureServer) sentBack(messages []any) string {
	s.t.Helper()
	assistant, _ := messages[1].(map[string]any)
	if assistant["role"] != "assistant" {
		s.t.Errorf("second message role = %v, want assistant", assistant["role"])
	}
	if s.protocol == probe.ProtocolAnthropic {
		if len(messages) != 3 {
			s.t.Fatalf("messages = %d, want user, assistant, tool results", len(messages))
		}
		results, _ := messages[2].(map[string]any)["content"].([]any)
		if len(results) != 1 || results[0].(map[string]any)["type"] != "tool_result" || results[0].(map[string]any)["tool_use_id"] == "" {
			s.t.Errorf("tool results = %v", results)
		}
		blocks, _ := assistant["content"].([]any)
		for _, b := range blocks {
			if b := b.(map[string]any); b["type"] == "thinking" {
				return b["signature"].(string)
			}
		}
		return ""
	}
	if len(messages) != 3 || messages[2].(map[string]any)["role"] != "tool" {
		s.t.Errorf("messages = %v, want user, assistant, tool", messages)
	}
	calls, _ := assistant["tool_calls"].([]any)
	if len(calls) == 0 {
		return ""
	}
	extra, _ := calls[0].(map[string]any)["extra_content"].(map[string]any)
	google, _ := extra["google"].(map[string]any)
	sig, _ := google["thought_signature"].(string)
	return sig
}

func TestSignature(t *testing.T) {
	families := []struct {
		protocol probe.Protocol
		model    string
	}{
		{probe.ProtocolAnthropic, "claude-sonnet-5"},
		{probe.ProtocolOpenAI, "gemini-3-pro-preview"},
	}
	tests := []struct {
		mode       string
		want       Status
		wantClaude Status // when it differs from want
		wantErrors bool
	}{
		{mode: honest, want: StatusPass},
		{mode: acceptsAny, want: StatusFail},
		{mode: rejectsAll, want: StatusFail},
		// Adaptive thinking may skip the thinking block; Gemini 3 always signs.
		{mode: unsigned, want: StatusWarn, wantClaude: StatusSkip},
		{mode: manualOnly, want: StatusPass},
		{mode: manualUnsigned, want: StatusWarn},
		{mode: noThinking, want: StatusSkip},
		{mode: rateLimited, want: StatusSkip, wantErrors: true},
		{mode: brokenFirst, want: StatusSkip, wantErrors: true},
		{mode: honestOnce, want: StatusWarn},
		{mode: acceptsAnyOnce, want: StatusWarn},
		{mode: noToolCall, want: StatusSkip},
		{mode: errorBody, want: StatusSkip},
	}
	for _, fam := range families {
		for _, tt := range tests {
			t.Run(fam.model+"/"+tt.mode, func(t *testing.T) {
				srv, ep := newSignatureServer(t, fam.protocol, tt.mode)
				samples := signatureRun(context.Background(), ep, fam.protocol, fam.model, "Oslo")
				f := Signature(samples, fam.protocol, fam.model)
				want := tt.want
				if fam.protocol == probe.ProtocolAnthropic && tt.wantClaude != "" {
					want = tt.wantClaude
				}
				if f.Check != "channel_signature" || f.Status != want {
					t.Errorf("%s = %s, want %s: %s", f.Check, f.Status, want, f.Summary)
				}
				var transport int
				for _, s := range samples {
					if s.Err != nil && !rejected(s) && !refusal(s) {
						transport++
					}
				}
				if (transport > 0) != tt.wantErrors {
					t.Errorf("transport errors = %d, want any: %v", transport, tt.wantErrors)
				}
				if limit := 3*SignatureProbes + 1; len(samples) > limit {
					t.Errorf("%d requests, want at most %d", len(samples), limit)
				}
				srv.checkRequests(t)
			})
		}
	}
}

// checkRequests verifies the request shapes and that every sent-back pair
// carried the issued signature, then the same signature with one byte changed.
func (s *signatureServer) checkRequests(t *testing.T) {
	t.Helper()
	for _, body := range s.bodies {
		tools, _ := body["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools = %v, want the weather tool", body["tools"])
		}
		if s.protocol == probe.ProtocolAnthropic {
			thinking, _ := body["thinking"].(map[string]any)
			adaptive := thinking["type"] == "adaptive" && len(thinking) == 1
			manual := thinking["type"] == "enabled" && thinking["budget_tokens"] == float64(probe.ThinkingBudget)
			if !adaptive && !manual {
				t.Errorf("thinking = %v", body["thinking"])
			}
			if adaptive && (s.mode == manualOnly || s.mode == manualUnsigned) && len(body["messages"].([]any)) > 1 {
				t.Errorf("sent a turn back with adaptive thinking after it was refused")
			}
			if mt, _ := body["max_tokens"].(float64); mt <= probe.ThinkingBudget {
				t.Errorf("max_tokens = %v, want above the thinking budget", mt)
			}
		} else if body["reasoning_effort"] != "low" {
			t.Errorf("reasoning_effort = %v, want low", body["reasoning_effort"])
		}
	}
	if s.mode == unsigned || s.mode == manualUnsigned || s.mode == noToolCall {
		if len(s.returned) != 0 {
			t.Errorf("sent back %d turns without a signature or tool call", len(s.returned))
		}
		return
	}
	if len(s.returned)%2 != 0 {
		t.Fatalf("sent back %d turns, want pairs", len(s.returned))
	}
	for i := 0; i < len(s.returned); i += 2 {
		original, forged := s.returned[i], s.returned[i+1]
		if !s.issued[original] {
			t.Errorf("original %q was not issued", original)
		}
		a, _ := base64.StdEncoding.DecodeString(original)
		b, err := base64.StdEncoding.DecodeString(forged)
		if err != nil || len(a) != len(b) {
			t.Fatalf("forged %q does not decode to the length of %q", forged, original)
		}
		diff := 0
		for j := range a {
			if a[j] != b[j] {
				diff++
			}
		}
		if diff != 1 || bytes.Equal(a, b) {
			t.Errorf("forged signature differs in %d bytes, want 1", diff)
		}
	}
}

func TestSignatureSkipsOtherModels(t *testing.T) {
	s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) {
		t.Error("sent a request")
		return probe.Response{}, nil
	}}
	for _, tt := range []struct {
		protocol probe.Protocol
		model    string
	}{
		{probe.ProtocolOpenAI, "gpt-6-sol"},
		{probe.ProtocolOpenAI, "claude-sonnet-5"}, // Claude over an OpenAI-compatible relay returns no signatures to verify
		{probe.ProtocolAnthropic, "gemini-3-pro"},
		{probe.ProtocolOpenAI, "gemini-2.5-pro"}, // Gemini 2.5 does not validate thought signatures
		{probe.ProtocolOpenAI, "gemini-pro"},
	} {
		if samples := signatureRun(context.Background(), s, tt.protocol, tt.model, "Oslo"); len(samples) != 0 {
			t.Errorf("%s over %s: %d samples, want none", tt.model, tt.protocol, len(samples))
		}
		if f := Signature(nil, tt.protocol, tt.model); f.Status != StatusSkip {
			t.Errorf("%s over %s: %s, want skip", tt.model, tt.protocol, f.Status)
		}
	}
}

func TestRunSignature(t *testing.T) {
	for _, tt := range []struct {
		mode string
		want Status
	}{{honest, StatusPass}, {acceptsAny, StatusFail}} {
		t.Run(tt.mode, func(t *testing.T) {
			_, ep := newSignatureServer(t, probe.ProtocolAnthropic, tt.mode)
			s := &fakeSender{responses: func(_ int, req probe.Request) (probe.Response, error) {
				if req.Signed || req.Reply != nil {
					return ep.Send(context.Background(), req)
				}
				return probe.Response{HTTPStatus: 200, ID: "msg_01MzmDFUhapVJeHMhAUX2ZwC", Model: "claude-sonnet-5", Text: "ok", Usage: probe.Usage{InputTokens: cleanInput(req), OutputTokens: 1}}, nil
			}}
			rep := Run(context.Background(), s, probe.ProtocolAnthropic, "claude-sonnet-5", 3)
			if want := 3*perRun + fixedRequests + 3*SignatureProbes; len(s.calls) != want || rep.Requests != want {
				t.Errorf("requests = %d (report %d), want %d", len(s.calls), rep.Requests, want)
			}
			sig := rep.Findings[len(rep.Findings)-1]
			if rep.Verdict != tt.want || sig.Check != "channel_signature" || sig.Status != tt.want {
				t.Errorf("verdict = %s, signature = %+v, want %s", rep.Verdict, sig, tt.want)
			}
			if len(rep.Errors) != 0 || rep.Successful != rep.Requests {
				t.Errorf("errors = %+v, successful %d of %d: rejected forgeries are answers, not errors", rep.Errors, rep.Successful, rep.Requests)
			}
		})
	}
}
