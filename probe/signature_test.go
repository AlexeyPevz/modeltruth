package probe

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSignedRequestRoundTrip(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString([]byte("synthetic-signature-0123456789"))
	tests := []struct {
		protocol Protocol
		answer   string
		// check inspects the second request body.
		check func(t *testing.T, body map[string]any)
	}{
		{ProtocolAnthropic,
			`{"id":"msg_1","model":"claude-sonnet-5","content":[{"type":"thinking","thinking":"t","signature":"` + sig + `"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Oslo"}}],"usage":{"input_tokens":600,"output_tokens":60}}`,
			func(t *testing.T, body map[string]any) {
				if th := body["thinking"].(map[string]any); th["type"] != "enabled" || th["budget_tokens"] != float64(ThinkingBudget) {
					t.Errorf("thinking = %v, want the manual thinking of the turn", th)
				}
				if body["max_tokens"] != float64(2*ThinkingBudget) {
					t.Errorf("max_tokens = %v, want %d", body["max_tokens"], 2*ThinkingBudget)
				}
				msgs := body["messages"].([]any)
				blocks := msgs[1].(map[string]any)["content"].([]any)
				if b := blocks[0].(map[string]any); b["type"] != "thinking" || b["thinking"] != "t" || b["signature"] != sig {
					t.Errorf("thinking block = %v, want it unchanged", b)
				}
				if r := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any); r["tool_use_id"] != "toolu_1" {
					t.Errorf("tool result = %v", r)
				}
			}},
		{ProtocolOpenAI,
			`{"id":"c1","model":"gemini-3-pro","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"},"extra_content":{"google":{"thought_signature":"` + sig + `"}}}]}}],"usage":{"prompt_tokens":100,"completion_tokens":20}}`,
			func(t *testing.T, body map[string]any) {
				if body["reasoning_effort"] != "low" {
					t.Errorf("reasoning_effort = %v", body["reasoning_effort"])
				}
				msgs := body["messages"].([]any)
				call := msgs[1].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
				if call["extra_content"].(map[string]any)["google"].(map[string]any)["thought_signature"] != sig {
					t.Errorf("tool call = %v, want the signature sent back", call)
				}
				if m := msgs[2].(map[string]any); m["role"] != "tool" || m["tool_call_id"] != "call_1" {
					t.Errorf("tool message = %v", m)
				}
			}},
	}
	for _, tt := range tests {
		t.Run(string(tt.protocol), func(t *testing.T) {
			var bodies []map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				var b map[string]any
				_ = json.Unmarshal(raw, &b)
				bodies = append(bodies, b)
				if len(bodies) == 1 {
					io.WriteString(w, tt.answer)
					return
				}
				http.Error(w, `{"error":{"message":"Invalid signature"}}`, http.StatusBadRequest)
			}))
			defer srv.Close()
			ep := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m", Protocol: tt.protocol}
			got, err := ep.Send(context.Background(), Request{User: "weather?", MaxTokens: 16, Signed: true, ManualThinking: true})
			if err != nil {
				t.Fatal(err)
			}
			turn := got.Turn
			if turn == nil || turn.Signature != sig || len(turn.ToolCalls) != 1 || !turn.ManualThinking {
				t.Fatalf("turn = %+v", turn)
			}
			if _, ok := bodies[0]["tools"]; !ok {
				t.Errorf("signed request has no tools: %v", bodies[0])
			}
			got, err = ep.Send(context.Background(), Request{User: "weather?", MaxTokens: 16, Reply: turn})
			if err == nil || got.HTTPStatus != http.StatusBadRequest || got.Turn != nil {
				t.Errorf("reply = %+v, %v; want HTTP 400 and no turn", got, err)
			}
			if got := bodies[1]["messages"].([]any); len(got) != 3 {
				t.Fatalf("reply messages = %d, want 3", len(got))
			}
			tt.check(t, bodies[1])
		})
	}
}

func TestSendWithoutSignedHasNoTurn(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(string(raw), "tools") || strings.Contains(string(raw), "reasoning_effort") {
			t.Errorf("plain request carries signed fields: %s", raw)
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()
	got, err := Endpoint{BaseURL: srv.URL, APIKey: "k", Model: "m"}.Send(context.Background(), Request{User: "u"})
	if err != nil || got.Turn != nil {
		t.Errorf("got %+v, %v; want no turn", got, err)
	}
}

func TestParseTurn(t *testing.T) {
	tests := []struct {
		name      string
		protocol  Protocol
		raw       string
		signature string
		calls     int
	}{
		{"anthropic thinking", ProtocolAnthropic, `{"content":[{"type":"thinking","thinking":"a","signature":"S1"},{"type":"thinking","thinking":"b","signature":"S2"},{"type":"tool_use","id":"t1"}]}`, "S1", 1},
		{"anthropic redacted only", ProtocolAnthropic, `{"content":[{"type":"redacted_thinking","data":"R1"},{"type":"tool_use","id":"t1"},{"type":"tool_use","id":"t2"}]}`, "R1", 2},
		{"anthropic text only", ProtocolAnthropic, `{"content":[{"type":"text","text":"hi"}]}`, "", 0},
		{"gemini signature on first call", ProtocolOpenAI, `{"choices":[{"message":{"tool_calls":[{"id":"c1","extra_content":{"google":{"thought_signature":"G1"}}},{"id":"c2"}]}}]}`, "G1", 2},
		{"openai without signature", ProtocolOpenAI, `{"choices":[{"message":{"tool_calls":[{"id":"c1"}]}}]}`, "", 1},
		{"openai no choices", ProtocolOpenAI, `{"choices":[]}`, "", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			turn, err := parseTurn(tt.protocol, []byte(tt.raw))
			if err != nil {
				t.Fatal(err)
			}
			if turn.Signature != tt.signature || len(turn.ToolCalls) != tt.calls {
				t.Errorf("turn = %+v, want signature %q and %d calls", turn, tt.signature, tt.calls)
			}
		})
	}
}

func TestTampered(t *testing.T) {
	b64 := base64.StdEncoding.EncodeToString([]byte("synthetic-signature-bytes"))
	raw := base64.RawURLEncoding.EncodeToString([]byte("synthetic-signature-bytes"))
	for _, sig := range []string{b64, raw, "not*base64*signature"} {
		msg, _ := json.Marshal([]any{
			map[string]any{"type": "thinking", "thinking": "keep", "signature": sig},
			map[string]any{"type": "tool_use", "id": "t1", "input": map[string]any{"n": json.Number("12345678901234567890")}},
		})
		turn := Turn{Protocol: ProtocolAnthropic, Message: msg, Signature: sig, ToolCalls: []string{"t1"}}
		forged, err := turn.Tampered()
		if err != nil {
			t.Fatal(err)
		}
		if forged.Signature == sig || len(forged.Signature) != len(sig) {
			t.Errorf("forged %q from %q", forged.Signature, sig)
		}
		if want := strings.Replace(string(msg), sig, forged.Signature, 1); string(forged.Message) != want {
			t.Errorf("message = %s, want %s", forged.Message, want)
		}
		if string(turn.Message) != string(msg) || turn.Signature != sig {
			t.Error("Tampered changed the original turn")
		}
	}
	if _, err := (Turn{Message: []byte(`[]`)}).Tampered(); err == nil {
		t.Error("tampered a turn without a signature")
	}
	if _, err := (Turn{Message: []byte(`[{"signature":"other"}]`), Signature: "S"}).Tampered(); err == nil {
		t.Error("tampered a signature missing from the message")
	}
}

func TestAnthropicThinkingModes(t *testing.T) {
	for _, tt := range []struct {
		req  Request
		want string
	}{
		{Request{User: "u", Signed: true}, `{"type":"adaptive"}`},
		{Request{User: "u", Signed: true, ManualThinking: true}, `{"budget_tokens":1024,"type":"enabled"}`},
		{Request{User: "u", Reply: &Turn{Message: []byte(`[]`)}}, `{"type":"adaptive"}`},
		{Request{User: "u", Reply: &Turn{Message: []byte(`[]`), ManualThinking: true}}, `{"budget_tokens":1024,"type":"enabled"}`},
	} {
		_, body, _, err := Endpoint{BaseURL: "http://x", Model: "m", Protocol: ProtocolAnthropic}.build(tt.req)
		if err != nil {
			t.Fatal(err)
		}
		var b struct {
			Thinking  json.RawMessage `json:"thinking"`
			MaxTokens int             `json:"max_tokens"`
		}
		_ = json.Unmarshal(body, &b)
		if string(b.Thinking) != tt.want || b.MaxTokens <= ThinkingBudget {
			t.Errorf("%+v: thinking %s, max_tokens %d; want %s above the budget", tt.req, b.Thinking, b.MaxTokens, tt.want)
		}
	}
}
