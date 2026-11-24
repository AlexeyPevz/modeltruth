package probe

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
)

// ThinkingBudget is the budget_tokens of an Anthropic request with
// ManualThinking, the smallest budget Anthropic accepts. max_tokens must
// exceed it, so a signed Anthropic request with a smaller MaxTokens asks for
// twice the budget.
const ThinkingBudget = 1024

// SignatureTool is the tool a signed request offers. The signature is
// only verified when it is sent back with a tool call: Anthropic requires
// the thinking block of the last assistant turn to come back unmodified
// during tool use, and Gemini validates thought signatures on function
// calls only.
const SignatureTool = "get_weather"

// signatureToolResult is the tool result every call in a sent-back turn gets.
const signatureToolResult = `{"forecast":"sunny","temperature_c":18}`

var signatureToolSchema = map[string]any{
	"type":       "object",
	"properties": map[string]any{"city": map[string]any{"type": "string"}},
	"required":   []string{"city"},
}

const signatureToolDescription = "Get the current weather for a city."

// Turn is the assistant turn that answered a signed request, kept so that it
// can be sent back.
type Turn struct {
	Protocol Protocol
	// Message is the turn as the endpoint returned it: the Anthropic content
	// array, or the OpenAI choices[0].message object.
	Message json.RawMessage
	// Signature is the first reasoning signature in the turn: the signature
	// of an Anthropic thinking block (the data of a redacted_thinking block
	// when there is none), or the Gemini thought_signature of a tool call
	// (tool_calls[].extra_content.google.thought_signature). Empty when the
	// turn carries none.
	Signature string
	// ToolCalls are the ids of the tool calls in the turn, in order.
	ToolCalls []string
	// ManualThinking is that of the request the turn answered; sending the
	// turn back repeats it.
	ManualThinking bool
}

// Tampered returns a copy of t whose signature has one byte changed: one
// bit of the middle byte of the decoded value when the signature is base64,
// otherwise its middle character. Nothing else in the message changes.
func (t Turn) Tampered() (Turn, error) {
	if t.Signature == "" {
		return Turn{}, errors.New("probe: turn has no signature")
	}
	forged := tamper(t.Signature)
	dec := json.NewDecoder(bytes.NewReader(t.Message))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return Turn{}, fmt.Errorf("probe: decode turn: %w", err)
	}
	v, n := replaceString(v, t.Signature, forged)
	if n == 0 {
		return Turn{}, errors.New("probe: signature not found in turn")
	}
	msg, err := json.Marshal(v)
	if err != nil {
		return Turn{}, err
	}
	out := t
	out.Message, out.Signature = msg, forged
	out.ToolCalls = append([]string(nil), t.ToolCalls...)
	return out, nil
}

func tamper(sig string) string {
	for _, enc := range []*base64.Encoding{base64.StdEncoding, base64.URLEncoding, base64.RawStdEncoding, base64.RawURLEncoding} {
		if b, err := enc.Strict().DecodeString(sig); err == nil && len(b) > 0 {
			b[len(b)/2] ^= 0x01
			return enc.EncodeToString(b)
		}
	}
	b := []byte(sig)
	i := len(b) / 2
	if b[i] == 'A' {
		b[i] = 'B'
	} else {
		b[i] = 'A'
	}
	return string(b)
}

// replaceString replaces every string value equal to old anywhere in v.
func replaceString(v any, old, new string) (any, int) {
	n := 0
	switch x := v.(type) {
	case string:
		if x == old {
			return new, 1
		}
	case []any:
		for i := range x {
			var c int
			x[i], c = replaceString(x[i], old, new)
			n += c
		}
	case map[string]any:
		for k := range x {
			var c int
			x[k], c = replaceString(x[k], old, new)
			n += c
		}
	}
	return v, n
}

func signed(req Request) bool {
	return req.Signed || req.Reply != nil
}

func manualThinking(req Request) bool {
	if req.Reply != nil {
		return req.Reply.ManualThinking
	}
	return req.ManualThinking
}

// anthropicMessages is the user prompt, followed for a reply by the turn
// sent back and a tool result for each of its tool calls.
func anthropicMessages(req Request) []any {
	messages := []any{map[string]any{"role": "user", "content": req.User}}
	if req.Reply == nil {
		return messages
	}
	results := []any{}
	for _, id := range req.Reply.ToolCalls {
		results = append(results, map[string]any{"type": "tool_result", "tool_use_id": id, "content": signatureToolResult})
	}
	return append(messages,
		map[string]any{"role": "assistant", "content": req.Reply.Message},
		map[string]any{"role": "user", "content": results})
}

// anthropicSigned enables adaptive thinking, which Claude 4.7 and later
// require, or with manual extended thinking, which Claude 4.5 and earlier
// require.
func anthropicSigned(payload map[string]any, maxTokens int, manual bool) {
	if maxTokens <= ThinkingBudget {
		payload["max_tokens"] = 2 * ThinkingBudget
	}
	payload["thinking"] = map[string]any{"type": "adaptive"}
	if manual {
		payload["thinking"] = map[string]any{"type": "enabled", "budget_tokens": ThinkingBudget}
	}
	payload["tools"] = []any{map[string]any{
		"name":         SignatureTool,
		"description":  signatureToolDescription,
		"input_schema": signatureToolSchema,
	}}
}

// openAIReply appends the turn sent back and a tool message for each of its
// tool calls.
func openAIReply(messages []any, reply *Turn) []any {
	messages = append(messages, reply.Message)
	for _, id := range reply.ToolCalls {
		messages = append(messages, map[string]any{"role": "tool", "tool_call_id": id, "content": signatureToolResult})
	}
	return messages
}

// openAISigned offers the tool and asks for reasoning_effort "low", which
// Gemini's OpenAI compatibility maps to a thinking budget.
func openAISigned(payload map[string]any) {
	payload["reasoning_effort"] = "low"
	payload["tools"] = []any{map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        SignatureTool,
			"description": signatureToolDescription,
			"parameters":  signatureToolSchema,
		},
	}}
}

func parseTurn(protocol Protocol, raw []byte) (*Turn, error) {
	if protocol == ProtocolAnthropic {
		return anthropicTurn(raw)
	}
	return openAITurn(raw)
}

func anthropicTurn(raw []byte) (*Turn, error) {
	var d struct {
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, fmt.Errorf("probe: decode Anthropic turn: %w", err)
	}
	t := &Turn{Protocol: ProtocolAnthropic, Message: d.Content}
	var blocks []struct {
		Type      string `json:"type"`
		Signature string `json:"signature"`
		Data      string `json:"data"`
		ID        string `json:"id"`
	}
	if len(d.Content) > 0 {
		if err := json.Unmarshal(d.Content, &blocks); err != nil {
			return nil, fmt.Errorf("probe: decode Anthropic content: %w", err)
		}
	}
	var redacted string
	for _, b := range blocks {
		switch b.Type {
		case "thinking":
			if t.Signature == "" {
				t.Signature = b.Signature
			}
		case "redacted_thinking":
			if redacted == "" {
				redacted = b.Data
			}
		case "tool_use":
			t.ToolCalls = append(t.ToolCalls, b.ID)
		}
	}
	if t.Signature == "" {
		t.Signature = redacted
	}
	return t, nil
}

func openAITurn(raw []byte) (*Turn, error) {
	var d struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, fmt.Errorf("probe: decode OpenAI turn: %w", err)
	}
	t := &Turn{Protocol: ProtocolOpenAI}
	if len(d.Choices) == 0 {
		return t, nil
	}
	t.Message = d.Choices[0].Message
	var m struct {
		ToolCalls []struct {
			ID           string `json:"id"`
			ExtraContent struct {
				Google struct {
					ThoughtSignature string `json:"thought_signature"`
				} `json:"google"`
			} `json:"extra_content"`
		} `json:"tool_calls"`
	}
	if len(t.Message) > 0 {
		if err := json.Unmarshal(t.Message, &m); err != nil {
			return nil, fmt.Errorf("probe: decode OpenAI message: %w", err)
		}
	}
	for _, c := range m.ToolCalls {
		t.ToolCalls = append(t.ToolCalls, c.ID)
		if t.Signature == "" {
			t.Signature = c.ExtraContent.Google.ThoughtSignature
		}
	}
	return t, nil
}
