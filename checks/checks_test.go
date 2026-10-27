package checks

import (
	"errors"
	"testing"

	"github.com/AlexeyPevz/modeltruth/probe"
)

func tiny(input int64) Sample {
	return Sample{Probe: ProbeTiny, PromptChars: len(tinyPrompt("ok")), Response: probe.Response{HTTPStatus: 200, ID: "chatcmpl-1", Model: "m", Text: "ok", Usage: probe.Usage{InputTokens: input, OutputTokens: 1}}}
}

func medium(input, output int64, text string) Sample {
	return Sample{Probe: ProbeMedium, PromptChars: len(mediumPrompt("ok")), Response: probe.Response{HTTPStatus: 200, ID: "chatcmpl-1", Model: "m", Text: text, Usage: probe.Usage{InputTokens: input, OutputTokens: output}}}
}

func withShape(id, model string) Sample {
	s := tiny(12)
	s.Response.ID, s.Response.Model = id, model
	return s
}

func TestHiddenPrompt(t *testing.T) {
	tests := []struct {
		name    string
		samples []Sample
		want    Status
	}{
		{"clean prompt sizes pass", []Sample{tiny(12), tiny(13), tiny(12)}, StatusPass},
		{"moderate overhead warns", []Sample{tiny(150), tiny(150)}, StatusWarn},
		{"hidden system prompt on every request fails", []Sample{tiny(4515), tiny(4514)}, StatusFail},
		{"hidden prompt on some requests still fails", []Sample{tiny(14), tiny(7497), tiny(14)}, StatusFail},
		{"only errors skips", []Sample{{Probe: ProbeTiny, Err: errors.New("timeout")}}, StatusSkip},
		{"no tiny samples skips", []Sample{medium(150, 2, "ok")}, StatusSkip},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := HiddenPrompt(tt.samples)
			if got.Status != tt.want {
				t.Errorf("status = %q, want %q (%s)", got.Status, tt.want, got.Summary)
			}
			if got.Check != "hidden_prompt" {
				t.Errorf("check = %q, want hidden_prompt", got.Check)
			}
		})
	}
}

func TestChannelConsistency(t *testing.T) {
	tests := []struct {
		name    string
		samples []Sample
		want    Status
	}{
		{"identical counts pass", []Sample{tiny(12), tiny(12), medium(160, 2, "ok"), medium(161, 2, "ok")}, StatusPass},
		{"small jitter passes", []Sample{tiny(4511), tiny(4517)}, StatusPass},
		{"two channels behind one model fail", []Sample{tiny(220), tiny(7497), tiny(220)}, StatusFail},
		{"medium probe split fails", []Sample{medium(136, 2, "ok"), medium(10, 2, "ok")}, StatusFail},
		{"single sample per probe skips", []Sample{tiny(12), medium(160, 2, "ok")}, StatusSkip},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ChannelConsistency(tt.samples)
			if got.Status != tt.want {
				t.Errorf("status = %q, want %q (%s)", got.Status, tt.want, got.Summary)
			}
		})
	}
}

func TestUsageSanity(t *testing.T) {
	floor := int64(len(mediumPrompt("ok")) / 6)
	tests := []struct {
		name    string
		samples []Sample
		want    Status
	}{
		{"plausible counts pass", []Sample{medium(floor+40, 2, "ok"), medium(floor+40, 2, "ok")}, StatusPass},
		{"input below what the prompt can tokenize to fails", []Sample{medium(floor+40, 2, "ok"), medium(10, 2, "ok")}, StatusFail},
		{"text without output tokens warns", []Sample{medium(floor+40, 0, "ok")}, StatusWarn},
		{"no medium samples skips", []Sample{tiny(12)}, StatusSkip},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UsageSanity(tt.samples)
			if got.Status != tt.want {
				t.Errorf("status = %q, want %q (%s)", got.Status, tt.want, got.Summary)
			}
		})
	}
}

func TestResponseShape(t *testing.T) {
	tests := []struct {
		name    string
		claimed string
		samples []Sample
		want    Status
	}{
		{"anthropic message id passes", "claude-sonnet-5", []Sample{withShape("msg_01MzmDFUhapVJeHMhAUX2ZwC", "claude-sonnet-5")}, StatusPass},
		{"vertex and bedrock ids pass", "claude-opus-5-5", []Sample{withShape("msg_vrtx_01MzmDFUhapVJeHMhAUX2ZwC", "claude-opus-5-5"), withShape("msg_bdrk_01MzmDFUhapVJeHMhAUX2ZwC", "claude-opus-5-5")}, StatusPass},
		{"gateway rewritten id and prefixed model pass", "claude-opus-5-5", []Sample{withShape("chatcmpl-a73d4121-2303-4e01-a4ac-fc3496f1cad9", "relay:claude-opus-5-5:token")}, StatusPass},
		{"hex message id is not anthropic format", "claude-haiku-4-5", []Sample{withShape("msg_cc21a0ac599b481db52475720dd9f6a7", "claude-haiku-4-5")}, StatusWarn},
		{"different model echoed warns", "claude-opus-5", []Sample{withShape("msg_01MzmDFUhapVJeHMhAUX2ZwC", "qwen3.8-max-free")}, StatusWarn},
		{"model echo ignores separators and case", "gpt-5.6-sol", []Sample{withShape("chatcmpl-x", "openai/GPT-5.6-Sol")}, StatusPass},
		{"no successful samples skips", "m", []Sample{{Probe: ProbeTiny, Err: errors.New("x")}}, StatusSkip},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ResponseShape(tt.samples, tt.claimed)
			if got.Status != tt.want {
				t.Errorf("status = %q, want %q (%s)", got.Status, tt.want, got.Summary)
			}
		})
	}
}
