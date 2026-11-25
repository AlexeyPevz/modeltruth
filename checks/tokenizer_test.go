package checks

import (
	"strings"
	"testing"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// tokenizerSamples builds one run of tokenizer probes: the base prompt twice
// and every text once, billed as base plus the given deltas (in
// tokenizerTexts order; a negative value is sent as is).
func tokenizerSamples(base1, base2 int64, deltas []int64) []Sample {
	s := []Sample{{Probe: ProbeTokenizerBase, Response: probe.Response{Usage: probe.Usage{InputTokens: base1}}}}
	for i, t := range tokenizerTexts {
		s = append(s, Sample{Probe: probeTokenizerPrefix + t.name, Response: probe.Response{Usage: probe.Usage{InputTokens: base1 + deltas[i]}}})
	}
	return append(s, Sample{Probe: ProbeTokenizerBase, Response: probe.Response{Usage: probe.Usage{InputTokens: base2}}})
}

func profileDeltas(t *testing.T, name string) []int64 {
	t.Helper()
	for _, p := range tokenizerBank {
		if p.name == name {
			out := make([]int64, len(p.deltas))
			for i, d := range p.deltas {
				out[i] = int64(d)
			}
			return out
		}
	}
	t.Fatalf("no profile %q", name)
	return nil
}

func with(d []int64, i int, v int64) []int64 {
	out := append([]int64(nil), d...)
	out[i] = v
	return out
}

func TestTokenizer(t *testing.T) {
	o200k := profileDeltas(t, "o200k")
	claude := profileDeltas(t, "claude")
	// Channel 687 sold as claude-haiku-4-5 on 2026-10-28: plausible counts,
	// but no Claude tokenizer bills 102 characters of rare words as 16 tokens.
	ch687 := []int64{15, 16, 16, 37, 14, 10, 27, 42, 34, 42, 57, 30}
	// Channels 3825 and 3755: word-count usage, 50 digits billed as 4 tokens.
	wordCount := []int64{26, 17, 7, 55, 4, 19, 21, 46, 23, 52, 90, 52}
	// Channel 2863 sold as claude-sonnet-5 on 2026-10-28: the Claude
	// tokenizer with every count about 10% higher.
	ch2863 := []int64{21, 18, 50, 52, 20, 11, 30, 51, 42, 51, 71, 39}
	o200kPlus25 := make([]int64, len(o200k))
	for i, d := range o200k {
		o200kPlus25[i] = (d*5 + 2) / 4
	}
	tests := []struct {
		name     string
		model    string
		samples  []Sample
		want     Status
		contains string
	}{
		{"OpenAI model billed like o200k passes", "gpt-5.4", tokenizerSamples(10, 10, o200k), StatusPass, "o200k"},
		{"one noisy text is tolerated", "gpt-6-astra", tokenizerSamples(10, 10, with(o200k, 9, 40)), StatusPass, "o200k"},
		{"gpt-6-luna billed like cl100k fails", "gpt-6-luna", tokenizerSamples(11, 11, profileDeltas(t, "cl100k")), StatusFail, "cl100k"},
		{"legacy gpt-4 billed like cl100k passes", "gpt-4", tokenizerSamples(11, 11, profileDeltas(t, "cl100k")), StatusPass, "cl100k"},
		{"GLM model billed like DeepSeek fails", "glm-5.3-flashx", tokenizerSamples(871, 871, profileDeltas(t, "deepseek-v3")), StatusFail, "deepseek-v3"},
		{"Gemini model billed like o200k fails", "gemini-4-argon", tokenizerSamples(4, 4, o200k), StatusFail, "o200k"},
		{"Gemini model billed like Gemma 3 passes", "gemini-4-argon", tokenizerSamples(34, 34, profileDeltas(t, "gemma3")), StatusPass, "gemma3"},
		{"Claude billed like the Claude tokenizer passes", "claude-opus-5-5", tokenizerSamples(15, 15, claude), StatusPass, "claude"},
		{"Claude matching no known tokenizer fails", "claude-sonnet-5", tokenizerSamples(15, 15, ch687), StatusFail, "no known anthropic tokenizer"},
		{"Claude 4.x without a reference is skipped", "claude-haiku-4-5-20251001", tokenizerSamples(15, 15, ch687), StatusSkip, "no reference anthropic-4 tokenizer"},
		{"Claude 4.x billed like o200k fails", "claude-sonnet-4-6", tokenizerSamples(10, 10, o200k), StatusFail, "o200k (openai), not a anthropic-4 tokenizer"},
		{"Claude 4.x billed like Claude 5 fails", "claude-haiku-4-5", tokenizerSamples(15, 15, claude), StatusFail, "claude (anthropic), not a anthropic-4 tokenizer"},
		{"word-count usage fails for Claude 4.x", "claude-sonnet-4-6", tokenizerSamples(15, 15, wordCount), StatusFail, "not computed by a tokenizer"},
		{"Claude billed 10% above the Claude tokenizer is inflated", "claude-sonnet-5", tokenizerSamples(14, 14, ch2863), StatusFail, "claude tokenizer inflated by 10%"},
		{"GPT billed 25% above o200k is inflated", "gpt-6-luna", tokenizerSamples(10, 10, o200kPlus25), StatusFail, "o200k tokenizer inflated by 25%"},
		{"word-count usage fails for any model", "kimi-k3", tokenizerSamples(15, 15, wordCount), StatusFail, "not computed by a tokenizer"},
		{"zero deltas fail", "claude-opus-5-5", tokenizerSamples(3, 3, make([]int64, len(tokenizerTexts))), StatusFail, "not computed by a tokenizer"},
		{"a consistent hidden prompt cancels out", "gpt-6-luna", tokenizerSamples(4390, 4390, o200k), StatusPass, "o200k"},
		{"content-dependent injection is not evidence", "gpt-6.1-sol", tokenizerSamples(4390, 4390, []int64{-4366, -4379, 28, 36, -4371, -4378, -4328, -4348, -4334, -4315, -4356, 27}), StatusSkip, "texts"},
		{"unstable base is skipped", "gpt-5.4", tokenizerSamples(798, 35, o200k), StatusSkip, "base"},
		{"model without a reference is skipped", "kimi-k3", tokenizerSamples(10, 10, o200k), StatusSkip, "o200k"},
		{"no tokenizer probes is skipped", "gpt-5.4", nil, StatusSkip, "base"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := Tokenizer(tt.samples, tt.model)
			if f.Status != tt.want {
				t.Fatalf("status = %q, want %q: %s", f.Status, tt.want, f.Summary)
			}
			if !strings.Contains(f.Summary, tt.contains) {
				t.Errorf("summary %q does not mention %q", f.Summary, tt.contains)
			}
		})
	}
}

func TestTokenizerFamily(t *testing.T) {
	tests := []struct{ model, want string }{
		{"gpt-5.4", "openai"}, {"gpt-4o-mini", "openai"}, {"gpt-4.1-mini", "openai"}, {"openai/GPT-6-Luna", "openai"},
		{"o3-mini", "openai"}, {"gpt-4", "openai-legacy"}, {"gpt-4-turbo", "openai-legacy"}, {"gpt-3.5-turbo", "openai-legacy"},
		{"claude-opus-5-5", "anthropic"}, {"[SP]claude-sonnet-5-5", "anthropic"}, {"gemini-4-argon", "google"},
		{"claude-haiku-4-5-20251001", "anthropic-4"}, {"claude-haiku-4-5-20251001-r", "anthropic-4"}, {"claude-sonnet-4-6", "anthropic-4"},
		{"claude-sonnet-4.6", "anthropic-4"}, {"claude-sonnet-4-20250514", "anthropic-4"}, {"anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic-4"},
		{"claude-opus-4-1-20250805", "anthropic-4"}, {"claude-opus-4-6-thinking", "anthropic-4"},
		{"claude-opus-4-7", "anthropic"}, {"claude-opus-4-8-1m", "anthropic"}, {"claude-sonnet-5", "anthropic"},
		{"[free]deepseek-v4.1-flash", "deepseek"}, {"glm-5.3-flashx", "zhipu"}, {"qwen3.8-flash", "qwen"},
		{"kimi-k3", ""}, {"grok-4.7", ""},
	}
	for _, tt := range tests {
		if got := tokenizerFamily(tt.model); got != tt.want {
			t.Errorf("tokenizerFamily(%q) = %q, want %q", tt.model, got, tt.want)
		}
	}
}

// TestTokenizerClaude4Reference runs the check with a stand-in anthropic-4
// profile (Claude 5 minus three tokens on every text) in the bank, the way
// the measured Claude 4.x reference will be judged once it is added.
func TestTokenizerClaude4Reference(t *testing.T) {
	claude := profileDeltas(t, "claude")
	ref := make([]int, len(claude))
	official := make([]int64, len(claude))
	for i, d := range claude {
		ref[i], official[i] = int(d)-3, d-3
	}
	inflated := make([]int64, len(official))
	for i, d := range official {
		inflated[i] = (d*23 + 10) / 20
	}
	saved := tokenizerBank
	tokenizerBank = append(append([]tokenizerProfile(nil), saved...), tokenizerProfile{"claude-4", "anthropic-4", "stand-in", ref})
	t.Cleanup(func() { tokenizerBank = saved })
	tests := []struct {
		name     string
		model    string
		samples  []Sample
		want     Status
		contains string
	}{
		{"official Claude 4.x counts pass", "claude-haiku-4-5-20251001", tokenizerSamples(15, 15, official), StatusPass, "claude-4"},
		{"one noisy text is tolerated", "claude-sonnet-4-6", tokenizerSamples(15, 15, with(official, 3, 60)), StatusPass, "claude-4"},
		{"GPT tokenizer sold as Claude 4.x fails", "claude-sonnet-4-6", tokenizerSamples(10, 10, profileDeltas(t, "o200k")), StatusFail, "o200k"},
		{"Claude 5 tokenizer sold as Claude 4.x fails", "claude-haiku-4-5", tokenizerSamples(15, 15, claude), StatusFail, "not a anthropic-4 tokenizer"},
		{"Claude 4.x counts sold as Claude 5 fail", "claude-opus-5-5", tokenizerSamples(15, 15, official), StatusFail, "claude-4 (anthropic-4)"},
		{"Claude 4.x billed 15% above the reference is inflated", "claude-sonnet-4-6", tokenizerSamples(15, 15, inflated), StatusFail, "claude-4 tokenizer inflated by 15%"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := Tokenizer(tt.samples, tt.model)
			if f.Status != tt.want {
				t.Fatalf("status = %q, want %q: %s", f.Status, tt.want, f.Summary)
			}
			if !strings.Contains(f.Summary, tt.contains) {
				t.Errorf("summary %q does not mention %q", f.Summary, tt.contains)
			}
		})
	}
}

func TestTokenizerBank(t *testing.T) {
	if TokenizerProbes != len(tokenizerTexts)+2 {
		t.Fatalf("TokenizerProbes = %d, want %d", TokenizerProbes, len(tokenizerTexts)+2)
	}
	for i, a := range tokenizerBank {
		if len(a.deltas) != len(tokenizerTexts) {
			t.Fatalf("profile %s has %d deltas, want %d", a.name, len(a.deltas), len(tokenizerTexts))
		}
		// Two profiles must differ on several texts, or one channel could
		// match both and the verdict would depend on bank order.
		for _, b := range tokenizerBank[i+1:] {
			if n := outliers(toInt64(a.deltas), b.deltas); n <= maxTokenizerOutliers+1 {
				t.Errorf("profiles %s and %s differ on only %d texts", a.name, b.name, n)
			}
		}
	}
}

func toInt64(d []int) map[int]int64 {
	out := map[int]int64{}
	for i, v := range d {
		out[i] = int64(v)
	}
	return out
}
