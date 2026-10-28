package checks

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Tokenizer probe names: the base prompt is sent twice (first and last), every
// text once, as the base prompt plus a newline plus the text.
const (
	ProbeTokenizerBase   = "tok_base"
	probeTokenizerPrefix = "tok_"
)

// TokenizerProbes is how many requests the tokenizer check adds to one run.
const TokenizerProbes = 14

// tokenizerBase is short and plain so that every tokenizer bills it as a few
// tokens and the difference to base+text is the text alone.
const tokenizerBase = "Reply with OK."

// tokenizerTexts separate tokenizer families: scripts, emoji, long rare words,
// code, digit runs and whitespace are where vocabularies differ most. The order
// is the order of every profile in tokenizerBank.
var tokenizerTexts = []struct{ name, text string }{
	{"ja", "東京都庁の展望台から富士山が見えた。"},
	{"emoji", "🙂🚀🧪🦀🧇"},
	{"rare", "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis floccinaucinihilipilification"},
	{"code", "func main() { for i := 0; i < 10; i++ { fmt.Printf(\"%d\\t%x\\n\", i<<2, i^7) } }"},
	{"digits", "31415926535897932384626433832795028841971693993751"},
	{"ws", "a\n\n\n    b\t\t c          d"},
	{"ru", "Съешь же ещё этих мягких французских булок, да выпей чаю."},
	{"ko", "다람쥐 헌 쳇바퀴에 타고파. 키스의 고유조건은 입술끼리 만나야 하고"},
	{"ar", "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر"},
	{"hi", "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा रावण का सर्वनाश करने वाले"},
	{"math", "∑_{i=1}^{n} i² = n(n+1)(2n+1)/6 ∀n∈ℕ, ∫₀^∞ e^{−x²} dx = √π/2"},
	{"url", "https://example.com/api/v2/users?id=8f3a9c&sort=desc&page=17#section-4"},
}

func tokenizerPrompt(text string) string {
	return tokenizerBase + "\n" + text
}

// tokenizerProfile is the input-token difference each text adds to the base
// prompt under one tokenizer. Chat templates and relay system prompts add a
// constant, so the differences are the same on every honest channel.
type tokenizerProfile struct {
	name   string
	family string
	source string
	deltas []int
}

// tokenizerBank, measured 2026-10-28. Public tokenizers were counted offline
// and matched live channels exactly; closed ones are taken only where two
// independent sellers agree.
var tokenizerBank = []tokenizerProfile{
	{"o200k", "openai", "tiktoken o200k_base; equal on 14 A6 channels (gpt-5.4, gpt-5.6-sol, gpt-6-astra, gpt-6.1-terra) and LinkAPI gpt-4.1-mini", []int{16, 12, 28, 36, 17, 8, 19, 32, 21, 26, 49, 27}},
	{"cl100k", "openai-legacy", "tiktoken cl100k_base", []int{22, 14, 30, 36, 17, 8, 36, 49, 39, 67, 50, 27}},
	{"claude", "anthropic", "LinkAPI claude-opus-5-5, claude-sonnet-5-5, claude-sonnet-5 and A6 channels 6481, 6328, 6484", []int{19, 16, 47, 48, 18, 10, 27, 46, 38, 46, 65, 36}},
	{"claude-2", "anthropic-legacy", "Xenova/claude-tokenizer (Claude 1 and 2)", []int{24, 15, 32, 40, 23, 8, 40, 51, 53, 76, 51, 34}},
	{"gemma3", "google", "Gemma 3 tokenizer; equal on A6 gemini-4-argon 4803, gemini-3.8-flash 1551, 3844 and LinkAPI gemini-3.8-flash", []int{12, 6, 27, 42, 51, 9, 19, 29, 24, 22, 45, 35}},
	{"gemini-b", "google", "A6 gemini-4-argon channels 1508 and 6430; digits tokenized one by one like Gemma", []int{11, 5, 25, 42, 51, 9, 19, 25, 22, 24, 44, 35}},
	{"deepseek-v3", "deepseek", "deepseek-ai/DeepSeek-V3 tokenizer; equal on 13 A6 deepseek-v4.1 channels", []int{13, 11, 29, 37, 17, 8, 22, 36, 24, 35, 45, 28}},
	{"glm-4", "zhipu", "THUDM/glm-4-9b-chat-hf and zai-org/GLM-4.5 tokenizers", []int{14, 14, 30, 36, 31, 8, 20, 40, 26, 67, 48, 27}},
	{"qwen", "qwen", "Qwen2.5 and Qwen3 tokenizers", []int{15, 6, 30, 37, 50, 8, 24, 32, 25, 63, 46, 28}},
	{"llama3", "meta", "Llama 3.2 tokenizer", []int{16, 14, 29, 36, 17, 8, 19, 30, 26, 40, 49, 27}},
	{"mistral-nemo", "mistral", "Mistral-Nemo-Instruct-2407 tokenizer", []int{14, 20, 29, 37, 50, 8, 21, 28, 20, 27, 50, 29}},
	{"minimax-m1", "minimax", "MiniMax-M1 tokenizer", []int{10, 11, 28, 36, 17, 8, 21, 29, 20, 38, 45, 28}},
}

// A text matches a profile within one token (template edge effects), and one
// text may miss: a relay that injects into a single request should not decide
// the verdict.
const (
	tokenizerTolerance   = 1
	maxTokenizerOutliers = 1
	minTokenizerTexts    = 8
)

// tokenizerFloor is half the smallest count any known tokenizer gives a text.
// A count below it is not produced by a tokenizer at all: such relays estimate
// usage from words or characters (50 digits billed as 4 tokens).
func tokenizerFloor(i int) int64 {
	lowest := tokenizerBank[0].deltas[i]
	for _, p := range tokenizerBank[1:] {
		lowest = min(lowest, p.deltas[i])
	}
	return int64((lowest + 1) / 2)
}

// tokenizerFamily maps a claimed model name to the family whose tokenizer it
// must use; "" means no reference.
func tokenizerFamily(model string) string {
	name := strings.ToLower(model)
	parts := nameParts(name)
	switch {
	case strings.Contains(name, "claude"):
		return "anthropic"
	case strings.Contains(name, "gemini"), strings.Contains(name, "gemma"):
		return "google"
	case strings.Contains(name, "deepseek"):
		return "deepseek"
	case strings.Contains(name, "glm"):
		return "zhipu"
	case strings.Contains(name, "qwen"):
		return "qwen"
	case strings.Contains(name, "minimax"):
		return "minimax"
	}
	for i, p := range parts {
		switch {
		case p == "gpt" && i+1 < len(parts):
			next := parts[i+1]
			// gpt-3.5 and gpt-4 / gpt-4-turbo use cl100k; gpt-4o, gpt-4.1
			// and every later model use o200k.
			if next == "3" || (next == "4" && (i+2 == len(parts) || parts[i+2] == "turbo" || len(parts[i+2]) == 4)) {
				return "openai-legacy"
			}
			return "openai"
		case p == "o1" || p == "o3" || p == "o4" || p == "chatgpt" || p == "codex":
			return "openai"
		}
	}
	return ""
}

// outliers counts the valid texts whose difference is further than the
// tolerance from the profile.
func outliers(deltas map[int]int64, profile []int) int {
	n := 0
	for i, d := range deltas {
		if diff := d - int64(profile[i]); diff > tokenizerTolerance || diff < -tokenizerTolerance {
			n++
		}
	}
	return n
}

// Tokenizer identifies the tokenizer behind reported usage. The base prompt
// and base+text are billed by the same chat template and system prompt, so
// their difference is the text's token count under the serving model's
// tokenizer: a GPT sold as Gemini, an old GPT-4 sold as GPT-6 or a DeepSeek
// sold as GLM is billed with its own tokenizer, and fabricated usage matches
// none. Only families with a reference in tokenizerBank are judged.
func Tokenizer(samples []Sample, claimedModel string) Finding {
	f := Finding{Check: "tokenizer"}
	bases := inputs(ok(samples, ProbeTokenizerBase))
	if len(bases) < 2 {
		f.Status, f.Summary = StatusSkip, "the base prompt did not get two answers"
		return f
	}
	lo, hi := bases[0], bases[0]
	for _, b := range bases {
		lo, hi = min(lo, b), max(hi, b)
	}
	deltas := map[string]int64{}
	f.Evidence = map[string]any{"base_input_tokens": bases, "deltas": deltas}
	if hi-lo > tokenizerTolerance {
		f.Status = StatusSkip
		f.Summary = fmt.Sprintf("the base prompt was billed as %d and %d input tokens, so differences cannot be compared", lo, hi)
		return f
	}
	// A base that already carries a hidden prompt makes a negative difference
	// mean "this request was not injected", not "usage is fake".
	injected := lo > hiddenPromptAllowance(len(tokenizerBase))+hiddenPromptFailTokens
	valid := map[int]int64{}
	var implausible []string
	for i, t := range tokenizerTexts {
		rs := ok(samples, probeTokenizerPrefix+t.name)
		if len(rs) == 0 {
			continue
		}
		d := rs[0].Usage.InputTokens - lo
		deltas[t.name] = d
		if d > int64(3*utf8.RuneCountInString(t.text)+40) {
			continue // a hidden prompt landed on this request only
		}
		if d < tokenizerFloor(i) {
			if !injected {
				implausible = append(implausible, fmt.Sprintf("%s (%d characters) %d", t.name, utf8.RuneCountInString(t.text), d))
			}
			continue
		}
		valid[i] = d
	}
	if len(implausible) > 0 {
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("usage is not computed by a tokenizer: texts were billed below any known tokenizer: %s tokens", strings.Join(implausible, ", "))
		return f
	}
	if len(valid) < minTokenizerTexts {
		f.Status = StatusSkip
		f.Summary = fmt.Sprintf("only %d of %d texts were billed without hidden-prompt interference", len(valid), len(tokenizerTexts))
		return f
	}
	var match *tokenizerProfile
	closest, closestMiss := &tokenizerBank[0], len(tokenizerTexts)+1
	for i := range tokenizerBank {
		p := &tokenizerBank[i]
		miss := outliers(valid, p.deltas)
		if miss < closestMiss {
			closest, closestMiss = p, miss
		}
		if miss <= maxTokenizerOutliers && match == nil {
			match = p
		}
	}
	family := tokenizerFamily(claimedModel)
	if match != nil {
		f.Evidence["match"] = match.name
	}
	f.Evidence["closest"], f.Evidence["closest_mismatches"], f.Evidence["family"] = closest.name, closestMiss, family
	switch {
	case family == "" && match != nil:
		f.Status, f.Summary = StatusSkip, fmt.Sprintf("no reference tokenizer for %q; usage is billed like %s (%s)", claimedModel, match.name, match.family)
	case family == "":
		f.Status, f.Summary = StatusSkip, fmt.Sprintf("no reference tokenizer for %q; usage matches no known tokenizer (closest %s)", claimedModel, closest.name)
	case match != nil && match.family == family:
		f.Status, f.Summary = StatusPass, fmt.Sprintf("usage is billed like %s, the %s tokenizer", match.name, family)
	case match != nil:
		f.Status, f.Summary = StatusFail, fmt.Sprintf("%q is billed like %s (%s), not a %s tokenizer: another model serves this name", claimedModel, match.name, match.family, family)
	default:
		f.Status, f.Summary = StatusFail, fmt.Sprintf("usage matches no known %s tokenizer (closest %s differs on %d of %d texts): another model serves this name", family, closest.name, closestMiss, len(valid))
	}
	return f
}
