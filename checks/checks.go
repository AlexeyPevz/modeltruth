// Package checks turns probe responses into findings. Every check here is
// deterministic and needs no reference model: it inspects what the endpoint
// itself reports about the request it served.
package checks

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// Status is the outcome of one check.
type Status string

const (
	StatusPass Status = "pass"
	StatusWarn Status = "warn"
	StatusFail Status = "fail"
	StatusSkip Status = "skip"
	// StatusError means no request succeeded, so nothing was checked.
	StatusError Status = "error"
)

// Finding is the result of one check with the evidence behind it.
type Finding struct {
	Check    string         `json:"check"`
	Status   Status         `json:"status"`
	Summary  string         `json:"summary"`
	Evidence map[string]any `json:"evidence,omitempty"`
}

// Sample is one probe request and what came back.
type Sample struct {
	Probe       string
	PromptChars int
	Response    probe.Response
	Err         error
}

// Thresholds for the tiny prompt. A bare "Reply with the single word: ok"
// costs roughly 8-20 input tokens on every tokenizer and chat template we
// have measured; a few hundred tokens means text was added to the request.
const (
	hiddenPromptWarnTokens = 60
	hiddenPromptFailTokens = 300
)

// Two identical requests to one model must report nearly the same input size.
// The absolute floor keeps tokenizer rounding on tiny prompts from tripping it.
const (
	channelSpreadRatio  = 1.10
	channelSpreadTokens = 20
)

// usageFloorCharsPerToken is a deliberately loose bound: English text never
// tokenizes to fewer than one token per six characters.
const usageFloorCharsPerToken = 6

var anthropicMessageID = regexp.MustCompile(`^msg_(vrtx_|bdrk_)?01[A-Za-z0-9]{22}$`)

func ok(samples []Sample, probeName string) []probe.Response {
	var out []probe.Response
	for _, s := range samples {
		if s.Err == nil && (probeName == "" || s.Probe == probeName) {
			out = append(out, s.Response)
		}
	}
	return out
}

// promptChars is the prompt length of the first successful sample of a probe;
// every request of one probe in a run carries the same prompt.
func promptChars(samples []Sample, probeName string) int {
	for _, s := range samples {
		if s.Err == nil && s.Probe == probeName {
			return s.PromptChars
		}
	}
	return 0
}

func inputs(responses []probe.Response) []int64 {
	out := make([]int64, len(responses))
	for i, r := range responses {
		out[i] = r.Usage.InputTokens
	}
	return out
}

// HiddenPrompt fails when the tiny prompt is billed as far more input than it
// contains, which means the relay prepends its own system prompt.
func HiddenPrompt(samples []Sample) Finding {
	f := Finding{Check: "hidden_prompt"}
	responses := ok(samples, ProbeTiny)
	if len(responses) == 0 {
		f.Status, f.Summary = StatusSkip, "no successful tiny-prompt responses"
		return f
	}
	counts := inputs(responses)
	var maxInput int64
	inflated := 0
	for _, c := range counts {
		maxInput = max(maxInput, c)
		if c > hiddenPromptFailTokens {
			inflated++
		}
	}
	chars := promptChars(samples, ProbeTiny)
	f.Evidence = map[string]any{"input_tokens": counts, "prompt_chars": chars}
	switch {
	case maxInput > hiddenPromptFailTokens:
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("a %d-character prompt was billed as up to %d input tokens in %d of %d requests: the endpoint adds a hidden prompt", chars, maxInput, inflated, len(counts))
	case maxInput > hiddenPromptWarnTokens:
		f.Status = StatusWarn
		f.Summary = fmt.Sprintf("tiny prompt billed as up to %d input tokens; expected under %d", maxInput, hiddenPromptWarnTokens)
	default:
		f.Status = StatusPass
		f.Summary = fmt.Sprintf("tiny prompt billed as at most %d input tokens", maxInput)
	}
	return f
}

// ChannelConsistency fails when identical requests report different input
// sizes, a sign that one model name is served by several upstream channels.
func ChannelConsistency(samples []Sample) Finding {
	f := Finding{Check: "channel_consistency"}
	evidence := map[string]any{}
	compared := 0
	var worst string
	for _, name := range []string{ProbeTiny, ProbeMedium} {
		counts := inputs(ok(samples, name))
		if len(counts) < 2 {
			continue
		}
		compared++
		evidence[name] = counts
		lo, hi := counts[0], counts[0]
		for _, c := range counts {
			lo, hi = min(lo, c), max(hi, c)
		}
		if float64(hi) > float64(lo)*channelSpreadRatio && hi-lo > channelSpreadTokens && worst == "" {
			worst = fmt.Sprintf("identical %s prompts were billed as %d to %d input tokens: requests reach different upstream channels", name, lo, hi)
		}
	}
	f.Evidence = evidence
	switch {
	case compared == 0:
		f.Status, f.Summary = StatusSkip, "need at least two successful responses to one prompt"
	case worst != "":
		f.Status, f.Summary = StatusFail, worst
	default:
		f.Status, f.Summary = StatusPass, "identical prompts report consistent input sizes"
	}
	return f
}

// UsageSanity fails when reported usage is lower than the prompt can possibly
// tokenize to, and warns when text came back without any output tokens.
func UsageSanity(samples []Sample) Finding {
	f := Finding{Check: "usage_sanity"}
	responses := ok(samples, ProbeMedium)
	if len(responses) == 0 {
		f.Status, f.Summary = StatusSkip, "no successful medium-prompt responses"
		return f
	}
	chars := promptChars(samples, ProbeMedium)
	floor := int64(chars / usageFloorCharsPerToken)
	counts := inputs(responses)
	f.Evidence = map[string]any{"input_tokens": counts, "input_floor": floor, "prompt_chars": chars}
	var under []int64
	missingOutput := 0
	for _, r := range responses {
		if r.Usage.InputTokens < floor {
			under = append(under, r.Usage.InputTokens)
		}
		if strings.TrimSpace(r.Text) != "" && r.Usage.OutputTokens == 0 {
			missingOutput++
		}
	}
	switch {
	case len(under) > 0:
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("a %d-character prompt was reported as %v input tokens, below the minimum of %d: usage is not real", chars, under, floor)
	case missingOutput > 0:
		f.Status = StatusWarn
		f.Summary = fmt.Sprintf("%d responses returned text but reported zero output tokens", missingOutput)
	default:
		f.Status, f.Summary = StatusPass, "reported usage is plausible for the prompt"
	}
	return f
}

// ResponseShape warns when ids or the echoed model name do not look like the
// claimed provider. Gateways legitimately rewrite both, so this never fails.
func ResponseShape(samples []Sample, claimedModel string) Finding {
	f := Finding{Check: "response_shape"}
	responses := ok(samples, "")
	if len(responses) == 0 {
		f.Status, f.Summary = StatusSkip, "no successful responses"
		return f
	}
	var problems, badIDs []string
	models := map[string]bool{}
	for _, r := range responses {
		if strings.HasPrefix(r.ID, "msg_") && !anthropicMessageID.MatchString(r.ID) {
			badIDs = append(badIDs, r.ID)
		}
		if r.Model != "" && !sameModel(claimedModel, r.Model) && !models[r.Model] {
			models[r.Model] = true
			problems = append(problems, fmt.Sprintf("response names model %q, requested %q", r.Model, claimedModel))
		}
	}
	if len(badIDs) > 0 {
		problems = append([]string{fmt.Sprintf("%d of %d message ids are not in Anthropic format (e.g. %q)", len(badIDs), len(responses), badIDs[0])}, problems...)
	}
	f.Evidence = map[string]any{"problems": problems, "non_anthropic_ids": badIDs}
	if len(problems) > 0 {
		f.Status, f.Summary = StatusWarn, strings.Join(problems, "; ")
		return f
	}
	f.Status, f.Summary = StatusPass, "ids and model echo are consistent with the claim"
	return f
}

// sameModel reports whether the claimed name appears as a contiguous run of
// name parts inside the echoed one, so "openai/GPT-5.6-Sol" matches
// "gpt-5.6-sol" and a gateway prefix or suffix does not break the match.
func sameModel(claimed, echoed string) bool {
	want, got := nameParts(claimed), nameParts(echoed)
	if len(want) == 0 {
		return true
	}
	for i := 0; i+len(want) <= len(got); i++ {
		match := true
		for j := range want {
			if got[i+j] != want[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func nameParts(name string) []string {
	return strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
}
