package checks

import (
	"context"
	"fmt"
	"math/rand/v2"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// Signature probes: the signed request, and its conversation sent back once
// with the signature as issued and once with one byte of it changed.
// ProbeSignatureAdaptive is a signed Claude request with adaptive thinking
// that the endpoint refused, after which the request is repeated as
// ProbeSignature with manual thinking.
const (
	ProbeSignature         = "signature"
	ProbeSignatureAdaptive = "signature_adaptive"
	ProbeSignatureOriginal = "signature_original"
	ProbeSignatureTampered = "signature_tampered"
)

// SignatureProbes is how many signature round trips one run attempts: two,
// so that a problem must repeat before it fails, as the layer 2 failure
// policy requires. Each round sends the signed request and, when the answer
// carries a signature and a tool call, the conversation back twice: up to
// 3*SignatureProbes requests per run, plus one when a Claude endpoint
// refuses adaptive thinking and the request is repeated with manual
// thinking.
const SignatureProbes = 2

// signatureCities fill the signed prompt; one is drawn per run.
var signatureCities = []string{"Lisbon", "Oslo", "Montreal", "Nairobi", "Osaka", "Valparaiso", "Tbilisi", "Hobart"}

func signaturePrompt(city string) string {
	return "What is the weather in " + city + " right now? Use the " + probe.SignatureTool + " tool."
}

// signatureOutputLimits are above probe.ThinkingBudget, which Anthropic
// requires of max_tokens when thinking is enabled.
var signatureOutputLimits = []int{2048, 4096}

// signatureFamily is "claude" for a Claude model over the Anthropic
// protocol, "gemini" for Gemini 3 or later over an OpenAI-compatible API, and
// "" when no vendor signature can be checked. Gemini 2.5 does not validate
// the thought signatures it issues, so a forgery proves nothing there.
func signatureFamily(protocol probe.Protocol, model string) string {
	parts := nameParts(model)
	switch {
	case protocol == probe.ProtocolAnthropic && slices.Contains(parts, "claude"):
		return "claude"
	case protocol != probe.ProtocolAnthropic:
		if i := slices.Index(parts, "gemini"); i >= 0 && i+1 < len(parts) {
			if major, err := strconv.Atoi(parts[i+1]); err == nil && major >= 3 {
				return "gemini"
			}
		}
	}
	return ""
}

// signatureRun sends SignatureProbes rounds for models whose signatures can
// be checked and nothing for the rest.
func signatureRun(ctx context.Context, s Sender, protocol probe.Protocol, claimedModel, city string) []Sample {
	if signatureFamily(protocol, claimedModel) == "" {
		return nil
	}
	prompt := signaturePrompt(city)
	var out []Sample
	manual := false
	for i := 0; i < SignatureProbes && ctx.Err() == nil; i++ {
		// One output limit per round, so a relay that routes by it sends
		// both replies where the signature was issued.
		limit := signatureOutputLimits[rand.IntN(len(signatureOutputLimits))]
		req := probe.Request{User: prompt, MaxTokens: limit, Signed: true, ManualThinking: manual}
		first, ok := sendRequest(ctx, s, ProbeSignature, req)
		if !ok {
			break
		}
		if refusal(first) && protocol == probe.ProtocolAnthropic && !manual {
			// Claude 4.5 and earlier refuse adaptive thinking.
			first.Probe = ProbeSignatureAdaptive
			out = append(out, first)
			manual, req.ManualThinking = true, true
			if first, ok = sendRequest(ctx, s, ProbeSignature, req); !ok {
				break
			}
		}
		out = append(out, first)
		turn := first.Response.Turn
		if first.Err != nil || turn == nil || turn.Signature == "" || len(turn.ToolCalls) == 0 {
			continue
		}
		forged, err := turn.Tampered()
		if err != nil {
			continue
		}
		for _, r := range []struct {
			name string
			turn *probe.Turn
		}{{ProbeSignatureOriginal, turn}, {ProbeSignatureTampered, &forged}} {
			if ctx.Err() != nil {
				break
			}
			if sample, ok := sendRequest(ctx, s, r.name, probe.Request{User: prompt, MaxTokens: limit, Reply: r.turn}); ok {
				out = append(out, sample)
			}
		}
	}
	return out
}

func isSignatureProbe(name string) bool {
	return name == ProbeSignature || name == ProbeSignatureAdaptive || name == ProbeSignatureOriginal || name == ProbeSignatureTampered
}

// refusal reports whether the endpoint refused the request outright (HTTP
// 400 or 422) rather than failing in transport.
func refusal(s Sample) bool {
	return s.Err != nil && (s.Response.HTTPStatus == http.StatusBadRequest || s.Response.HTTPStatus == http.StatusUnprocessableEntity)
}

// answer reports whether a failed sample is still an answer of the endpoint,
// not an error of the run: a refused sent-back conversation, or the refused
// adaptive request that the run repeats with manual thinking.
func answer(s Sample) bool {
	return (s.Probe == ProbeSignatureAdaptive || s.Probe == ProbeSignatureOriginal || s.Probe == ProbeSignatureTampered) && refusal(s)
}

// aboutSignature reports whether a refusal names the signature, as vendors
// do when one does not verify ("Invalid `signature` in `thinking` block",
// "Corrupted thought signature"), rather than another part of the request.
func aboutSignature(s *Sample) bool {
	return s.Err != nil && strings.Contains(strings.ToLower(s.Err.Error()), "signature")
}

type signatureRound struct {
	first              Sample
	original, tampered *Sample
}

// signatureRounds groups samples into rounds: a signature sample and the
// sent-back conversations that follow it.
func signatureRounds(samples []Sample) []signatureRound {
	var out []signatureRound
	for i := range samples {
		s := &samples[i]
		switch {
		case s.Probe == ProbeSignature:
			out = append(out, signatureRound{first: *s})
		case len(out) == 0:
		case s.Probe == ProbeSignatureOriginal:
			out[len(out)-1].original = s
		case s.Probe == ProbeSignatureTampered:
			out[len(out)-1].tampered = s
		}
	}
	return out
}

// verdictOf is "accepted" (2xx with an answer), "rejected" (HTTP 400 or 422) or "" when the
// request was not sent or failed in transport.
func verdictOf(s *Sample) string {
	switch {
	case s == nil:
		return ""
	case s.Err == nil && (s.Response.ID != "" || s.Response.Text != "" || s.Response.Usage.OutputTokens > 0):
		// An HTTP 200 without an answer (an error body) accepted nothing.
		return "accepted"
	case refusal(*s) && s.Probe != ProbeSignature:
		return "rejected"
	}
	return ""
}

var signatureLabels = map[string]struct{ kind, vendor string }{
	"claude": {"Claude thinking", "Anthropic"},
	"gemini": {"Gemini thought", "Google"},
}

// Signature checks that a reasoning signature issued by the endpoint is
// accepted back in the next turn and that the same signature with one
// changed byte is rejected. Only the vendor can tell the two apart, so an
// endpoint that accepts the forgery is not backed by the vendor, and one
// that rejects its own signature issued it on another channel than the one
// that verified it. Its own signature counts as rejected only when the
// refusal names the signature or the forgery was accepted: an endpoint that
// refuses both conversations for another reason (tool result format, a
// limit) proves nothing and skips, or warns when other rounds completed. A
// problem in one round warns; the same problem in every round (at least
// SignatureProbes of them) fails. An answer without a
// signature warns where the vendor always signs: Claude with manual
// thinking, and Gemini 3 on its first tool call. A refused signed request
// (the endpoint does not accept thinking or tools), another model family, a
// turn without a tool call or a thinking block under adaptive thinking, and
// transport errors skip.
func Signature(samples []Sample, protocol probe.Protocol, claimedModel string) Finding {
	f := Finding{Check: "channel_signature"}
	family := signatureFamily(protocol, claimedModel)
	if family == "" {
		f.Status, f.Summary = StatusSkip, "signature round trips apply to Claude over the Anthropic protocol and Gemini over OpenAI-compatible APIs"
		return f
	}
	label := signatureLabels[family]
	rounds := signatureRounds(samples)
	var (
		answered, refused, unsigned, signed, complete int
		forgedAccepted, ownRejected, ownStatus        int
		replayRefused, problemRounds, replayStatus    int
		problems                                      []string
		evidence                                      []map[string]any
	)
	for _, r := range rounds {
		ev := map[string]any{"status": r.first.Response.HTTPStatus}
		evidence = append(evidence, ev)
		if r.first.Err != nil {
			if refusal(r.first) {
				refused++
			}
			continue
		}
		answered++
		turn := r.first.Response.Turn
		if turn == nil {
			turn = &probe.Turn{}
		}
		ev["signature_chars"], ev["tool_calls"] = len(turn.Signature), len(turn.ToolCalls)
		if turn.Signature == "" {
			// Manual thinking always produces a thinking block; adaptive
			// thinking may skip it. Gemini 3 always signs its first tool call.
			if family == "claude" && turn.ManualThinking || family == "gemini" && len(turn.ToolCalls) > 0 {
				unsigned++
			}
			continue
		}
		signed++
		original, tampered := verdictOf(r.original), verdictOf(r.tampered)
		if r.original != nil {
			ev["original"] = r.original.Response.HTTPStatus
		}
		if r.tampered != nil {
			ev["tampered"] = r.tampered.Response.HTTPStatus
		}
		ownBad := original == "rejected" && (tampered == "accepted" || aboutSignature(r.original))
		replayBad := original == "rejected" && tampered == "rejected" && !ownBad
		if ownBad {
			problems = append(problems, fmt.Sprintf("the endpoint rejected its own signature (HTTP %d)", r.original.Response.HTTPStatus))
		}
		if tampered == "accepted" {
			problems = append(problems, "the endpoint accepted a signature with one changed byte")
		}
		if replayBad {
			replayStatus = r.original.Response.HTTPStatus
			problems = append(problems, fmt.Sprintf("the endpoint refused the conversation sent back with either signature (HTTP %d) without naming the signature", replayStatus))
		}
		if ownBad || tampered == "accepted" || replayBad {
			problemRounds++
		}
		if original == "" || tampered == "" {
			continue
		}
		complete++
		switch {
		case replayBad:
			replayRefused++
		case ownBad:
			ownRejected++
			ownStatus = r.original.Response.HTTPStatus
		}
		if tampered == "accepted" {
			forgedAccepted++
		}
	}
	f.Evidence = map[string]any{"rounds": evidence, "problems": problems}
	every := func(bad int) bool { return complete >= SignatureProbes && bad == complete }
	switch {
	case answered == 0 && refused > 0 && refused == len(rounds):
		f.Status = StatusSkip
		f.Summary = fmt.Sprintf("the endpoint refused the request with thinking and a tool (HTTP %d): the model or the endpoint does not support it", rounds[0].first.Response.HTTPStatus)
	case answered == 0:
		f.Status, f.Summary = StatusSkip, "no successful signature probe"
	case every(forgedAccepted):
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("the endpoint accepted a %s signature with one changed byte in all %d round trips: it does not verify signatures, %s is not behind it", label.kind, complete, label.vendor)
	case every(ownRejected):
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("the endpoint rejected its own %s signature in all %d round trips (HTTP %d): the signature was not issued by the channel that verifies it", label.kind, complete, ownStatus)
	case complete > 0 && replayRefused == complete:
		f.Status = StatusSkip
		f.Summary = fmt.Sprintf("the endpoint does not accept the conversation sent back (HTTP %d for both signatures, the error does not name the signature)", replayStatus)
	case problemRounds > 0:
		f.Status, f.Summary = StatusWarn, fmt.Sprintf("%d of %d signature round trips: %s", problemRounds, signed, problems[0])
	case unsigned > 0:
		f.Status, f.Summary = StatusWarn, fmt.Sprintf("%d of %d answers carried no %s signature, which %s always issues for this request", unsigned, answered, label.kind, label.vendor)
	case complete == 0:
		f.Status, f.Summary = StatusSkip, "no signature round trip completed: the model called no tool or did not think, or the conversation sent back failed in transport"
	default:
		f.Status, f.Summary = StatusPass, fmt.Sprintf("%d round trips: the endpoint accepted its own %s signature and rejected one with a changed byte", complete, label.kind)
	}
	return f
}
