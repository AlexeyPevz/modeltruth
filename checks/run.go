package checks

import (
	"context"
	"math/rand/v2"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// Sender sends one probe request. probe.Endpoint implements it.
type Sender interface {
	Send(ctx context.Context, req probe.Request) (probe.Response, error)
}

// SampleError is one request that did not produce a usable response.
type SampleError struct {
	Probe      string `json:"probe"`
	HTTPStatus int    `json:"http_status,omitempty"`
	Error      string `json:"error"`
}

// Report is the outcome of one run against one model.
type Report struct {
	Model      string        `json:"model"`
	Verdict    Status        `json:"verdict"`
	Requests   int           `json:"requests"`
	Successful int           `json:"successful"`
	Errors     []SampleError `json:"errors,omitempty"`
	Findings   []Finding     `json:"findings"`
}

// Run sends the tiny and medium prompts repeats times each, interleaved so a
// time-based channel switch shows up in both, and evaluates every check. The
// answer word is drawn per run; see answerWords.
func Run(ctx context.Context, s Sender, claimedModel string, repeats int) Report {
	return runWithWord(ctx, s, claimedModel, repeats, answerWords[rand.IntN(len(answerWords))])
}

func runWithWord(ctx context.Context, s Sender, claimedModel string, repeats int, word string) Report {
	if repeats < 1 {
		repeats = 1
	}
	prompts := []struct {
		name string
		text string
	}{
		{ProbeTiny, tinyPrompt(word)},
		{ProbeMedium, mediumPrompt(word)},
	}
	var samples []Sample
	for i := 0; i < repeats; i++ {
		for _, p := range prompts {
			if ctx.Err() != nil {
				break
			}
			resp, err := s.Send(ctx, probe.Request{User: p.text, MaxTokens: 16})
			samples = append(samples, Sample{Probe: p.name, PromptChars: len(p.text), Response: resp, Err: err})
		}
	}
	rep := Report{Model: claimedModel, Requests: len(samples)}
	for _, sample := range samples {
		if sample.Err != nil {
			rep.Errors = append(rep.Errors, SampleError{Probe: sample.Probe, HTTPStatus: sample.Response.HTTPStatus, Error: sample.Err.Error()})
			continue
		}
		rep.Successful++
	}
	rep.Findings = []Finding{
		HiddenPrompt(samples),
		ChannelConsistency(samples),
		UsageSanity(samples),
		ResponseShape(samples, claimedModel),
	}
	rep.Verdict = Verdict(rep.Findings)
	if rep.Successful == 0 {
		// An endpoint that answered nothing must not read as a clean or
		// skipped check; transport errors are still never evidence of fraud.
		rep.Verdict = StatusError
	}
	return rep
}

// Verdict is the worst status among findings; skips count only when nothing
// else ran.
func Verdict(findings []Finding) Status {
	rank := map[Status]int{StatusSkip: 0, StatusPass: 1, StatusWarn: 2, StatusFail: 3}
	worst := StatusSkip
	for _, f := range findings {
		if rank[f.Status] > rank[worst] {
			worst = f.Status
		}
	}
	return worst
}
