package checks

import (
	"context"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// Sender sends one probe request. probe.Endpoint implements it.
type Sender interface {
	Send(ctx context.Context, req probe.Request) (probe.Response, error)
}

// Report is the outcome of one run against one model.
type Report struct {
	Model    string    `json:"model"`
	Verdict  Status    `json:"verdict"`
	Requests int       `json:"requests"`
	Findings []Finding `json:"findings"`
}

// Run sends the tiny and medium prompts repeats times each, interleaved so a
// time-based channel switch shows up in both, and evaluates every check.
func Run(ctx context.Context, s Sender, claimedModel string, repeats int) Report {
	if repeats < 1 {
		repeats = 1
	}
	prompts := []struct {
		name string
		req  probe.Request
	}{
		{ProbeTiny, probe.Request{User: tinyPromptText, MaxTokens: 16}},
		{ProbeMedium, probe.Request{User: mediumPromptText, MaxTokens: 16}},
	}
	var samples []Sample
	for i := 0; i < repeats; i++ {
		for _, p := range prompts {
			if ctx.Err() != nil {
				break
			}
			resp, err := s.Send(ctx, p.req)
			samples = append(samples, Sample{Probe: p.name, Response: resp, Err: err})
		}
	}
	findings := []Finding{
		HiddenPrompt(samples),
		ChannelConsistency(samples),
		UsageSanity(samples),
		ResponseShape(samples, claimedModel),
	}
	return Report{Model: claimedModel, Verdict: Verdict(findings), Requests: len(samples), Findings: findings}
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
