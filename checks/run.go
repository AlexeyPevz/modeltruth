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
// HTTPStatus is the status of the last attempt.
type SampleError struct {
	Probe      string `json:"probe"`
	HTTPStatus int    `json:"http_status,omitempty"`
	Retries    int    `json:"retries,omitempty"`
	Error      string `json:"error"`
}

// Report is the outcome of one run against one model. Retries is the total
// number of repeated attempts across all requests; Interrupted is set when
// the run was cancelled and the report covers only the requests answered so far.
type Report struct {
	Model       string        `json:"model"`
	Verdict     Status        `json:"verdict"`
	Requests    int           `json:"requests"`
	Successful  int           `json:"successful"`
	Retries     int           `json:"retries,omitempty"`
	Interrupted bool          `json:"interrupted,omitempty"`
	Errors      []SampleError `json:"errors,omitempty"`
	Findings    []Finding     `json:"findings"`
}

// Run sends the tiny, medium and TaskProbes task prompts repeats times each,
// interleaved so a time-based channel switch shows up in all of them, then the
// tiny prompt StreamProbes times as a stream, then the TokenizerProbes
// tokenizer prompts once, and evaluates every check. The answer word and the task prompts are drawn per
// run, the output limit per request; see answerWords, taskPrompts and
// outputLimits.
func Run(ctx context.Context, s Sender, claimedModel string, repeats int) Report {
	return runWith(ctx, s, claimedModel, repeats, answerWords[rand.IntN(len(answerWords))], drawTasks(TaskProbes))
}

func runWith(ctx context.Context, s Sender, claimedModel string, repeats int, word string, tasks []string) Report {
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
	for i, t := range tasks {
		prompts = append(prompts, struct {
			name string
			text string
		}{taskProbe(i), t})
	}
	var samples []Sample
	for i := 0; i < repeats; i++ {
		for _, p := range prompts {
			if ctx.Err() != nil {
				break
			}
			if sample, ok := send(ctx, s, p.name, p.text, false); ok {
				samples = append(samples, sample)
			}
		}
	}
	for i := 0; i < StreamProbes && ctx.Err() == nil; i++ {
		if sample, ok := send(ctx, s, ProbeStream, tinyPrompt(word), true); ok {
			samples = append(samples, sample)
		}
	}
	samples = append(samples, tokenizerRun(ctx, s)...)
	rep := Report{Model: claimedModel, Requests: len(samples), Interrupted: ctx.Err() != nil}
	for _, sample := range samples {
		rep.Retries += sample.Response.Retries
		if sample.Err != nil {
			rep.Errors = append(rep.Errors, SampleError{Probe: sample.Probe, HTTPStatus: sample.Response.HTTPStatus, Retries: sample.Response.Retries, Error: sample.Err.Error()})
			continue
		}
		rep.Successful++
	}
	rep.Findings = []Finding{
		HiddenPrompt(samples),
		ChannelConsistency(samples),
		UsageSanity(samples),
		ResponseShape(samples, claimedModel),
		Tokenizer(samples, claimedModel),
		StreamShape(samples),
	}
	rep.Verdict = Verdict(rep.Findings)
	if rep.Successful == 0 {
		// An endpoint that answered nothing must not read as a clean or
		// skipped check; transport errors are still never evidence of fraud.
		rep.Verdict = StatusError
	}
	return rep
}

// tokenizerRun sends the base prompt, every tokenizer text, and the base
// prompt again, so a channel switch during the sequence shows up as two
// different base counts.
func tokenizerRun(ctx context.Context, s Sender) []Sample {
	prompts := make([]struct{ name, text string }, 0, TokenizerProbes)
	prompts = append(prompts, struct{ name, text string }{ProbeTokenizerBase, tokenizerBase})
	for _, t := range tokenizerTexts {
		prompts = append(prompts, struct{ name, text string }{probeTokenizerPrefix + t.name, tokenizerPrompt(t.text)})
	}
	prompts = append(prompts, struct{ name, text string }{ProbeTokenizerBase, tokenizerBase})
	var out []Sample
	for _, p := range prompts {
		if ctx.Err() != nil {
			break
		}
		if sample, ok := send(ctx, s, p.name, p.text, false); ok {
			out = append(out, sample)
		}
	}
	return out
}

// send sends one probe prompt. A request that failed because the run was
// cancelled says nothing about the endpoint, so it is dropped (ok false).
func send(ctx context.Context, s Sender, name, text string, stream bool) (Sample, bool) {
	resp, err := s.Send(ctx, probe.Request{User: text, MaxTokens: drawOutputLimit(), Stream: stream})
	if err != nil && ctx.Err() != nil {
		return Sample{}, false
	}
	return Sample{Probe: name, PromptChars: len(text), Response: resp, Err: err}, true
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
