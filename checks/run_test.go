package checks

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/AlexeyPevz/modeltruth/probe"
)

type fakeSender struct {
	calls     []probe.Request
	responses func(i int, req probe.Request) (probe.Response, error)
}

func (f *fakeSender) Send(_ context.Context, req probe.Request) (probe.Response, error) {
	f.calls = append(f.calls, req)
	return f.responses(len(f.calls)-1, req)
}

// perRun is the number of requests one repeat sends: tiny, medium and the
// task prompts.
const perRun = 2 + TaskProbes

func cleanInput(req probe.Request) int64 {
	return int64(len(req.User)/4 + 8)
}

func TestRun(t *testing.T) {
	clean := func(_ int, req probe.Request) (probe.Response, error) {
		return probe.Response{HTTPStatus: 200, ID: "msg_01MzmDFUhapVJeHMhAUX2ZwC", Model: "claude-sonnet-5", Text: "ok", Usage: probe.Usage{InputTokens: cleanInput(req), OutputTokens: 1}}, nil
	}
	injected := func(i int, req probe.Request) (probe.Response, error) {
		r, _ := clean(i, req)
		r.Usage.InputTokens += 4380
		return r, nil
	}
	// The relay recognises the probe prompts and injects only into work.
	injectsIntoWork := func(i int, req probe.Request) (probe.Response, error) {
		r, _ := clean(i, req)
		if !strings.HasPrefix(req.User, "Reply with the single word") && !strings.HasPrefix(req.User, "A lighthouse keeper") {
			r.Usage.InputTokens += 4386
		}
		return r, nil
	}
	// The relay recognises probes by their small output limit (seen on A6 in
	// October 2026: max_tokens 16 always clean, 300 injected).
	injectsAboveProbeLimit := func(i int, req probe.Request) (probe.Response, error) {
		r, _ := clean(i, req)
		if req.MaxTokens > 64 {
			r.Usage.InputTokens += 4388
		}
		return r, nil
	}
	tests := []struct {
		name        string
		responses   func(int, probe.Request) (probe.Response, error)
		wantVerdict Status
	}{
		{"clean endpoint passes", clean, StatusPass},
		{"injected endpoint fails", injected, StatusFail},
		{"injection only into work prompts fails", injectsIntoWork, StatusFail},
		{"injection only above a probe-sized output limit fails", injectsAboveProbeLimit, StatusFail},
		{"unreachable endpoint is an error, not a pass or skip", func(int, probe.Request) (probe.Response, error) { return probe.Response{}, errors.New("dial") }, StatusError},
		{"some failed requests do not hide a clean result", func(i int, req probe.Request) (probe.Response, error) {
			if i == 0 {
				return probe.Response{HTTPStatus: 503}, errors.New("probe: HTTP 503: busy")
			}
			return clean(i, req)
		}, StatusPass},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &fakeSender{responses: tt.responses}
			rep := Run(context.Background(), s, "claude-sonnet-5", 3)
			if rep.Verdict != tt.wantVerdict {
				t.Errorf("verdict = %q, want %q: %+v", rep.Verdict, tt.wantVerdict, rep.Findings)
			}
			if len(s.calls) != 3*perRun || rep.Requests != 3*perRun {
				t.Errorf("requests = %d (report %d), want %d", len(s.calls), rep.Requests, 3*perRun)
			}
			if len(rep.Findings) != 4 {
				t.Errorf("findings = %d, want 4", len(rep.Findings))
			}
			if rep.Successful+len(rep.Errors) != rep.Requests {
				t.Errorf("successful %d + errors %d != requests %d", rep.Successful, len(rep.Errors), rep.Requests)
			}
		})
	}
}

func TestRunReportsErrors(t *testing.T) {
	s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) {
		return probe.Response{HTTPStatus: 401}, errors.New("probe: HTTP 401: invalid key")
	}}
	rep := Run(context.Background(), s, "m", 2)
	if len(rep.Errors) != 2*perRun {
		t.Fatalf("errors = %d, want %d", len(rep.Errors), 2*perRun)
	}
	if e := rep.Errors[0]; e.HTTPStatus != 401 || e.Probe != ProbeTiny || e.Error != "probe: HTTP 401: invalid key" {
		t.Errorf("first error = %+v", e)
	}
}

func TestRunPromptVariation(t *testing.T) {
	tasks := []string{taskPrompts[0], taskPrompts[1], taskPrompts[2]}
	t.Run("every repeat sends the same prompts in the same order", func(t *testing.T) {
		s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) { return probe.Response{HTTPStatus: 200}, nil }}
		runWith(context.Background(), s, "m", 3, "ready", tasks)
		want := append([]string{tinyPrompt("ready"), mediumPrompt("ready")}, tasks...)
		if len(s.calls) != 3*len(want) {
			t.Fatalf("calls = %d, want %d", len(s.calls), 3*len(want))
		}
		for i, c := range s.calls {
			if c.User != want[i%len(want)] {
				t.Fatalf("call %d user = %q, want %q", i, c.User, want[i%len(want)])
			}
		}
	})
	t.Run("the word and the task prompts change between runs", func(t *testing.T) {
		tinies, taskSets := map[string]bool{}, map[string]bool{}
		for i := 0; i < 40; i++ {
			s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) { return probe.Response{HTTPStatus: 200}, nil }}
			Run(context.Background(), s, "m", 1)
			tinies[s.calls[0].User] = true
			var set []string
			for _, c := range s.calls[2:] {
				set = append(set, c.User)
			}
			taskSets[strings.Join(set, "|")] = true
		}
		if len(tinies) < 2 || len(taskSets) < 2 {
			t.Errorf("40 runs used %d tiny prompts and %d task sets, want several of each", len(tinies), len(taskSets))
		}
	})
	t.Run("output limits look like real clients and vary between requests", func(t *testing.T) {
		s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) { return probe.Response{HTTPStatus: 200}, nil }}
		Run(context.Background(), s, "m", 4)
		limits := map[int]bool{}
		for i, c := range s.calls {
			if c.MaxTokens < 256 || c.MaxTokens > 4096 {
				t.Fatalf("call %d max tokens = %d, want 256..4096", i, c.MaxTokens)
			}
			limits[c.MaxTokens] = true
		}
		if len(limits) < 2 {
			t.Errorf("%d calls used %d distinct output limits, want several", len(s.calls), len(limits))
		}
	})
	t.Run("drawn task prompts are distinct", func(t *testing.T) {
		seen := map[string]bool{}
		for _, p := range drawTasks(TaskProbes) {
			if seen[p] {
				t.Fatalf("task prompt %q drawn twice", p)
			}
			seen[p] = true
		}
		if len(seen) != TaskProbes {
			t.Fatalf("drew %d prompts, want %d", len(seen), TaskProbes)
		}
	})
}
