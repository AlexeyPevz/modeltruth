package checks

import (
	"context"
	"errors"
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

func TestRun(t *testing.T) {
	clean := func(_ int, req probe.Request) (probe.Response, error) {
		input := int64(12)
		if len(req.User) > 200 {
			input = 140
		}
		return probe.Response{HTTPStatus: 200, ID: "msg_01MzmDFUhapVJeHMhAUX2ZwC", Model: "claude-sonnet-5", Text: "ok", Usage: probe.Usage{InputTokens: input, OutputTokens: 1}}, nil
	}
	injected := func(i int, req probe.Request) (probe.Response, error) {
		r, _ := clean(i, req)
		r.Usage.InputTokens += 4380
		return r, nil
	}
	tests := []struct {
		name        string
		responses   func(int, probe.Request) (probe.Response, error)
		wantVerdict Status
	}{
		{"clean endpoint passes", clean, StatusPass},
		{"injected endpoint fails", injected, StatusFail},
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
			if len(s.calls) != 6 || rep.Requests != 6 {
				t.Errorf("requests = %d (report %d), want 6", len(s.calls), rep.Requests)
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
	if len(rep.Errors) != 4 {
		t.Fatalf("errors = %d, want 4", len(rep.Errors))
	}
	if e := rep.Errors[0]; e.HTTPStatus != 401 || e.Probe != ProbeTiny || e.Error != "probe: HTTP 401: invalid key" {
		t.Errorf("first error = %+v", e)
	}
}

func TestRunPromptVariation(t *testing.T) {
	t.Run("one word for every request of a run", func(t *testing.T) {
		s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) { return probe.Response{HTTPStatus: 200}, nil }}
		runWithWord(context.Background(), s, "m", 3, "ready")
		for i, c := range s.calls {
			want := tinyPrompt("ready")
			if i%2 == 1 {
				want = mediumPrompt("ready")
			}
			if c.User != want {
				t.Fatalf("call %d user = %q, want %q", i, c.User, want)
			}
		}
	})
	t.Run("the word changes between runs", func(t *testing.T) {
		seen := map[string]bool{}
		for i := 0; i < 40; i++ {
			s := &fakeSender{responses: func(int, probe.Request) (probe.Response, error) { return probe.Response{HTTPStatus: 200}, nil }}
			Run(context.Background(), s, "m", 1)
			seen[s.calls[0].User] = true
		}
		if len(seen) < 2 {
			t.Errorf("40 runs used %d distinct tiny prompts, want several", len(seen))
		}
	})
}
