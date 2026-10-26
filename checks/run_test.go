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
		if req.User == mediumPromptText {
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
		{"unreachable endpoint skips", func(int, probe.Request) (probe.Response, error) { return probe.Response{}, errors.New("dial") }, StatusSkip},
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
		})
	}
}
