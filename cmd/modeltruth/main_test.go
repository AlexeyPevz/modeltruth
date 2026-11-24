package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"
)

func TestRunCheckExitCodes(t *testing.T) {
	tests := []struct {
		name     string
		input    int
		wantCode int
		wantText string
	}{
		{"clean endpoint exits 0", 12, 0, "verdict:  PASS"},
		{"hidden prompt exits 2", 4500, 2, "verdict:  FAIL"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				answer(w, r, tt.input)
			}))
			defer srv.Close()
			t.Setenv("MODELTRUTH_API_KEY", "test-key")
			var out, errOut bytes.Buffer
			code := run(context.Background(), []string{"check", "--base-url", srv.URL, "--model", "kimi-k3"}, &out, &errOut)
			if code != tt.wantCode {
				t.Errorf("exit code = %d, want %d; stderr: %s", code, tt.wantCode, errOut.String())
			}
			if !strings.Contains(out.String(), tt.wantText) {
				t.Errorf("output missing %q:\n%s", tt.wantText, out.String())
			}
		})
	}
}

func TestRunCheckFailingEndpointIsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"upstream saturated"}`))
	}))
	defer srv.Close()
	t.Setenv("MODELTRUTH_API_KEY", "test-key")
	var out, errOut bytes.Buffer
	code := run(context.Background(), []string{"check", "--base-url", srv.URL, "--model", "gpt-6-sol", "--repeats", "1", "--retries", "0"}, &out, &errOut)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	for _, want := range []string{"verdict:  ERROR", "HTTP 503"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}

// answer replies like an OpenAI endpoint that bills base plus two tokens per
// three characters of the messages: plausible for every check, and kimi-k3
// has no reference tokenizer to match. Stream requests get an SSE stream
// with usage in the final chunk.
func answer(w http.ResponseWriter, r *http.Request, base int) {
	var body struct {
		Stream   bool `json:"stream"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	input := base
	for _, m := range body.Messages {
		input += utf8.RuneCountInString(m.Content) * 2 / 3
	}
	usage := `"usage":{"prompt_tokens":` + itoa(input) + `,"completion_tokens":1}`
	if body.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"chatcmpl-1","model":"kimi-k3","choices":[{"delta":{"content":"ok"}}],"usage":null}` + "\n\n" +
			`data: {"id":"chatcmpl-1","model":"kimi-k3","choices":[],` + usage + `}` + "\n\n" +
			"data: [DONE]\n\n"))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"kimi-k3","choices":[{"message":{"content":"ok"}}],` + usage + `}`))
}

// usageHandler answers like the clean endpoint in TestRunCheckExitCodes.
func usageHandler(w http.ResponseWriter, r *http.Request) {
	answer(w, r, 12)
}

func TestRunCheckRetries(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", "test-key")
	t.Run("a retried request yields a normal sample", func(t *testing.T) {
		var calls atomic.Int64
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if calls.Add(1)%2 == 1 {
				w.Header().Set("Retry-After", "0")
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			usageHandler(w, r)
		}))
		defer srv.Close()
		var out, errOut bytes.Buffer
		code := run(context.Background(), []string{"check", "--base-url", srv.URL, "--model", "kimi-k3", "--repeats", "1"}, &out, &errOut)
		if code != 0 {
			t.Errorf("exit code = %d, want 0; stderr: %s\n%s", code, errOut.String(), out.String())
		}
		for _, want := range []string{"verdict:  PASS", "(21 ok, 21 retries)"} {
			if !strings.Contains(out.String(), want) {
				t.Errorf("output missing %q:\n%s", want, out.String())
			}
		}
		if strings.Contains(out.String(), "failed requests") {
			t.Errorf("successful retries must not be listed as failed requests:\n%s", out.String())
		}
	})
	t.Run("exhausted retries report the last status", func(t *testing.T) {
		var calls atomic.Int64
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "0")
			if calls.Add(1)%2 == 1 {
				w.WriteHeader(http.StatusBadGateway)
				return
			}
			w.WriteHeader(http.StatusTooManyRequests)
		}))
		defer srv.Close()
		var out, errOut bytes.Buffer
		code := run(context.Background(), []string{"check", "--base-url", srv.URL, "--model", "m", "--repeats", "1", "--retries", "1", "--json"}, &out, &errOut)
		if code != 1 {
			t.Errorf("exit code = %d, want 1", code)
		}
		var rep struct {
			Retries int `json:"retries"`
			Errors  []struct {
				HTTPStatus int `json:"http_status"`
				Retries    int `json:"retries"`
			} `json:"errors"`
		}
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("decode report: %v\n%s", err, out.String())
		}
		if len(rep.Errors) != 21 || rep.Retries != 21 {
			t.Fatalf("errors = %d, retries = %d, want 21 and 21", len(rep.Errors), rep.Retries)
		}
		for i, e := range rep.Errors {
			if e.HTTPStatus != http.StatusTooManyRequests || e.Retries != 1 {
				t.Fatalf("error %d = %+v, want last status 429 after 1 retry", i, e)
			}
		}
	})
	t.Run("negative retries is a usage error", func(t *testing.T) {
		var out, errOut bytes.Buffer
		if code := run(context.Background(), []string{"check", "--base-url", "http://x", "--model", "m", "--retries", "-1"}, &out, &errOut); code != 1 {
			t.Errorf("exit code = %d, want 1", code)
		}
	})
}

func TestRunCheckInterrupted(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", "test-key")
	tests := []struct {
		name     string
		cancelAt int64
		wantCode int
		wantText []string
	}{
		{"partial report after some answers", 4, 0, []string{"requests: 3 (3 ok)", "interrupted:"}},
		{"no answer before the interrupt exits 1", 1, 1, []string{"requests: 0 (0 ok)", "verdict:  ERROR", "interrupted:"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var calls atomic.Int64
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if calls.Add(1) == tt.cancelAt {
					// Ctrl-C while this request is in flight. The body is
					// drained so the server notices the client going away.
					_, _ = io.Copy(io.Discard, r.Body)
					cancel()
					<-r.Context().Done()
					return
				}
				usageHandler(w, r)
			}))
			defer srv.Close()
			var out, errOut bytes.Buffer
			code := run(ctx, []string{"check", "--base-url", srv.URL, "--model", "kimi-k3"}, &out, &errOut)
			if code != tt.wantCode {
				t.Errorf("exit code = %d, want %d\n%s", code, tt.wantCode, out.String())
			}
			for _, want := range tt.wantText {
				if !strings.Contains(out.String(), want) {
					t.Errorf("output missing %q:\n%s", want, out.String())
				}
			}
			if got := calls.Load(); got != tt.cancelAt {
				t.Errorf("server saw %d requests, want %d", got, tt.cancelAt)
			}
		})
	}
}

func TestRunRequiresKey(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", "")
	var out, errOut bytes.Buffer
	if code := run(context.Background(), []string{"check", "--base-url", "http://x", "--model", "m"}, &out, &errOut); code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(errOut.String(), "MODELTRUTH_API_KEY") {
		t.Errorf("stderr should name the key variable: %s", errOut.String())
	}
}

func itoa(n int) string {
	var b [20]byte
	i := len(b)
	for {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
		if n == 0 {
			return string(b[i:])
		}
	}
}
