package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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
				w.Header().Set("Content-Type", "application/json")
				var body struct {
					Messages []struct {
						Content string `json:"content"`
					} `json:"messages"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				// Two tokens per three characters: plausible for every check,
				// and kimi-k3 has no reference tokenizer to match.
				input := tt.input
				for _, m := range body.Messages {
					input += utf8.RuneCountInString(m.Content) * 2 / 3
				}
				_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"kimi-k3","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":` + itoa(input) + `,"completion_tokens":1}}`))
			}))
			defer srv.Close()
			t.Setenv("MODELTRUTH_API_KEY", "test-key")
			var out, errOut bytes.Buffer
			code := run([]string{"check", "--base-url", srv.URL, "--model", "kimi-k3"}, &out, &errOut)
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
	code := run([]string{"check", "--base-url", srv.URL, "--model", "gpt-6-sol", "--repeats", "1"}, &out, &errOut)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	for _, want := range []string{"verdict:  ERROR", "HTTP 503"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunRequiresKey(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", "")
	var out, errOut bytes.Buffer
	if code := run([]string{"check", "--base-url", "http://x", "--model", "m"}, &out, &errOut); code != 1 {
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
