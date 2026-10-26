package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
				input := tt.input
				if r.ContentLength > 300 {
					input += 130
				}
				_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-6-sol","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":` + itoa(input) + `,"completion_tokens":1}}`))
			}))
			defer srv.Close()
			t.Setenv("MODELTRUTH_API_KEY", "test-key")
			var out, errOut bytes.Buffer
			code := run([]string{"check", "--base-url", srv.URL, "--model", "gpt-6-sol"}, &out, &errOut)
			if code != tt.wantCode {
				t.Errorf("exit code = %d, want %d; stderr: %s", code, tt.wantCode, errOut.String())
			}
			if !strings.Contains(out.String(), tt.wantText) {
				t.Errorf("output missing %q:\n%s", tt.wantText, out.String())
			}
		})
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
