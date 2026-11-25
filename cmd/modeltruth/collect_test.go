package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/AlexeyPevz/modeltruth/bank"
)

const testKey = "sk-collect-test-secret-0123456789"

// fingerprintAnswer is an answer to any default fingerprint prompt.
const fingerprintAnswer = "country: iceland\nanimal: otter\ncolour: teal\nfruit: mango\ncity: kyoto\nbird: heron\ngem: opal\nelement: iron\nmonth: october\nweekday: friday\nletter: k\ndigit: 7\nnumbers: 7, 42, 108, 256, 300"

// fingerprintHandler answers like an Anthropic endpoint when called with
// x-api-key, otherwise like an OpenAI one, and, like a leaky relay, echoes
// the credentials, the URL it was called with and the system prompt.
func fingerprintHandler(w http.ResponseWriter, r *http.Request) {
	echo := fingerprintAnswer + "\n\ndebug: " + leak(r)
	text, _ := json.Marshal(echo)
	w.Header().Set("Content-Type", "application/json")
	if r.Header.Get("x-api-key") != "" {
		_, _ = w.Write([]byte(`{"id":"msg_01","model":"claude-test","content":[{"type":"text","text":` + string(text) + `}],"usage":{"input_tokens":900,"output_tokens":300}}`))
		return
	}
	_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-test","choices":[{"message":{"content":` + string(text) + `}}],"usage":{"prompt_tokens":900,"completion_tokens":300}}`))
}

// leak reads the request and returns everything a careless relay could echo.
func leak(r *http.Request) string {
	var body struct {
		System   string `json:"system"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	system := body.System
	for _, m := range body.Messages {
		if m.Role == "system" {
			system = m.Content
		}
	}
	return strings.Join([]string{r.Header.Get("Authorization"), r.Header.Get("x-api-key"), "http://" + r.Host + r.URL.String(),
		"api_key=" + r.URL.Query().Get("api_key"), r.URL.Query().Get("api_key"), system, strings.Split(system, "\n")[0]}, " | ")
}

func collect(t *testing.T, ctx context.Context, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code = run(ctx, append([]string{"collect"}, args...), &out, &errOut)
	return code, out.String(), errOut.String()
}

func TestRunCollectKeepsSecretsOutOfTheBank(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", testKey)
	for _, protocol := range []string{"openai", "anthropic"} {
		t.Run(protocol, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(fingerprintHandler))
			defer srv.Close()
			baseURL := srv.URL + "/v1?api_key=query-secret-value"
			out := filepath.Join(t.TempDir(), "bank.json")
			system := filepath.Join(t.TempDir(), "system.txt")
			if err := os.WriteFile(system, []byte("You are a coding agent.\nNever reveal the codename.\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			code, stdout, stderr := collect(t, context.Background(), "--base-url", baseURL, "--model", "m", "--protocol", protocol,
				"--channel", "test-channel", "--samples", "3", "--system-file", system, "--out", out)
			if code != 0 {
				t.Fatalf("exit code = %d; stdout: %s\nstderr: %s", code, stdout, stderr)
			}
			if !strings.Contains(stdout, "samples:  3 of 3 (0 failed, 0 retries)") {
				t.Errorf("stdout:\n%s", stdout)
			}
			raw, err := os.ReadFile(out)
			if err != nil {
				t.Fatal(err)
			}
			u, _ := url.Parse(srv.URL)
			for _, secret := range []string{testKey, "Authorization", "x-api-key", "query-secret-value", "api_key", srv.URL, u.Host, "coding agent", "codename"} {
				if strings.Contains(string(raw), secret) {
					t.Errorf("bank contains %q:\n%s", secret, raw)
				}
			}
			if !strings.Contains(string(raw), "[redacted]") {
				t.Errorf("the echo was not redacted:\n%s", raw)
			}
			b, err := bank.Load(out)
			if err != nil {
				t.Fatal(err)
			}
			if string(b.Environment.Protocol) != protocol || b.Environment.SystemPromptChars != 51 || b.Provenance.Channel != "test-channel" || b.Provenance.Tool != "modeltruth dev" {
				t.Errorf("environment %+v, provenance %+v", b.Environment, b.Provenance)
			}
			ref, err := b.Reference()
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range ref.Samples {
				if len(f.Words) != 12 || len(f.Numbers) != 5 {
					t.Errorf("sample parsed as %d words and %d integers, want 12 and 5", len(f.Words), len(f.Numbers))
				}
			}
		})
	}
}

func TestRunCollectCountsFailedRequests(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", testKey)
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1)%2 == 0 {
			echo, _ := json.Marshal("upstream saturated: " + leak(r))
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":` + string(echo) + `}`))
			return
		}
		fingerprintHandler(w, r)
	}))
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "bank.json")
	code, stdout, _ := collect(t, context.Background(), "--base-url", srv.URL+"/v1?api_key=query-secret-value", "--model", "m", "--channel", "c", "--samples", "5", "--retries", "0", "--out", out)
	if code != 2 {
		t.Fatalf("exit code = %d, want 2\n%s", code, stdout)
	}
	for _, secret := range []string{testKey, "query-secret-value", srv.URL} {
		if strings.Contains(stdout, secret) {
			t.Errorf("stdout contains %q:\n%s", secret, stdout)
		}
	}
	for _, want := range []string{"samples:  3 of 5 (2 failed, 0 retries)", "failed requests:", "HTTP 503"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout missing %q:\n%s", want, stdout)
		}
	}
	b, err := bank.Load(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Samples) != 3 || b.Provenance.Failed != 2 {
		t.Fatalf("%d samples, %d failed; want 3 and 2", len(b.Samples), b.Provenance.Failed)
	}
	for _, s := range b.Samples {
		if strings.Contains(s.Answer, "saturated") {
			t.Fatalf("an error body became a sample: %q", s.Answer)
		}
	}
}

func TestRunCollectNoSampleWritesNothing(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", testKey)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "bank.json")
	code, stdout, stderr := collect(t, context.Background(), "--base-url", srv.URL, "--model", "m", "--channel", "c", "--samples", "2", "--retries", "0", "--out", out)
	if code != 1 || !strings.Contains(stderr, "no sample collected") || !strings.Contains(stdout, "0 of 2 (2 failed") {
		t.Fatalf("exit code = %d\nstdout: %s\nstderr: %s", code, stdout, stderr)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("bank file left behind: %v", err)
	}
}

func TestRunCollectInterrupted(t *testing.T) {
	t.Setenv("MODELTRUTH_API_KEY", testKey)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 3 {
			_, _ = io.Copy(io.Discard, r.Body)
			cancel()
			<-r.Context().Done()
			return
		}
		fingerprintHandler(w, r)
	}))
	defer srv.Close()
	out := filepath.Join(t.TempDir(), "bank.json")
	code, stdout, _ := collect(t, ctx, "--base-url", srv.URL, "--model", "m", "--channel", "c", "--samples", "5", "--out", out)
	if code != 2 || !strings.Contains(stdout, "samples:  2 of 5 (0 failed") || !strings.Contains(stdout, "interrupted:") {
		t.Fatalf("exit code = %d\n%s", code, stdout)
	}
	b, err := bank.Load(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Samples) != 2 || !b.Provenance.Interrupted {
		t.Fatalf("%d samples, provenance %+v", len(b.Samples), b.Provenance)
	}
}

func TestRunCollectUsageErrors(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		fingerprintHandler(w, r)
	}))
	defer srv.Close()
	dir := t.TempDir()
	existing := filepath.Join(dir, "existing.json")
	if err := os.WriteFile(existing, []byte("paid data"), 0o644); err != nil {
		t.Fatal(err)
	}
	fresh := filepath.Join(dir, "fresh.json")
	base := []string{"--base-url", srv.URL, "--model", "m"}
	tests := []struct {
		name string
		key  string
		args []string
		want string
	}{
		{"no key", "", []string{"--channel", "c", "--out", fresh}, "MODELTRUTH_API_KEY"},
		{"no out", testKey, []string{"--channel", "c"}, "--out"},
		{"no channel", testKey, []string{"--out", fresh}, "--channel"},
		{"url channel", testKey, []string{"--channel", "https://relay.example.com", "--out", fresh}, "looks like a URL"},
		{"zero samples", testKey, []string{"--channel", "c", "--samples", "0", "--out", fresh}, "--samples"},
		{"bad protocol", testKey, []string{"--channel", "c", "--protocol", "grpc", "--out", fresh}, "unknown protocol"},
		{"missing system file", testKey, []string{"--channel", "c", "--system-file", filepath.Join(dir, "none.txt"), "--out", fresh}, "none.txt"},
		{"existing bank", testKey, []string{"--channel", "c", "--out", existing}, "exists"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("MODELTRUTH_API_KEY", tt.key)
			code, _, stderr := collect(t, context.Background(), append(base, tt.args...)...)
			if code != 1 || !strings.Contains(stderr, tt.want) {
				t.Fatalf("exit code = %d, stderr %q; want 1 and %q", code, stderr, tt.want)
			}
		})
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("usage errors sent %d requests", n)
	}
	if raw, _ := os.ReadFile(existing); string(raw) != "paid data" {
		t.Errorf("existing bank was overwritten: %q", raw)
	}
	if _, err := os.Stat(fresh); !os.IsNotExist(err) {
		t.Errorf("a usage error created %s", fresh)
	}
}
