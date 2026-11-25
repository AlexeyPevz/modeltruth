// Command modeltruth checks whether an LLM API endpoint serves the model it
// claims. The API key is read from an environment variable so it never lands
// in shell history or process listings.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"strings"
	"time"

	"github.com/AlexeyPevz/modeltruth/bank"
	"github.com/AlexeyPevz/modeltruth/checks"
	"github.com/AlexeyPevz/modeltruth/fingerprint"
	"github.com/AlexeyPevz/modeltruth/probe"
)

var version = "dev"

const usageText = `modeltruth checks whether an LLM API endpoint serves the model it claims.

Usage:
  modeltruth check --base-url URL --model NAME [--protocol openai|anthropic]
                   [--retries N] [--max-wait D] [--json]
  modeltruth collect --base-url URL --model NAME --channel NAME --out FILE
                     [--samples N] [--protocol openai|anthropic] [--system-file FILE]
                     [--max-tokens N] [--retries N] [--max-wait D]
  modeltruth version

The API key is read from $MODELTRUTH_API_KEY (override with --api-key-env).

check: Ctrl-C stops the run and prints the report for the requests answered so far.
Exit codes: 0 no failed checks, 1 usage error or no request succeeded, 2 at least one check failed.

collect: sends N fingerprint prompts from a trusted key and writes a reference
bank to FILE (never overwritten). --channel names where the key comes from,
e.g. anthropic-direct; it must not be a URL. Ctrl-C stops the run and writes
the samples collected so far.
Exit codes: 0 all N samples written, 1 usage error, no sample or the file could
not be written, 2 bank written with fewer samples than asked (failed requests or Ctrl-C).
`

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	// After the first Ctrl-C a second one kills the process as usual.
	go func() {
		<-ctx.Done()
		stop()
	}()
	code := run(ctx, os.Args[1:], os.Stdout, os.Stderr)
	stop()
	os.Exit(code)
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usageText)
		return 1
	}
	switch args[0] {
	case "version":
		fmt.Fprintln(stdout, version)
		return 0
	case "check":
		return runCheck(ctx, args[1:], stdout, stderr)
	case "collect":
		return runCollect(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usageText)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command %q\n\n%s", args[0], usageText)
		return 1
	}
}

func runCheck(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", "", "endpoint base URL, e.g. https://api.example.com/v1")
	model := fs.String("model", "", "model name to request and verify")
	protocol := fs.String("protocol", string(probe.ProtocolOpenAI), "wire protocol: openai or anthropic")
	keyEnv := fs.String("api-key-env", "MODELTRUTH_API_KEY", "environment variable holding the API key")
	repeats := fs.Int("repeats", 3, "how many times each probe prompt is sent")
	timeout := fs.Duration("timeout", 2*time.Minute, "per-request timeout")
	retries := fs.Int("retries", 2, "how many times a request is repeated after HTTP 429, 500, 502, 503, 504 or a timeout")
	maxWait := fs.Duration("max-wait", probe.DefaultMaxWait, "longest pause before a retry; a longer Retry-After ends the retries")
	asJSON := fs.Bool("json", false, "print the report as JSON")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	key := os.Getenv(*keyEnv)
	switch {
	case *baseURL == "" || *model == "":
		fmt.Fprintln(stderr, "--base-url and --model are required")
		return 1
	case key == "":
		fmt.Fprintf(stderr, "set the API key in $%s\n", *keyEnv)
		return 1
	case *protocol != string(probe.ProtocolOpenAI) && *protocol != string(probe.ProtocolAnthropic):
		fmt.Fprintf(stderr, "unknown protocol %q\n", *protocol)
		return 1
	case *retries < 0:
		fmt.Fprintln(stderr, "--retries must not be negative")
		return 1
	case *maxWait <= 0:
		fmt.Fprintln(stderr, "--max-wait must be positive")
		return 1
	}
	ep := probe.Endpoint{
		BaseURL:  *baseURL,
		APIKey:   key,
		Model:    *model,
		Protocol: probe.Protocol(*protocol),
		HTTP:     &http.Client{Timeout: *timeout},
		Retries:  *retries,
		MaxWait:  *maxWait,
	}
	rep := checks.Run(ctx, ep, ep.Protocol, *model, *repeats)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		printReport(stdout, rep)
	}
	switch rep.Verdict {
	case checks.StatusFail:
		return 2
	case checks.StatusError:
		return 1
	}
	return 0
}

func printReport(w io.Writer, rep checks.Report) {
	retried := ""
	if rep.Retries > 0 {
		retried = fmt.Sprintf(", %d retries", rep.Retries)
	}
	fmt.Fprintf(w, "model:    %s\nrequests: %d (%d ok%s)\nverdict:  %s\n", rep.Model, rep.Requests, rep.Successful, retried, strings.ToUpper(string(rep.Verdict)))
	if rep.Interrupted {
		fmt.Fprintln(w, "interrupted: the report covers only the requests answered before the run was stopped")
	}
	fmt.Fprintln(w)
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "[%-4s] %-20s %s\n", strings.ToUpper(string(f.Status)), f.Check, f.Summary)
	}
	if len(rep.Errors) > 0 {
		fmt.Fprintf(w, "\nfailed requests:\n")
		for _, e := range rep.Errors {
			if e.Retries > 0 {
				fmt.Fprintf(w, "  %-6s %s (after %d retries)\n", e.Probe, e.Error, e.Retries)
				continue
			}
			fmt.Fprintf(w, "  %-6s %s\n", e.Probe, e.Error)
		}
	}
}

func runCollect(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("collect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", "", "endpoint base URL of a trusted channel; not written to the bank")
	model := fs.String("model", "", "model name to request and record")
	channel := fs.String("channel", "", "name of the channel recorded in the bank, e.g. anthropic-direct (not a URL)")
	out := fs.String("out", "", "bank file to create; an existing file is never overwritten")
	samples := fs.Int("samples", 30, "how many fingerprint prompts to send")
	protocol := fs.String("protocol", string(probe.ProtocolOpenAI), "wire protocol: openai or anthropic")
	systemFile := fs.String("system-file", "", "file whose text is sent as the system prompt (the bank records only its length)")
	maxTokens := fs.Int("max-tokens", bank.DefaultMaxTokens, "output limit of every request")
	keyEnv := fs.String("api-key-env", "MODELTRUTH_API_KEY", "environment variable holding the API key")
	timeout := fs.Duration("timeout", 2*time.Minute, "per-request timeout")
	retries := fs.Int("retries", 2, "how many times a request is repeated after HTTP 429, 500, 502, 503, 504 or a timeout")
	maxWait := fs.Duration("max-wait", probe.DefaultMaxWait, "longest pause before a retry; a longer Retry-After ends the retries")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	key := os.Getenv(*keyEnv)
	switch {
	case *baseURL == "" || *model == "" || *out == "" || *channel == "":
		fmt.Fprintln(stderr, "--base-url, --model, --channel and --out are required")
		return 1
	case key == "":
		fmt.Fprintf(stderr, "set the API key in $%s\n", *keyEnv)
		return 1
	case *protocol != string(probe.ProtocolOpenAI) && *protocol != string(probe.ProtocolAnthropic):
		fmt.Fprintf(stderr, "unknown protocol %q\n", *protocol)
		return 1
	case *samples < 1:
		fmt.Fprintln(stderr, "--samples must be at least 1")
		return 1
	case *maxTokens < 1:
		fmt.Fprintln(stderr, "--max-tokens must be positive")
		return 1
	case *retries < 0:
		fmt.Fprintln(stderr, "--retries must not be negative")
		return 1
	case *maxWait <= 0:
		fmt.Fprintln(stderr, "--max-wait must be positive")
		return 1
	}
	if err := bank.ValidChannel(*channel); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	var system string
	if *systemFile != "" {
		raw, err := os.ReadFile(*systemFile)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		system = string(raw)
	}
	// The file is created before the first paid request, so a bad path or
	// an existing bank stops the run before it costs anything.
	f, err := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	ep := probe.Endpoint{
		BaseURL:  *baseURL,
		APIKey:   key,
		Model:    *model,
		Protocol: probe.Protocol(*protocol),
		HTTP:     &http.Client{Timeout: *timeout},
		Retries:  *retries,
		MaxWait:  *maxWait,
	}
	secrets := secretReplacer(key, *baseURL, system)
	var failures []string
	b, err := bank.Collect(ctx, ep, bank.CollectOptions{
		Model:     *model,
		Protocol:  ep.Protocol,
		Channel:   *channel,
		Tool:      "modeltruth " + version,
		Samples:   *samples,
		System:    system,
		MaxTokens: *maxTokens,
		Spec:      fingerprint.DefaultSpec(),
		OnError:   func(err error) { failures = append(failures, secrets.Replace(err.Error())) },
	})
	if err == nil && len(b.Samples) == 0 {
		err = errors.New("no sample collected; the bank was not written")
	}
	if err == nil {
		redact(b, secrets)
		err = b.Write(f)
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	var prov bank.Provenance
	if b != nil {
		prov = b.Provenance
		fmt.Fprintf(stdout, "model:    %s\nsamples:  %d of %d (%d failed, %d retries)\n", b.Model, prov.SampleCount, prov.Requested, prov.Failed, prov.Retries)
		if prov.Interrupted {
			fmt.Fprintln(stdout, "interrupted: the bank holds only the samples collected before the run was stopped")
		}
	}
	if len(failures) > 0 {
		fmt.Fprintln(stdout, "\nfailed requests:")
		for _, e := range failures {
			fmt.Fprintf(stdout, "  %s\n", e)
		}
	}
	if err != nil {
		_ = os.Remove(*out)
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "bank:     %s\n", *out)
	if prov.SampleCount < prov.Requested {
		return 2
	}
	return 0
}

// secretReplacer hides what identifies the access or the system prompt: a
// relay that echoes request headers, its own address or the instructions must
// not get them into a bank meant to be published or into the terminal log.
// Query values and prompt lines shorter than 8 characters are left alone, so
// that a parameter like v=1 does not damage answers.
func secretReplacer(key, baseURL, system string) *strings.Replacer {
	secrets := []string{key, baseURL, strings.TrimSpace(system)}
	if u, err := url.Parse(baseURL); err == nil {
		secrets = append(secrets, u.RawQuery, u.Host)
		for _, values := range u.Query() {
			for _, v := range values {
				if len(v) >= 8 {
					secrets = append(secrets, v)
				}
			}
		}
	}
	for _, line := range strings.Split(system, "\n") {
		if line = strings.TrimSpace(line); len(line) >= 8 {
			secrets = append(secrets, line)
		}
	}
	// Longer secrets first, so that a match is never cut short by a secret
	// that is a part of it.
	sort.Slice(secrets, func(i, j int) bool { return len(secrets[i]) > len(secrets[j]) })
	var pairs []string
	for _, secret := range secrets {
		if secret != "" {
			pairs = append(pairs, secret, "[redacted]")
		}
	}
	return strings.NewReplacer(pairs...)
}

func redact(b *bank.Bank, r *strings.Replacer) {
	for i := range b.Samples {
		s := &b.Samples[i]
		s.Answer = r.Replace(s.Answer)
		s.ResponseModel = r.Replace(s.ResponseModel)
	}
}
