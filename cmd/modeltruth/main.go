// Command modeltruth checks whether an LLM API endpoint serves the model it
// claims. The API key is read from an environment variable so it never lands
// in shell history or process listings.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/AlexeyPevz/modeltruth/checks"
	"github.com/AlexeyPevz/modeltruth/probe"
)

var version = "dev"

const usageText = `modeltruth checks whether an LLM API endpoint serves the model it claims.

Usage:
  modeltruth check --base-url URL --model NAME [--protocol openai|anthropic] [--json]
  modeltruth version

The API key is read from $MODELTRUTH_API_KEY (override with --api-key-env).
Exit codes: 0 no failed checks, 1 usage error or no request succeeded, 2 at least one check failed.
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usageText)
		return 1
	}
	switch args[0] {
	case "version":
		fmt.Fprintln(stdout, version)
		return 0
	case "check":
		return runCheck(args[1:], stdout, stderr)
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usageText)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command %q\n\n%s", args[0], usageText)
		return 1
	}
}

func runCheck(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", "", "endpoint base URL, e.g. https://api.example.com/v1")
	model := fs.String("model", "", "model name to request and verify")
	protocol := fs.String("protocol", string(probe.ProtocolOpenAI), "wire protocol: openai or anthropic")
	keyEnv := fs.String("api-key-env", "MODELTRUTH_API_KEY", "environment variable holding the API key")
	repeats := fs.Int("repeats", 3, "how many times each probe prompt is sent")
	timeout := fs.Duration("timeout", 2*time.Minute, "per-request timeout")
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
	}
	ep := probe.Endpoint{
		BaseURL:  *baseURL,
		APIKey:   key,
		Model:    *model,
		Protocol: probe.Protocol(*protocol),
		HTTP:     &http.Client{Timeout: *timeout},
	}
	rep := checks.Run(context.Background(), ep, *model, *repeats)
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
	fmt.Fprintf(w, "model:    %s\nrequests: %d (%d ok)\nverdict:  %s\n\n", rep.Model, rep.Requests, rep.Successful, strings.ToUpper(string(rep.Verdict)))
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "[%-4s] %-20s %s\n", strings.ToUpper(string(f.Status)), f.Check, f.Summary)
	}
	if len(rep.Errors) > 0 {
		fmt.Fprintf(w, "\nfailed requests:\n")
		for _, e := range rep.Errors {
			fmt.Fprintf(w, "  %-6s %s\n", e.Probe, e.Error)
		}
	}
}
