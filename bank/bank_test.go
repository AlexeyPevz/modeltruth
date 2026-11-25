package bank

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/AlexeyPevz/modeltruth/fingerprint"
	"github.com/AlexeyPevz/modeltruth/probe"
)

var update = flag.Bool("update", false, "rewrite testdata/synthetic-bank.json")

// senderFunc adapts a function to Sender.
type senderFunc func(ctx context.Context, req probe.Request) (probe.Response, error)

func (f senderFunc) Send(ctx context.Context, req probe.Request) (probe.Response, error) {
	return f(ctx, req)
}

// syntheticWords are the answers of a made-up model; nothing in them comes
// from a real one.
var syntheticWords = map[fingerprint.Category][]string{
	fingerprint.Country: {"iceland", "norway", "japan"},
	fingerprint.Animal:  {"otter", "fox", "owl"},
	fingerprint.Colour:  {"teal", "blue", "green"},
	fingerprint.Fruit:   {"mango", "apple", "kiwi"},
	fingerprint.City:    {"kyoto", "lisbon", "prague"},
	fingerprint.Bird:    {"heron", "robin", "owl"},
	fingerprint.Gem:     {"opal", "emerald", "ruby"},
	fingerprint.Element: {"iron", "neon", "carbon"},
	fingerprint.Month:   {"october", "march", "june"},
	fingerprint.Weekday: {"friday", "tuesday", "sunday"},
	fingerprint.Letter:  {"k", "m", "r"},
	fingerprint.Digit:   {"7", "3", "4"},
}

// syntheticAnswer answers p like a model that prefers the first word of each
// list and integers from 100 to 160.
func syntheticAnswer(p fingerprint.Prompt, r *rand.Rand) string {
	var lines []string
	for _, c := range p.Categories {
		words := syntheticWords[c]
		w := words[0]
		if r.IntN(3) == 0 {
			w = words[r.IntN(len(words))]
		}
		lines = append(lines, fmt.Sprintf("%s: %s", c, w))
	}
	var nums []string
	for range p.Numbers.Count {
		n := p.Numbers.Min + r.IntN(p.Numbers.Max-p.Numbers.Min+1)
		if r.IntN(2) == 0 {
			n = 100 + r.IntN(61)
		}
		nums = append(nums, fmt.Sprint(n))
	}
	if len(nums) > 0 {
		lines = append(lines, "numbers: "+strings.Join(nums, ", "))
	}
	return strings.Join(lines, "\n")
}

// syntheticSender answers every prompt with syntheticAnswer; it regenerates
// the prompt from the seed the test hands out.
func syntheticSender(spec fingerprint.Spec, seeds *[]uint64, r *rand.Rand) Sender {
	return senderFunc(func(_ context.Context, req probe.Request) (probe.Response, error) {
		p, err := fingerprint.Generate(spec, (*seeds)[len(*seeds)-1])
		if err != nil || p.Text != req.User {
			return probe.Response{}, fmt.Errorf("unexpected prompt: %v", err)
		}
		return probe.Response{HTTPStatus: 200, Model: "synthetic-model", Text: syntheticAnswer(p, r), Usage: probe.Usage{InputTokens: 900, OutputTokens: 700}}, nil
	})
}

func seedSequence(seeds *[]uint64, next uint64) func() uint64 {
	return func() uint64 {
		*seeds = append(*seeds, next)
		next++
		return next - 1
	}
}

func collectSynthetic(t *testing.T, ctx context.Context, samples int, s Sender, seeds *[]uint64) *Bank {
	t.Helper()
	b, err := Collect(ctx, s, CollectOptions{
		Model:    "synthetic-model",
		Protocol: probe.ProtocolOpenAI,
		Channel:  "synthetic",
		Tool:     "modeltruth test",
		Samples:  samples,
		Spec:     fingerprint.DefaultSpec(),
		Seed:     seedSequence(seeds, 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func defaultEnv() Environment {
	return Environment{Protocol: probe.ProtocolOpenAI, MaxTokens: DefaultMaxTokens, Template: fingerprint.TemplateVersion, Spec: fingerprint.DefaultSpec()}
}

// TestSyntheticBank keeps testdata/synthetic-bank.json the output of a
// collection against syntheticSender; go test ./bank -update rewrites it.
func TestSyntheticBank(t *testing.T) {
	var seeds []uint64
	b := collectSynthetic(t, context.Background(), 6, syntheticSender(fingerprint.DefaultSpec(), &seeds, rand.New(rand.NewPCG(1, 2))), &seeds)
	day := time.Date(2026, 11, 1, 12, 0, 0, 0, time.UTC)
	b.Provenance.Started = day
	for i := range b.Samples {
		b.Samples[i].Time = day.Add(time.Duration(i+1) * time.Minute)
	}
	b.Provenance.Finished = day.Add(time.Duration(len(b.Samples)+1) * time.Minute)
	var buf bytes.Buffer
	if err := b.Write(&buf); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join("testdata", "synthetic-bank.json")
	if *update {
		if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	file, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(file, buf.Bytes()) {
		t.Fatalf("%s is stale; run go test ./bank -update", path)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := loaded.CheckEnvironment(defaultEnv()); err != nil {
		t.Fatal(err)
	}
	ref, err := loaded.Reference()
	if err != nil {
		t.Fatal(err)
	}
	if ref.Model != "synthetic-model" || len(ref.Samples) != 6 {
		t.Fatalf("reference = %s with %d samples, want synthetic-model with 6", ref.Model, len(ref.Samples))
	}
	for i, f := range ref.Samples {
		if len(f.Words) != 12 || len(f.Numbers) != 200 || f.Rejected != 0 || f.Seed != uint64(i+1) {
			t.Fatalf("sample %d parsed as %d words, %d integers, %d rejected, seed %d", i, len(f.Words), len(f.Numbers), f.Rejected, f.Seed)
		}
	}
}

func TestCollectCountsErrorsSeparately(t *testing.T) {
	var seeds []uint64
	good := syntheticSender(fingerprint.DefaultSpec(), &seeds, rand.New(rand.NewPCG(3, 4)))
	calls := 0
	s := senderFunc(func(ctx context.Context, req probe.Request) (probe.Response, error) {
		calls++
		switch calls {
		case 2:
			return probe.Response{HTTPStatus: 503, Retries: 2}, errors.New("probe: HTTP 503: upstream saturated")
		case 4:
			return probe.Response{}, errors.New("dial tcp: connection refused")
		}
		return good.Send(ctx, req)
	})
	b := collectSynthetic(t, context.Background(), 5, s, &seeds)
	p := b.Provenance
	if len(b.Samples) != 3 || p.SampleCount != 3 || p.Requested != 5 || p.Failed != 2 || p.Retries != 2 || p.Interrupted {
		t.Fatalf("provenance = %+v with %d samples, want 3 of 5, 2 failed, 2 retries", p, len(b.Samples))
	}
	for i, want := range []uint64{1, 3, 5} {
		if s := b.Samples[i]; s.Seed != want || s.Answer == "" {
			t.Errorf("sample %d = seed %d, answer %q; want seed %d with an answer", i, s.Seed, s.Answer, want)
		}
	}
	if err := b.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCollectCountsEmptyAnswersAsFailed(t *testing.T) {
	var seeds []uint64
	s := senderFunc(func(ctx context.Context, req probe.Request) (probe.Response, error) {
		return probe.Response{HTTPStatus: 200, Text: " \n"}, nil
	})
	b := collectSynthetic(t, context.Background(), 2, s, &seeds)
	if len(b.Samples) != 0 || b.Provenance.Failed != 2 {
		t.Fatalf("%d samples, %d failed; want 0 and 2", len(b.Samples), b.Provenance.Failed)
	}
}

func TestCollectInterrupted(t *testing.T) {
	var seeds []uint64
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	good := syntheticSender(fingerprint.DefaultSpec(), &seeds, rand.New(rand.NewPCG(5, 6)))
	calls := 0
	s := senderFunc(func(ctx context.Context, req probe.Request) (probe.Response, error) {
		if calls++; calls == 3 {
			cancel()
			return probe.Response{}, ctx.Err()
		}
		return good.Send(ctx, req)
	})
	b := collectSynthetic(t, ctx, 5, s, &seeds)
	if len(b.Samples) != 2 || b.Provenance.Failed != 0 || !b.Provenance.Interrupted || calls != 3 {
		t.Fatalf("%d samples, provenance %+v, %d calls; want 2 samples, interrupted, no failure, 3 calls", len(b.Samples), b.Provenance, calls)
	}
}

func TestCollectRecordsEnvironment(t *testing.T) {
	var got probe.Request
	s := senderFunc(func(_ context.Context, req probe.Request) (probe.Response, error) {
		got = req
		return probe.Response{Text: "country: iceland"}, nil
	})
	spec := fingerprint.Spec{Categories: []fingerprint.Category{fingerprint.Country}}
	b, err := Collect(context.Background(), s, CollectOptions{
		Model: "m", Protocol: probe.ProtocolAnthropic, Channel: "anthropic-direct", Samples: 1,
		System: "Ты — ассистент.", MaxTokens: 300, Spec: spec, Seed: func() uint64 { return 42 },
	})
	if err != nil {
		t.Fatal(err)
	}
	want := Environment{Protocol: probe.ProtocolAnthropic, SystemPromptChars: 15, MaxTokens: 300, Template: fingerprint.TemplateVersion, Spec: spec}
	if err := b.CheckEnvironment(want); err != nil {
		t.Fatal(err)
	}
	if got.System != "Ты — ассистент." || got.MaxTokens != 300 || got.Stream || got.Signed {
		t.Fatalf("request = %+v", got)
	}
	p, _ := fingerprint.Generate(spec, 42)
	if got.User != p.Text || b.Samples[0].Seed != 42 {
		t.Fatalf("prompt or seed not recorded: %q, seed %d", got.User, b.Samples[0].Seed)
	}
}

func TestCollectRejectsBadOptions(t *testing.T) {
	s := senderFunc(func(context.Context, probe.Request) (probe.Response, error) {
		t.Fatal("no request may be sent")
		return probe.Response{}, nil
	})
	ok := CollectOptions{Model: "m", Protocol: probe.ProtocolOpenAI, Channel: "c", Samples: 1, Spec: fingerprint.DefaultSpec()}
	for name, mod := range map[string]func(*CollectOptions){
		"no model":      func(o *CollectOptions) { o.Model = "" },
		"no samples":    func(o *CollectOptions) { o.Samples = 0 },
		"protocol":      func(o *CollectOptions) { o.Protocol = "grpc" },
		"url channel":   func(o *CollectOptions) { o.Channel = "https://relay.example.com/v1" },
		"empty spec":    func(o *CollectOptions) { o.Spec = fingerprint.Spec{} },
		"blank channel": func(o *CollectOptions) { o.Channel = " " },
	} {
		o := ok
		mod(&o)
		if _, err := Collect(context.Background(), s, o); err == nil {
			t.Errorf("%s: Collect accepted %+v", name, o)
		}
	}
}

func encode(t *testing.T, b *Bank) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := b.Write(&buf); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestReadRoundTrip(t *testing.T) {
	var seeds []uint64
	b := collectSynthetic(t, context.Background(), 2, syntheticSender(fingerprint.DefaultSpec(), &seeds, rand.New(rand.NewPCG(7, 8))), &seeds)
	got, err := Read(bytes.NewReader(encode(t, b)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encode(t, got), encode(t, b)) {
		t.Fatal("bank changed in a write and read")
	}
}

func TestReadRejects(t *testing.T) {
	var seeds []uint64
	b := collectSynthetic(t, context.Background(), 2, syntheticSender(fingerprint.DefaultSpec(), &seeds, rand.New(rand.NewPCG(9, 10))), &seeds)
	raw := string(encode(t, b))
	tests := []struct {
		name, old, new, want string
	}{
		{"other format version", `"version": 1`, `"version": 2`, "format version 2, this modeltruth reads version 1"},
		{"version missing", `"version": 1,`, ``, "format version 0"},
		{"other format", `"format": "modeltruth-bank"`, `"format": "other"`, `format "other"`},
		{"unknown field", `"model":`, `"extra": 1, "model":`, "unknown field"},
		{"sample count", `"sample_count": 2`, `"sample_count": 3`, "sample_count 3"},
		{"url channel", `"channel": "synthetic"`, `"channel": "https://relay.example.com"`, "looks like a URL"},
		{"unknown protocol", `"protocol": "openai"`, `"protocol": "grpc"`, "unknown protocol"},
		{"no model", `"model": "synthetic-model"`, `"model": ""`, "model is empty"},
		{"bad spec", `"count": 200`, `"count": 5000`, "spec"},
		{"not json", raw, "{", "decode"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !strings.Contains(raw, tt.old) {
				t.Fatalf("fixture lacks %q", tt.old)
			}
			_, err := Read(strings.NewReader(strings.Replace(raw, tt.old, tt.new, 1)))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

func TestCheckEnvironment(t *testing.T) {
	b := &Bank{Model: "m", Environment: defaultEnv()}
	same := defaultEnv()
	same.Spec.Categories = append([]fingerprint.Category(nil), same.Spec.Categories...)
	same.Spec.Categories[0], same.Spec.Categories[1] = same.Spec.Categories[1], same.Spec.Categories[0]
	if err := b.CheckEnvironment(same); err != nil {
		t.Fatalf("category order must not matter: %v", err)
	}

	probeEnv := defaultEnv()
	probeEnv.Protocol = probe.ProtocolAnthropic
	probeEnv.SystemPromptChars = 7300
	probeEnv.MaxTokens = 300
	probeEnv.Template = "fp0"
	probeEnv.Spec.Numbers.Count = 80
	err := b.CheckEnvironment(probeEnv)
	var mismatch *MismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("error = %v, want *MismatchError", err)
	}
	if len(mismatch.Differences) != 5 {
		t.Fatalf("differences = %q, want 5", mismatch.Differences)
	}
	for _, want := range []string{"protocol: bank openai, probe anthropic", "system prompt chars: bank 0, probe 7300", "max_tokens: bank 2048, probe 300", "template: bank fp1, probe fp0", "200 integers", "80 integers"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestReferenceNeedsCurrentTemplate(t *testing.T) {
	b := &Bank{Model: "m", Environment: defaultEnv()}
	b.Environment.Template = "fp0"
	if _, err := b.Reference(); err == nil || !strings.Contains(err.Error(), "template fp0") {
		t.Fatalf("error = %v, want a template mismatch", err)
	}
}

func TestValidChannel(t *testing.T) {
	for _, name := range []string{"anthropic-direct", "linkapi group 3", "google_ai_studio"} {
		if err := ValidChannel(name); err != nil {
			t.Errorf("%q: %v", name, err)
		}
	}
	for _, name := range []string{"", "  ", "https://x", "relay.example.com/v1", "x?key=1", "user@host"} {
		if err := ValidChannel(name); err == nil {
			t.Errorf("%q accepted", name)
		}
	}
}
