// Package bank stores the reference answers of the tier layer: one file per
// model and request environment, with the raw answers, the seed and template
// of every prompt and where and when the answers were collected. Features are
// not stored; they are parsed again from the raw answers on load, so a bank
// stays usable when the parser improves.
package bank

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/AlexeyPevz/modeltruth/fingerprint"
	"github.com/AlexeyPevz/modeltruth/probe"
)

// Format and Version identify the file format. Any change a reader of the
// previous version cannot ignore bumps Version, and Read rejects every other
// version.
const (
	Format  = "modeltruth-bank"
	Version = 1
)

// maxBankBytes bounds what Read accepts.
const maxBankBytes = 64 << 20

// Bank is one bank file: references of one model in one environment.
type Bank struct {
	Format      string      `json:"format"`
	Version     int         `json:"version"`
	Model       string      `json:"model"`
	Environment Environment `json:"environment"`
	Provenance  Provenance  `json:"provenance"`
	Samples     []Sample    `json:"samples"`
}

// Environment is the request shape the references were taken in. A
// fingerprint taken behind a long system prompt differs from one taken on a
// clean request, so a probe is only comparable with a bank of the same
// environment (docs/methodology.md, "Environment effect").
type Environment struct {
	Protocol probe.Protocol `json:"protocol"`
	// SystemPromptChars is the length of the system prompt in characters;
	// zero means none was sent. The text itself is not stored.
	SystemPromptChars int `json:"system_prompt_chars"`
	MaxTokens         int `json:"max_tokens"`
	// Template is the fingerprint template version of every prompt, Spec
	// what every prompt asked for; each sample records its own seed.
	Template string           `json:"template"`
	Spec     fingerprint.Spec `json:"spec"`
}

// Provenance says where and when the samples were collected. Channel is a
// name chosen by the collector ("anthropic-direct"), never a URL or a key.
type Provenance struct {
	Channel  string    `json:"channel"`
	Tool     string    `json:"tool"`
	Started  time.Time `json:"started"`
	Finished time.Time `json:"finished"`
	// SampleCount is len(Samples). Requested is how many samples were asked
	// for; Failed counts requests that ended in a transport or HTTP error
	// and are not samples, Retries the repeated attempts of all requests.
	SampleCount int  `json:"sample_count"`
	Requested   int  `json:"requested"`
	Failed      int  `json:"failed"`
	Retries     int  `json:"retries"`
	Interrupted bool `json:"interrupted,omitempty"`
}

// Sample is one answered prompt. Answer is the raw text of the response;
// ResponseModel is the model name the endpoint echoed.
type Sample struct {
	Seed          uint64    `json:"seed"`
	Time          time.Time `json:"time"`
	ResponseModel string    `json:"response_model,omitempty"`
	InputTokens   int64     `json:"input_tokens"`
	OutputTokens  int64     `json:"output_tokens"`
	Answer        string    `json:"answer"`
}

// Read decodes and validates a bank. A file of another format or version is
// rejected, as are unknown fields.
func Read(r io.Reader) (*Bank, error) {
	raw, err := io.ReadAll(io.LimitReader(r, maxBankBytes+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > maxBankBytes {
		return nil, fmt.Errorf("bank: file exceeds %d bytes", maxBankBytes)
	}
	// The header is read on its own first, so a file of another version is
	// reported as such rather than as a field this version does not know.
	var head struct {
		Format  string `json:"format"`
		Version int    `json:"version"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, fmt.Errorf("bank: decode: %w", err)
	}
	if head.Format != Format {
		return nil, fmt.Errorf("bank: format %q, want %q", head.Format, Format)
	}
	if head.Version != Version {
		return nil, fmt.Errorf("bank: format version %d, this modeltruth reads version %d", head.Version, Version)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var b Bank
	if err := dec.Decode(&b); err != nil {
		return nil, fmt.Errorf("bank: decode: %w", err)
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return &b, nil
}

// Load reads the bank file at path.
func Load(path string) (*Bank, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := Read(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return b, nil
}

// Validate checks that b is a complete bank of this version.
func (b *Bank) Validate() error {
	env, prov := b.Environment, b.Provenance
	switch {
	case b.Format != Format || b.Version != Version:
		return fmt.Errorf("bank: format %q version %d, want %q version %d", b.Format, b.Version, Format, Version)
	case b.Model == "":
		return errors.New("bank: model is empty")
	case env.Protocol != probe.ProtocolOpenAI && env.Protocol != probe.ProtocolAnthropic:
		return fmt.Errorf("bank: unknown protocol %q", env.Protocol)
	case env.SystemPromptChars < 0:
		return errors.New("bank: negative system prompt length")
	case env.MaxTokens <= 0:
		return errors.New("bank: max_tokens must be positive")
	case env.Template == "":
		return errors.New("bank: template version is empty")
	case prov.SampleCount != len(b.Samples):
		return fmt.Errorf("bank: sample_count %d, but the file holds %d samples", prov.SampleCount, len(b.Samples))
	}
	if _, err := fingerprint.Generate(env.Spec, 0); err != nil {
		return fmt.Errorf("bank: spec: %w", err)
	}
	return ValidChannel(prov.Channel)
}

// ValidChannel rejects channel names that are empty or look like a URL or a
// credential: a bank is meant to be published.
func ValidChannel(name string) error {
	switch {
	case strings.TrimSpace(name) == "":
		return errors.New("bank: channel name is empty")
	case strings.Contains(name, "://"), strings.ContainsAny(name, "/?@=#"):
		return fmt.Errorf("bank: channel %q looks like a URL; use a name such as anthropic-direct", name)
	}
	return nil
}

// Write encodes b as indented JSON.
func (b *Bank) Write(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(b)
}

// MismatchError lists how a probe's environment differs from a bank's.
type MismatchError struct {
	Model       string
	Differences []string
}

func (e *MismatchError) Error() string {
	return fmt.Sprintf("bank of %s was collected in another environment: %s", e.Model, strings.Join(e.Differences, "; "))
}

// CheckEnvironment reports whether a probe sent in env is comparable with
// b. It returns a *MismatchError naming every field that differs, or nil.
func (b *Bank) CheckEnvironment(env Environment) error {
	var diffs []string
	have := b.Environment
	add := func(field string, bank, probe any) {
		diffs = append(diffs, fmt.Sprintf("%s: bank %v, probe %v", field, bank, probe))
	}
	if have.Protocol != env.Protocol {
		add("protocol", have.Protocol, env.Protocol)
	}
	if have.SystemPromptChars != env.SystemPromptChars {
		add("system prompt chars", have.SystemPromptChars, env.SystemPromptChars)
	}
	if have.MaxTokens != env.MaxTokens {
		add("max_tokens", have.MaxTokens, env.MaxTokens)
	}
	if have.Template != env.Template {
		add("template", have.Template, env.Template)
	}
	if !sameSpec(have.Spec, env.Spec) {
		add("spec", specString(have.Spec), specString(env.Spec))
	}
	if len(diffs) == 0 {
		return nil
	}
	return &MismatchError{Model: b.Model, Differences: diffs}
}

// sameSpec compares specs regardless of category order, which does not
// change the prompts.
func sameSpec(a, b fingerprint.Spec) bool {
	if a.Numbers != b.Numbers || len(a.Categories) != len(b.Categories) {
		return false
	}
	set := map[fingerprint.Category]bool{}
	for _, c := range a.Categories {
		set[c] = true
	}
	for _, c := range b.Categories {
		if !set[c] {
			return false
		}
	}
	return true
}

func specString(s fingerprint.Spec) string {
	return fmt.Sprintf("%d categories, %d integers %d..%d", len(s.Categories), s.Numbers.Count, s.Numbers.Min, s.Numbers.Max)
}

// Reference regenerates every sample's prompt from its seed and parses the
// answer. It fails when the bank was made from another template version than
// this modeltruth generates, since its prompts cannot be regenerated.
func (b *Bank) Reference() (fingerprint.Reference, error) {
	if b.Environment.Template != fingerprint.TemplateVersion {
		return fingerprint.Reference{}, fmt.Errorf("bank of %s uses template %s, this modeltruth generates %s", b.Model, b.Environment.Template, fingerprint.TemplateVersion)
	}
	ref := fingerprint.Reference{Model: b.Model}
	for _, s := range b.Samples {
		p, err := fingerprint.Generate(b.Environment.Spec, s.Seed)
		if err != nil {
			return fingerprint.Reference{}, err
		}
		ref.Samples = append(ref.Samples, fingerprint.Parse(p, s.Answer))
	}
	return ref, nil
}
