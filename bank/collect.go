package bank

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/AlexeyPevz/modeltruth/fingerprint"
	"github.com/AlexeyPevz/modeltruth/probe"
)

// DefaultMaxTokens leaves room for the 200 integers of the default spec.
const DefaultMaxTokens = 2048

// Sender sends one request. probe.Endpoint implements it.
type Sender interface {
	Send(ctx context.Context, req probe.Request) (probe.Response, error)
}

// CollectOptions describe one collection run. Seed draws the seed of each
// prompt (rand.Uint64 when nil).
type CollectOptions struct {
	Model    string
	Protocol probe.Protocol
	Channel  string
	Tool     string
	Samples  int
	// System is sent as the system prompt of every request; the bank
	// records only its length.
	System    string
	MaxTokens int
	Spec      fingerprint.Spec
	Seed      func() uint64
	// OnError is called for every failed request.
	OnError func(err error)
}

// Collect sends opts.Samples fingerprint prompts one at a time, each with a
// fresh seed, and returns the bank of the answers. A request that fails is
// counted in Provenance.Failed and is not a sample. When ctx is cancelled the
// request in flight is dropped and the bank holds the samples collected so
// far, marked Interrupted.
func Collect(ctx context.Context, s Sender, opts CollectOptions) (*Bank, error) {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = DefaultMaxTokens
	}
	if opts.Seed == nil {
		opts.Seed = rand.Uint64
	}
	switch {
	case opts.Model == "":
		return nil, errors.New("bank: model is required")
	case opts.Protocol != probe.ProtocolOpenAI && opts.Protocol != probe.ProtocolAnthropic:
		return nil, fmt.Errorf("bank: unknown protocol %q", opts.Protocol)
	case opts.Samples < 1:
		return nil, errors.New("bank: samples must be at least 1")
	}
	if err := ValidChannel(opts.Channel); err != nil {
		return nil, err
	}
	if _, err := fingerprint.Generate(opts.Spec, 0); err != nil {
		return nil, err
	}
	b := &Bank{
		Format:  Format,
		Version: Version,
		Model:   opts.Model,
		Environment: Environment{
			Protocol:          opts.Protocol,
			SystemPromptChars: utf8.RuneCountInString(opts.System),
			MaxTokens:         opts.MaxTokens,
			Template:          fingerprint.TemplateVersion,
			Spec:              opts.Spec,
		},
		Provenance: Provenance{
			Channel:   opts.Channel,
			Tool:      opts.Tool,
			Started:   now(),
			Requested: opts.Samples,
		},
		Samples: []Sample{},
	}
	for range opts.Samples {
		if ctx.Err() != nil {
			break
		}
		p, err := fingerprint.Generate(opts.Spec, opts.Seed())
		if err != nil {
			return nil, err
		}
		resp, err := s.Send(ctx, probe.Request{System: opts.System, User: p.Text, MaxTokens: opts.MaxTokens})
		b.Provenance.Retries += resp.Retries
		if ctx.Err() != nil {
			break
		}
		if err == nil && strings.TrimSpace(resp.Text) == "" {
			// An error envelope sent with HTTP 200 decodes to an empty answer.
			err = fmt.Errorf("empty answer (HTTP %d)", resp.HTTPStatus)
		}
		if err != nil {
			b.Provenance.Failed++
			if opts.OnError != nil {
				opts.OnError(err)
			}
			continue
		}
		b.Samples = append(b.Samples, Sample{
			Seed:          p.Seed,
			Time:          now(),
			ResponseModel: resp.Model,
			InputTokens:   resp.Usage.InputTokens,
			OutputTokens:  resp.Usage.OutputTokens,
			Answer:        resp.Text,
		})
	}
	b.Provenance.Interrupted = ctx.Err() != nil
	b.Provenance.Finished = now()
	b.Provenance.SampleCount = len(b.Samples)
	return b, nil
}

func now() time.Time { return time.Now().UTC().Truncate(time.Second) }
