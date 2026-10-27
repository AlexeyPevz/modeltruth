# modeltruth

Open, reproducible checks that an LLM API endpoint serves the model it claims.

API relays and aggregators often advertise one model and serve another: a cheaper
tier of the same family, an older generation, a different vendor, or the right
model behind a hidden system prompt that you pay for on every request. Most
public checkers are closed web services, so you cannot see what they test or
whether the test means anything. modeltruth publishes the method, the code, and
(as they are collected) the reference data, so anyone can rerun a check and
argue with the result.

> Status: early. The first release ships deterministic checks that need no
> reference data. Fingerprint checks that tell Opus from Sonnet from Haiku are
> in development; see [docs/methodology.md](docs/methodology.md).

## Quick start

Download a binary for your platform from
[Releases](https://github.com/AlexeyPevz/modeltruth/releases) (checksums in
`checksums.txt`), or build from source:

```sh
go install github.com/AlexeyPevz/modeltruth/cmd/modeltruth@latest

export MODELTRUTH_API_KEY=sk-...   # use a throwaway key with a small limit
modeltruth check --base-url https://relay.example.com/v1 --model claude-opus-5-5
modeltruth check --base-url https://relay.example.com --model claude-opus-5-5 --protocol anthropic --json
```

The key is read from an environment variable and is only sent to the endpoint
under test. A run sends 6 short requests (`--repeats 3`).

Exit codes: `0` no failed checks, `1` usage error or no request succeeded
(verdict `error`, failed requests are listed in the report), `2` at least one
check failed.

## What is checked today

| Check | Detects | How |
|---|---|---|
| `hidden_prompt` | The relay prepends its own system prompt (and bills you for it) | A 30-character probe or a short realistic task must not be billed as hundreds of extra input tokens |
| `channel_consistency` | One model name served by several upstream channels | Identical requests must report the same input size |
| `usage_sanity` | Fabricated or under-reported usage | A ~700-character prompt cannot tokenize to fewer than len/6 tokens |
| `response_shape` | Rewritten or foreign responses | Anthropic message id format, echoed model name |

None of these prove a model is genuine. They catch the cheap tricks and tell
you how the endpoint is wired; the fingerprint layers below are what identify
the model itself.

## Roadmap

1. **Channel proofs**: Claude thinking signatures and Gemini thought signatures
   must round-trip and a tampered signature must be rejected.
2. **Generation**: exact tokenizer deltas on fixed texts, error and parameter
   contracts, knowledge cutoff.
3. **Tier**: statistical fingerprints of "random" choices, scored against every
   model of the family, with abstention when evidence is weak.
4. **Reference bank**: versioned, with provenance (channel, date, sample count)
   and raw samples, plus `modeltruth collect` to build your own from a trusted key.
5. **Monitoring mode** for operators who need to probe their own upstreams on a schedule.

## Trust model

- The method is public. The word each probe asks the model to reply with is
  drawn per run, so a relay cannot match one fixed probe string; every request
  inside a run is identical so their sizes can be compared. Fully templated
  prompts arrive with the fingerprint layers.
- No reference, no accusation: a check that lacks reference data for the
  claimed model reports `skip`, never `fail`.
- Transport and upstream errors (429, 5xx, timeouts) are never counted as
  evidence of substitution. They are reported, and a run in which no request
  succeeded ends with verdict `error` instead of a silent pass.

## Prior art

The method draws on public work by others; no code from these projects is
included. See [docs/methodology.md](docs/methodology.md#prior-art) for what was
learned from each: ModelTrace, BazaarLink LLMprobe-engine, Veridrop, cctest,
ztest, hvoy, LMSpeed, and the papers "One Token Is Enough" (arXiv 2607.10252)
and "Model Equality Testing" (ICLR 2025).

## License

Apache-2.0. See [LICENSE](LICENSE).
