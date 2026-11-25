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
under test. A run sends 31 short requests (`--repeats 3`: five prompts three
times, the tiny prompt twice as a stream, then 14 tokenizer prompts once). For
a Claude model with `--protocol anthropic`, or Gemini 3 and later over an
OpenAI-compatible API, it then sends up to 6 signature requests (two round
trips of three; one more when the endpoint refuses adaptive thinking and the
request is repeated with manual thinking). These ask for thinking and a tool
call, so they bill more output than the other probes.

Requests that fail with HTTP 429, 500, 502, 503, 504 or a timeout are repeated
up to `--retries` times (default `2`; `--retries 0` sends every request once).
The pause before a retry follows the endpoint's `Retry-After` header, otherwise
1s, 2s, ...; it never exceeds `--max-wait` (default `30s`), and a `Retry-After`
longer than that ends the retries for that request. Other 4xx errors are not
retried. Requests are still sent one at a time, never in parallel. The report
shows how many retries were made; a request that succeeded on a retry is an
ordinary sample, and one that ran out of retries is listed under failed
requests with its last HTTP status.

Ctrl-C stops the run and prints the report for the requests answered so far,
marked as interrupted; the request in flight is dropped.

Exit codes: `0` no failed checks, `1` usage error or no request succeeded
(verdict `error`, failed requests are listed in the report; also after Ctrl-C
before the first answer), `2` at least one check failed.

## What is checked today

| Check | Detects | How |
|---|---|---|
| `hidden_prompt` | The relay prepends its own system prompt (and bills you for it) | A 30-character probe or a short realistic task must not be billed as hundreds of extra input tokens |
| `channel_consistency` | One model name served by several upstream channels | Identical requests must report the same input size |
| `usage_sanity` | Fabricated or under-reported usage | A ~700-character prompt cannot tokenize to fewer than len/6 tokens |
| `response_shape` | Rewritten or foreign responses | Anthropic message id format, echoed model name |
| `stream_shape` | Streams that are not the vendor's, or billed differently | The tiny prompt sent twice with `stream: true`: Anthropic event order (`message_start` → `content_block_*` → `message_delta` → `message_stop`), OpenAI usage in the final chunk and `[DONE]`, and the same input size as without streaming |
| `channel_signature` | A relay that is not backed by the vendor it sells | Claude thinking signatures and Gemini 3 thought signatures must round-trip: the endpoint must accept its own signature in the next turn and reject the same signature with one changed byte |
| `tokenizer` | Another model behind the name, fabricated usage | The input-token difference that 12 fixed texts add to a base prompt must match the claimed family's tokenizer (o200k, Claude, Gemini, DeepSeek, GLM, Qwen, MiniMax) |

The first five catch the cheap tricks and tell you how the endpoint is wired.
`channel_signature` proves that the vendor checks the reasoning signatures, so
the channel is real; it does not tell which model or tier answered.
`tokenizer` identifies the tokenizer family, so a GPT sold as Gemini or an old
GPT-4 sold as a new GPT fails; it cannot tell tiers that share a tokenizer
(Opus from Sonnet), which is what the fingerprint layers below are for.

## Reference bank

Tier fingerprints need reference answers of every model, taken in the same
request environment as the probe. `modeltruth collect` builds such a bank from
a key you trust (the vendor's own API, or a channel you have verified):

```sh
export MODELTRUTH_API_KEY=sk-...
modeltruth collect --base-url https://api.anthropic.com --protocol anthropic \
  --model claude-opus-5-5 --channel anthropic-direct --samples 30 --out opus-5-5.json
```

It sends `--samples` fingerprint prompts (default 30) one at a time, each made
from a fresh seed, and writes one JSON file per model and environment. Options:
`--system-file FILE` sends the file's text as the system prompt of every
request (the bank records only its length, for references taken behind a
client's system prompt), `--max-tokens` (default 2048), and `--protocol`,
`--api-key-env`, `--timeout`, `--retries` and `--max-wait` as for `check`.

The file holds the format version, the model, the environment (protocol,
system prompt length, `max_tokens`, template version and what the prompts
asked for), the provenance (the `--channel` name, the modeltruth version,
start and end time, samples asked for, collected and failed, retries) and
every sample with its seed, time, echoed model name, token usage and raw
answer. Features are parsed again from the raw answers when a bank is loaded.
See [bank/testdata/synthetic-bank.json](bank/testdata/synthetic-bank.json)
for an example made from synthetic answers.

Nothing that identifies your access is written: not the key, not the
authorization headers, not the base URL or its query. `--channel` must be a
name, not a URL. If the endpoint echoes the key, the base URL, its host or
query values, or the whole system prompt in an answer or an
error, they are replaced with `[redacted]` in the bank and on the terminal. A request that fails (HTTP
error, timeout, unreadable body) is not a sample: it is counted under
`failed` and listed on the terminal. An existing file is never overwritten.
Ctrl-C stops the run and writes the samples collected so far, marked as
interrupted.

Exit codes: `0` all samples written, `1` usage error, no sample collected (no
file is left) or the file could not be written, `2` the bank was written with
fewer samples than asked for (failed requests or Ctrl-C).

A bank of another format version is rejected on load, and a probe sent in an
environment other than the bank's is reported as a mismatch naming each
differing field. The tier check that uses banks is not wired into `check` yet.

## Roadmap

1. **Channel proofs**: verify `channel_signature` live against Anthropic's and
   Google's own APIs and paid relays.
2. **Generation**: exact tokenizer deltas on fixed texts, error and parameter
   contracts, knowledge cutoff.
3. **Tier**: statistical fingerprints of "random" choices, scored against every
   model of the family, with abstention when evidence is weak.
4. **Reference bank**: versioned, with provenance (channel, date, sample count)
   and raw samples, plus `modeltruth collect` to build your own from a trusted key.
   The format and `collect` ship (see above); a published bank follows.
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
  succeeded ends with verdict `error` instead of a silent pass. Retries are not
  evidence either: no check reads them, they are only counted in the report.

## Prior art

The method draws on public work by others; no code from these projects is
included. See [docs/methodology.md](docs/methodology.md#prior-art) for what was
learned from each: ModelTrace, BazaarLink LLMprobe-engine, Veridrop, cctest,
ztest, hvoy, LMSpeed, and the papers "One Token Is Enough" (arXiv 2607.10252)
and "Model Equality Testing" (ICLR 2025).

## License

Apache-2.0. See [LICENSE](LICENSE).
