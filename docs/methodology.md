# Methodology

This document describes what modeltruth measures, why, and how much each
signal can be trusted. It is the reference for every check in the code.

## Threats

A relay that claims model **M** may instead serve:

1. a different vendor's model;
2. a cheaper tier of the same family (Sonnet or Haiku instead of Opus, a mini
   instead of the full model);
3. an older generation of the same tier;
4. the genuine model through a channel that changes behaviour or cost: a
   reverse-engineered subscription client with its own system prompt, a pool
   that mixes several such channels, or a proxy that rewrites usage.

No single signal covers all four. Cryptographic signatures prove the vendor and
channel but not the tier; behavioural fingerprints separate tiers but need
reference data; tokenizer counts separate generations but not tiers that share a
tokenizer. modeltruth therefore combines layers and reports each one separately.

## Layers

| Layer | Question | Signals | Needs reference | Failure policy |
|---|---|---|---|---|
| 1. Protocol | Is this response what the vendor's API returns? | id format, model echo, usage fields, SSE event order, hidden prompt size, consistency of identical requests | No | 2 consecutive failures |
| 2. Channel | Is the vendor actually behind it? | Claude thinking signature and Gemini thought signature round-trip; a tampered signature must be rejected | No | 2 consecutive failures |
| 3. Generation | Which generation answered? | Exact tokenizer deltas on fixed texts, parameter and error contracts, knowledge cutoff | Public tokenizers, or one official run | 2 consecutive failures |
| 4. Tier | Which tier answered? | Distributions of "random" choices (words, numbers), capability edges | Yes, per model and environment | 3 failures within 24 hours |
| 5. Signature model name | Which model signed the thinking block? | Model identifier inside the Claude signature, if the format allows it | No | 2 consecutive failures |

Layer 1 and the tokenizer part of layer 3 ship. The others are in development.

## Decision rules

- **No reference, no accusation.** A tier check reports `fail` only when the
  claimed model scores low *and* another candidate wins with posterior ≥ 0.85
  and a clear margin. "Unknown model", "not in bank", and close calls are
  `skip`.
- **Candidates are not narrowed by the claim.** The tier classifier scores every
  model in the bank, so a Sonnet served as Opus cannot pass just by being a
  Claude.
- **Environment must match.** A fingerprint taken behind a long system prompt
  differs from one taken on a clean request (see the pilot below). Reference and
  probe must use the same request shape.
- **Errors are not evidence.** 429, 5xx and timeouts are recorded and excluded;
  a run in which no request succeeded reports `error`, never `pass` or `skip`.

## Layer 1 thresholds

- `hidden_prompt`: every tiny and task prompt may be billed for half a token
  per character; more than 45 tokens beyond that is a warning, more than 285
  is a failure. The tiny prompt `Reply with the single word: <word>` is about
  30 characters (word drawn per run), so the limits stay near 60 and 300
  tokens for it; clean channels measured so far report 10–19. Each run also
  sends three task prompts drawn from a list of realistic requests (a
  multiplication, a short Python function, a SQL query, a pet question...).
  Some relays serve probe-like prompts clean and prepend their own system
  prompt only to work: on 2026-10-27 three A6 channels billed `Reply with OK.`
  as 13 tokens and a multiplication question as 4,399. Relays that front
  subscription clients typically add 4,000–8,000 tokens. Every request asks
  for a realistic output limit (`max_tokens` drawn from 256–4096): on
  2026-10-27 an A6 channel served all requests with `max_tokens` 16 clean
  and injected 1,000–4,400 tokens into 9 of 16 identical requests with 300.
- `channel_consistency`: identical prompts (tiny, medium and each task prompt)
  must not differ by more than 10% and 20 tokens in reported input. A larger
  spread means requests reach different upstream channels.
- `usage_sanity`: English text never tokenizes to fewer than one token per six
  characters, so the ~700-character prompt must report at least len/6 input
  tokens. Text with zero output tokens is a warning.
- `response_shape`: Anthropic message ids are `msg_01` + 22 base62 characters
  (`msg_vrtx_01…` on Vertex, `msg_bdrk_01…` on Bedrock). `msg_` followed by 32
  hex characters is a reverse-proxy signature. Gateways legitimately rewrite ids
  and model names, so this check only warns.

## Layer 3: tokenizer

Each run sends `Reply with OK.` twice (first and last) and the same prompt
plus a newline plus each of 12 fixed texts (Japanese, emoji, rare English
words, code, 50 digits, whitespace, Russian, Korean, Arabic, Hindi, math,
URL). The chat template and any relay system prompt add the same constant to
every request, so each difference is the text's token count under the serving
model's tokenizer.

- A difference below half of what any known tokenizer gives (50 digits billed
  as 4 tokens, zero or negative) is a failure: usage is estimated, not
  tokenized. When the base itself carries a hidden prompt, negative
  differences only mean that request was not injected and are ignored.
- A text billed far above its size carried a hidden prompt on that request
  alone and is ignored; fewer than 8 usable texts, or two base counts more
  than one token apart, skip the check.
- The rest must match a profile of the claimed family within one token, with
  one text allowed to miss. Matching another family, or no profile, fails.
- Differences that follow a profile of the claimed family at one constant
  factor of 1.04 or more fail as inflated usage: the right tokenizer, with
  every bill multiplied (a Claude channel billing 10% above Claude on all 12
  texts).

The bank (2026-10-28): o200k and cl100k from tiktoken, Gemma 3, DeepSeek-V3,
GLM-4, Qwen, Llama 3, Mistral Nemo and MiniMax-M1 counted offline from public
tokenizers; every public profile matched live channels exactly. Claude's
tokenizer is closed: its profile is the one LinkAPI (claude-opus-5-5,
claude-sonnet-5-5, claude-sonnet-5) and three A6 channels agreed on. On
2026-10-29 it was confirmed against Anthropic's own claude-opus-5-5 (8 texts)
and claude-sonnet-5-5 (2 texts), and o200k against OpenAI's gpt-6.1-sol
(4 texts), both served by Factory and measured from the session token usage. On A6 the
check found a GPT (o200k) sold as gemini-4-argon, DeepSeek sold as
glm-5.3-flashx, a cl100k model sold as gpt-6-luna and gpt-5.6-luna, word-count
usage on a channel selling Claude and Gemini, Claude channels billing 10% above
the Claude tokenizer, and four gpt-6.1-sol channels whose tokenizer is not
o200k.

Tiers that share a tokenizer (Opus, Sonnet, Haiku of one generation) are not
separated; a channel that switches upstreams between requests may pass one run.

## Pilot, 2026-10-26

A small feasibility study, not a validation.

**Tier separation with word choices.** Prompt: 80 "first-instinct" integers
1–100 plus 12 single-word choices (country, animal, colour, fruit, city, bird,
gem, element, month, weekday, letter, digit). References were collected for
Claude Opus 5.5, Sonnet 5.5, Haiku 5.5, Sonnet 5 and Haiku 4.5 in one client
environment (a coding-agent harness with its own system prompt), 5–6 samples
each. Leave-one-out classification with a Laplace-smoothed multinomial:

| Features | Accuracy, 3 tiers (Opus/Sonnet/Haiku 5.5) |
|---|---|
| 12 word choices | 15 / 16 |
| 80 integers | 4 / 16 (chance) |

- Tiers separate sharply on words: Haiku favours Paris, ruby, iron, March and
  Friday; Sonnet favours otter, teal, heron and opal (5 of 5); Opus favours
  emerald or sapphire, Iceland, Kyoto, and never opal.
- Generations within a tier do not separate: Sonnet 5 ≈ Sonnet 5.5, Haiku 4.5 ≈
  Haiku 5.5. Layer 3 is needed for that.
- The integer task as posed is too uniform. ModelTrace's design (1–355, 200–330
  numbers, Hellinger distance, multi-environment bank) is the next attempt.
- Haiku miscounted the 80 integers (72–82); Opus and Sonnet never did. This is
  a usable capability edge.

**Environment effect.** The same prompt sent to a relay's Opus 5.5 reached two
channels: clean requests (~220 input tokens) and requests with a ~7,300-token
hidden prompt. The hidden-prompt answers matched the harness references (posterior
0.89–0.99); the clean answers shifted (wolf, hawk, iron, Prague). Reference data
is only comparable within one request shape.

**Layer 1 on a live relay.** Without any reference, the deterministic checks
found: a 4,300–4,400-token hidden prompt on two models, Claude message ids in a
non-Anthropic format, per-request switching between channels on one model, and
on one model a 600-character prompt reported as 10 input tokens.

## Anti-gaming

Publishing the method lets relays special-case it. Mitigations:

- layer 1 already draws its answer word per run; fingerprint prompts will be
  generated per run from templates (ranges, counts, languages, ordering) with a
  recorded seed, and references calibrated across variants;
- cryptographic checks are not weakened by publication;
- monitoring mode mixes probes into ordinary-looking traffic and randomizes timing.

## Prior art

No code from these projects is used; ideas are credited here.

- **ModelTrace** (github.com/xqy2006/ModelTrace): long lists of "first-instinct"
  integers, Hellinger similarity to per-model centroids after projecting out
  environment directions, softmax calibration; closed-set caveat.
- **BazaarLink LLMprobe-engine** (AGPL-3.0, not copied): family first, then
  sub-model; "random X" bias probes scored by log-likelihood with abstention
  gates; references collected through a provider-pinned router; token-inflation,
  SSE and supply-chain probes.
- **Veridrop**: protocol and usage checks; its thinking-signature check only
  tests presence and length, which a relay can fake.
- **cctest**: number-picking attribution against a bank of official models;
  token-usage audit against a recorded official baseline (±10–20%).
- **ztest**: weighted probes with hard score caps, identity at family level only,
  reference sets require 30 samples over 3 days and expire after 120 days.
- **hvoy**: exact tokenizer deltas against official APIs, error contracts that
  differ between tiers, model name decoded from the Claude signature.
- **LMSpeed**: instruction-conflict and identity-override probes, canary-based
  context window test.
- Papers: "One Token Is Enough" (arXiv 2607.10252), "Model Equality Testing"
  (ICLR 2025), "Are You Getting What You Pay For?", rank-based uniformity test
  (arXiv 2506.06975), relay threat taxonomy (arXiv 2604.08407).
