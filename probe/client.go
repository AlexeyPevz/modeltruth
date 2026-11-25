// Package probe sends single requests to an LLM endpoint and normalizes the
// parts of the response that checks need: ids, model echo, text and usage.
package probe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Protocol is the wire format the endpoint speaks.
type Protocol string

const (
	ProtocolOpenAI    Protocol = "openai"
	ProtocolAnthropic Protocol = "anthropic"
)

// maxBodyBytes bounds how much of a response is read; probes ask for short
// answers, so anything larger is itself suspicious and not worth buffering.
const maxBodyBytes = 1 << 20

// DefaultMaxWait caps the pause before one retry when Endpoint.MaxWait is
// zero.
const DefaultMaxWait = 30 * time.Second

// Request is one probe prompt.
type Request struct {
	System    string
	User      string
	MaxTokens int
	// Stream asks for a server-sent event stream (with usage, on OpenAI via
	// stream_options.include_usage); Response.Stream then describes it.
	Stream bool
	// Signed asks for a reasoning turn that should end in a call to
	// SignatureTool: on Anthropic the tool with adaptive thinking, or
	// with ManualThinking extended thinking (budget ThinkingBudget) for
	// models that predate adaptive thinking; on OpenAI-compatible APIs the
	// tool with reasoning_effort "low". Response.Turn then holds the
	// assistant turn.
	Signed         bool
	ManualThinking bool
	// Reply continues a signed request: User again, then Reply.Message as
	// the assistant turn, then a tool result for each of its tool calls,
	// with the same tool and thinking settings as the request it answered.
	Reply *Turn
}

// Usage holds token counts as reported by the endpoint. InputTokens is the
// total prompt size including cached and cache-written tokens, because a
// hidden system prompt is often served from cache and must still count.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
}

// Stream describes a streamed response. Events lists the event types in
// arrival order: the Anthropic "type" of each event, or for OpenAI "chunk"
// (choices only), "usage" (usage only), "chunk+usage" (both), "empty"
// (neither) and "[DONE]".
// Usage is set when the stream reported the input size, which is what
// checks compare against non-streamed requests. JSON is set
// when the endpoint answered the stream request with a plain JSON body.
type Stream struct {
	Protocol Protocol
	Events   []string
	Usage    bool
	JSON     bool
}

// Response is the normalized result of one request. HTTPStatus and Latency
// describe the last attempt; Retries counts the attempts before it.
type Response struct {
	HTTPStatus int
	ID         string
	Model      string
	Text       string
	Usage      Usage
	Latency    time.Duration
	Retries    int
	// Stream is set for stream requests only.
	Stream *Stream
	// Turn is set for successful Signed requests only.
	Turn *Turn
}

// Endpoint is an API under test.
type Endpoint struct {
	BaseURL  string
	APIKey   string
	Model    string
	Protocol Protocol
	HTTP     *http.Client
	// Retries is how many times a request is repeated after HTTP 429, 500,
	// 502, 503, 504 or a timeout. Zero sends every request once.
	Retries int
	// MaxWait caps the pause before a retry (DefaultMaxWait when zero). A
	// Retry-After longer than MaxWait ends the retries instead of being cut
	// short, so the endpoint is never asked again earlier than it allowed.
	MaxWait time.Duration
}

// Send performs one request, repeating it up to Retries times on transient
// failures. A non-2xx status is returned as an error that still carries the
// HTTP status of the last attempt in the response.
func (e Endpoint) Send(ctx context.Context, req Request) (Response, error) {
	client := e.HTTP
	if client == nil {
		client = &http.Client{Timeout: 120 * time.Second}
	}
	url, body, headers, err := e.build(req)
	if err != nil {
		return Response{}, err
	}
	maxWait := e.MaxWait
	if maxWait <= 0 {
		maxWait = DefaultMaxWait
	}
	for attempt := 0; ; attempt++ {
		out, retryAfter, err := e.sendOnce(ctx, client, url, body, headers, req)
		out.Retries = attempt
		if err == nil || attempt >= e.Retries || !retryable(ctx, out.HTTPStatus, err) {
			return out, err
		}
		wait := backoff(attempt, maxWait)
		if retryAfter >= 0 {
			if retryAfter > maxWait {
				return out, fmt.Errorf("%w (Retry-After %s exceeds max wait %s)", err, retryAfter, maxWait)
			}
			wait = retryAfter
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return out, fmt.Errorf("%w (retry interrupted: %v)", err, ctx.Err())
		case <-timer.C:
		}
	}
}

// sendOnce performs a single attempt. retryAfter is the parsed Retry-After
// header, or -1 when the response has none.
func (e Endpoint) sendOnce(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string, req Request) (out Response, retryAfter time.Duration, err error) {
	retryAfter = -1
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Response{}, retryAfter, err
	}
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}
	start := time.Now()
	resp, err := client.Do(httpReq)
	if err != nil {
		return Response{}, retryAfter, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		retryAfter = parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return Response{HTTPStatus: resp.StatusCode}, retryAfter, err
	}
	out = Response{HTTPStatus: resp.StatusCode, Latency: time.Since(start)}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return out, retryAfter, fmt.Errorf("probe: HTTP %d: %s", resp.StatusCode, snippet(raw))
	}
	if req.Stream && !strings.Contains(resp.Header.Get("Content-Type"), "json") {
		if len(raw) >= maxBodyBytes {
			return out, retryAfter, fmt.Errorf("probe: stream exceeds %d bytes", maxBodyBytes)
		}
		return out, retryAfter, e.parseStream(raw, &out)
	}
	switch e.Protocol {
	case ProtocolAnthropic:
		err = parseAnthropic(raw, &out)
	default:
		err = parseOpenAI(raw, &out)
	}
	if req.Stream {
		out.Stream = &Stream{Protocol: e.protocol(), JSON: true, Usage: out.Usage.InputTokens > 0}
	}
	if req.Signed && err == nil {
		if out.Turn, err = parseTurn(e.protocol(), raw); err == nil {
			out.Turn.ManualThinking = req.ManualThinking
		}
	}
	return out, retryAfter, err
}

func (e Endpoint) protocol() Protocol {
	if e.Protocol == ProtocolAnthropic {
		return ProtocolAnthropic
	}
	return ProtocolOpenAI
}

// retryable reports whether a failed attempt is worth repeating: rate limits,
// gateway-type 5xx and timeouts. A cancelled run and other 4xx or 5xx never
// are, even when reading their body timed out.
func retryable(ctx context.Context, status int, err error) bool {
	if ctx.Err() != nil {
		return false
	}
	switch status {
	case http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	}
	if status >= 400 {
		return false
	}
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

// backoff is the pause before retry attempt+1 when the endpoint gave no
// Retry-After: 1s, 2s, 4s, ... capped at maxWait.
func backoff(attempt int, maxWait time.Duration) time.Duration {
	if attempt > 10 {
		return maxWait
	}
	return min(time.Second<<attempt, maxWait)
}

// parseRetryAfter reads a Retry-After value in delay-seconds or HTTP-date
// form; it returns -1 when the header is absent or malformed.
func parseRetryAfter(v string, now time.Time) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return -1
	}
	if secs, err := strconv.ParseInt(v, 10, 64); err == nil {
		if secs < 0 {
			return -1
		}
		if secs > int64(24*time.Hour/time.Second) {
			secs = int64(24 * time.Hour / time.Second)
		}
		return time.Duration(secs) * time.Second
	}
	if at, err := http.ParseTime(v); err == nil {
		return max(at.Sub(now), 0)
	}
	return -1
}

func (e Endpoint) build(req Request) (string, []byte, map[string]string, error) {
	base := strings.TrimRight(strings.TrimSpace(e.BaseURL), "/")
	if base == "" {
		return "", nil, nil, errors.New("probe: base URL is required")
	}
	if !strings.HasSuffix(base, "/v1") {
		base += "/v1"
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}
	headers := map[string]string{"Content-Type": "application/json"}
	switch e.Protocol {
	case ProtocolAnthropic:
		payload := map[string]any{
			"model":      e.Model,
			"max_tokens": maxTokens,
			"messages":   anthropicMessages(req),
		}
		if signed(req) {
			anthropicSigned(payload, maxTokens, manualThinking(req))
		}
		if req.System != "" {
			payload["system"] = req.System
		}
		if req.Stream {
			payload["stream"] = true
		}
		body, err := json.Marshal(payload)
		headers["x-api-key"] = e.APIKey
		headers["anthropic-version"] = "2023-06-01"
		return base + "/messages", body, headers, err
	case ProtocolOpenAI, "":
		messages := []any{}
		if req.System != "" {
			messages = append(messages, map[string]any{"role": "system", "content": req.System})
		}
		messages = append(messages, map[string]any{"role": "user", "content": req.User})
		if req.Reply != nil {
			messages = openAIReply(messages, req.Reply)
		}
		payload := map[string]any{"model": e.Model, "max_tokens": maxTokens, "messages": messages}
		if signed(req) {
			openAISigned(payload)
		}
		if req.Stream {
			payload["stream"] = true
			payload["stream_options"] = map[string]any{"include_usage": true}
		}
		body, err := json.Marshal(payload)
		headers["Authorization"] = "Bearer " + e.APIKey
		return base + "/chat/completions", body, headers, err
	default:
		return "", nil, nil, fmt.Errorf("probe: unknown protocol %q", e.Protocol)
	}
}

type openAIUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u openAIUsage) usage() Usage {
	return Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		CachedTokens: u.PromptTokensDetails.CachedTokens,
	}
}

type anthropicUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
}

func (u anthropicUsage) usage() Usage {
	return Usage{
		InputTokens:  u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens,
		OutputTokens: u.OutputTokens,
		CachedTokens: u.CacheReadInputTokens,
	}
}

func parseOpenAI(raw []byte, out *Response) error {
	var d struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage openAIUsage `json:"usage"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("probe: decode OpenAI response: %w", err)
	}
	out.ID, out.Model = d.ID, d.Model
	if len(d.Choices) > 0 {
		out.Text = d.Choices[0].Message.Content
	}
	out.Usage = d.Usage.usage()
	return nil
}

func parseAnthropic(raw []byte, out *Response) error {
	var d struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage anthropicUsage `json:"usage"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("probe: decode Anthropic response: %w", err)
	}
	out.ID, out.Model = d.ID, d.Model
	var text strings.Builder
	for _, block := range d.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	out.Text = text.String()
	out.Usage = d.Usage.usage()
	return nil
}

func snippet(raw []byte) string {
	s := strings.TrimSpace(string(raw))
	if len(s) > 200 {
		s = s[:200] + "..."
	}
	return s
}
