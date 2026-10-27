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
	"net/http"
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

// Request is one probe prompt.
type Request struct {
	System    string
	User      string
	MaxTokens int
}

// Usage holds token counts as reported by the endpoint. InputTokens is the
// total prompt size including cached and cache-written tokens, because a
// hidden system prompt is often served from cache and must still count.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
}

// Response is the normalized result of one request.
type Response struct {
	HTTPStatus int
	ID         string
	Model      string
	Text       string
	Usage      Usage
	Latency    time.Duration
}

// Endpoint is an API under test.
type Endpoint struct {
	BaseURL  string
	APIKey   string
	Model    string
	Protocol Protocol
	HTTP     *http.Client
}

// Send performs one request. A non-2xx status is returned as an error that
// still carries the HTTP status in the response.
func (e Endpoint) Send(ctx context.Context, req Request) (Response, error) {
	client := e.HTTP
	if client == nil {
		client = &http.Client{Timeout: 120 * time.Second}
	}
	url, body, headers, err := e.build(req)
	if err != nil {
		return Response{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Response{}, err
	}
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}
	start := time.Now()
	resp, err := client.Do(httpReq)
	if err != nil {
		return Response{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return Response{HTTPStatus: resp.StatusCode}, err
	}
	out := Response{HTTPStatus: resp.StatusCode, Latency: time.Since(start)}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return out, fmt.Errorf("probe: HTTP %d: %s", resp.StatusCode, snippet(raw))
	}
	switch e.Protocol {
	case ProtocolAnthropic:
		err = parseAnthropic(raw, &out)
	default:
		err = parseOpenAI(raw, &out)
	}
	return out, err
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
			"messages":   []map[string]any{{"role": "user", "content": req.User}},
		}
		if req.System != "" {
			payload["system"] = req.System
		}
		body, err := json.Marshal(payload)
		headers["x-api-key"] = e.APIKey
		headers["anthropic-version"] = "2023-06-01"
		return base + "/messages", body, headers, err
	case ProtocolOpenAI, "":
		messages := []map[string]any{}
		if req.System != "" {
			messages = append(messages, map[string]any{"role": "system", "content": req.System})
		}
		messages = append(messages, map[string]any{"role": "user", "content": req.User})
		body, err := json.Marshal(map[string]any{"model": e.Model, "max_tokens": maxTokens, "messages": messages})
		headers["Authorization"] = "Bearer " + e.APIKey
		return base + "/chat/completions", body, headers, err
	default:
		return "", nil, nil, fmt.Errorf("probe: unknown protocol %q", e.Protocol)
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
		Usage struct {
			PromptTokens        int64 `json:"prompt_tokens"`
			CompletionTokens    int64 `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int64 `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("probe: decode OpenAI response: %w", err)
	}
	out.ID, out.Model = d.ID, d.Model
	if len(d.Choices) > 0 {
		out.Text = d.Choices[0].Message.Content
	}
	out.Usage = Usage{
		InputTokens:  d.Usage.PromptTokens,
		OutputTokens: d.Usage.CompletionTokens,
		CachedTokens: d.Usage.PromptTokensDetails.CachedTokens,
	}
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
		Usage struct {
			InputTokens              int64 `json:"input_tokens"`
			OutputTokens             int64 `json:"output_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		} `json:"usage"`
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
	out.Usage = Usage{
		InputTokens:  d.Usage.InputTokens + d.Usage.CacheCreationInputTokens + d.Usage.CacheReadInputTokens,
		OutputTokens: d.Usage.OutputTokens,
		CachedTokens: d.Usage.CacheReadInputTokens,
	}
	return nil
}

func snippet(raw []byte) string {
	s := strings.TrimSpace(string(raw))
	if len(s) > 200 {
		s = s[:200] + "..."
	}
	return s
}
