package probe

import (
	"encoding/json"
	"fmt"
	"strings"
)

type sseEvent struct {
	name, data string
}

// parseSSE splits a server-sent event body into events. Comment lines and
// fields other than event and data are ignored; multi-line data is joined
// with newlines as the SSE specification requires.
func parseSSE(raw []byte) []sseEvent {
	var (
		out  []sseEvent
		name string
		data []string
	)
	flush := func() {
		if len(data) > 0 {
			out = append(out, sseEvent{name: name, data: strings.Join(data, "\n")})
		}
		name, data = "", nil
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSuffix(line, "\r")
		if line == "" {
			flush()
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			name = value
		case "data":
			data = append(data, value)
		}
	}
	flush()
	return out
}

// parseStream fills out from an event stream. An error event inside the
// stream is returned as an error: it is an upstream failure, not a shape.
func (e Endpoint) parseStream(raw []byte, out *Response) error {
	events := parseSSE(raw)
	if e.protocol() == ProtocolAnthropic {
		return parseAnthropicStream(events, out)
	}
	return parseOpenAIStream(events, out)
}

func parseOpenAIStream(events []sseEvent, out *Response) error {
	st := &Stream{Protocol: ProtocolOpenAI}
	var (
		text  strings.Builder
		usage *openAIUsage
	)
	for _, ev := range events {
		if strings.TrimSpace(ev.data) == "[DONE]" {
			st.Events = append(st.Events, "[DONE]")
			continue
		}
		var c struct {
			ID      string `json:"id"`
			Model   string `json:"model"`
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *openAIUsage    `json:"usage"`
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(ev.data), &c); err != nil {
			return fmt.Errorf("probe: decode OpenAI stream chunk: %w", err)
		}
		if len(c.Error) > 0 && string(c.Error) != "null" {
			return fmt.Errorf("probe: stream error: %s", snippet(c.Error))
		}
		if out.ID == "" {
			out.ID = c.ID
		}
		if out.Model == "" {
			out.Model = c.Model
		}
		if len(c.Choices) == 0 && c.Usage == nil {
			continue
		}
		for _, ch := range c.Choices {
			text.WriteString(ch.Delta.Content)
		}
		kind := "chunk"
		if c.Usage != nil {
			usage = c.Usage
			kind = "usage"
			if len(c.Choices) > 0 {
				kind = "chunk+usage"
			}
		}
		st.Events = append(st.Events, kind)
	}
	out.Text = text.String()
	if usage != nil {
		out.Usage = usage.usage()
	}
	st.Usage = out.Usage.InputTokens > 0
	out.Stream = st
	return nil
}

func parseAnthropicStream(events []sseEvent, out *Response) error {
	st := &Stream{Protocol: ProtocolAnthropic}
	var (
		text         strings.Builder
		start, delta *anthropicUsage
	)
	for _, ev := range events {
		var d struct {
			Type    string `json:"type"`
			Message struct {
				ID    string          `json:"id"`
				Model string          `json:"model"`
				Usage *anthropicUsage `json:"usage"`
			} `json:"message"`
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
			Usage *anthropicUsage `json:"usage"`
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(ev.data), &d); err != nil {
			return fmt.Errorf("probe: decode Anthropic stream event: %w", err)
		}
		typ := d.Type
		if typ == "" {
			typ = ev.name
		}
		switch typ {
		case "error":
			return fmt.Errorf("probe: stream error: %s", snippet(d.Error))
		case "message_start":
			out.ID, out.Model = d.Message.ID, d.Message.Model
			start = d.Message.Usage
		case "content_block_delta":
			if d.Delta.Type == "text_delta" {
				text.WriteString(d.Delta.Text)
			}
		case "message_delta":
			delta = d.Usage
		}
		st.Events = append(st.Events, typ)
	}
	out.Text = text.String()
	out.Usage = mergeAnthropicUsage(start, delta).usage()
	st.Usage = out.Usage.InputTokens > 0
	out.Stream = st
	return nil
}

// mergeAnthropicUsage combines the usage of message_start with the final
// counts of message_delta: every non-zero field of the latter wins.
func mergeAnthropicUsage(start, delta *anthropicUsage) anthropicUsage {
	var u anthropicUsage
	if start != nil {
		u = *start
	}
	if delta == nil {
		return u
	}
	for _, f := range []struct{ dst, src *int64 }{
		{&u.InputTokens, &delta.InputTokens},
		{&u.OutputTokens, &delta.OutputTokens},
		{&u.CacheCreationInputTokens, &delta.CacheCreationInputTokens},
		{&u.CacheReadInputTokens, &delta.CacheReadInputTokens},
	} {
		if *f.src > 0 {
			*f.dst = *f.src
		}
	}
	return u
}
