package checks

import (
	"fmt"
	"strings"

	"github.com/AlexeyPevz/modeltruth/probe"
)

// ProbeStream is the tiny prompt sent with stream: true.
const ProbeStream = "stream"

// StreamProbes is how many streamed requests one run sends: two, so that a
// stream problem must repeat on consecutive requests before it fails, as
// the layer 1 failure policy requires.
const StreamProbes = 2

// StreamShape checks streamed responses against the vendor's SSE contract:
// the Anthropic event order or the OpenAI chunk sequence, usage in the
// stream, and that the stream reports the same input as the identical
// request without streaming. A problem in one stream warns; the same kind of
// problem in every stream (at least StreamProbes of them) fails. A stream
// without usage only warns: many compatible servers omit it.
func StreamShape(samples []Sample) Finding {
	f := Finding{Check: "stream_shape"}
	var streams []probe.Response
	for _, s := range samples {
		if s.Err == nil && s.Probe == ProbeStream {
			streams = append(streams, s.Response)
		}
	}
	if len(streams) == 0 {
		f.Status, f.Summary = StatusSkip, "no successful streamed responses"
		return f
	}
	refs := inputs(ok(samples, ProbeTiny))
	var (
		problems, events []string
		streamInputs     []int64
		orderBad         []string
		mismatched       []int64
		noUsage          int
		protocol         probe.Protocol
	)
	for _, r := range streams {
		st := r.Stream
		if st == nil {
			st = &probe.Stream{}
		}
		if protocol == "" {
			protocol = st.Protocol
		}
		events = append(events, compactEvents(st.Events))
		if p := streamOrderProblem(st); p != "" {
			orderBad = append(orderBad, p)
			problems = append(problems, p)
		}
		if !st.Usage {
			noUsage++
			continue
		}
		in := r.Usage.InputTokens
		streamInputs = append(streamInputs, in)
		if len(refs) > 0 && !matchesAny(in, refs) {
			mismatched = append(mismatched, in)
			problems = append(problems, fmt.Sprintf("streamed request billed as %d input tokens, the same request without streaming as %v", in, refs))
		}
	}
	f.Evidence = map[string]any{
		"events":              events,
		"stream_input_tokens": streamInputs,
		"input_tokens":        refs,
		"problems":            problems,
	}
	n := len(streams)
	every := func(bad int) bool { return n >= StreamProbes && bad == n }
	switch {
	case every(len(orderBad)):
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("all %d streamed responses break the %s event order (%s): the stream is not the vendor's", n, protocol, orderBad[0])
	case every(len(mismatched)):
		f.Status = StatusFail
		f.Summary = fmt.Sprintf("streamed requests were billed as %v input tokens, the same request without streaming as %v: streams reach another channel or their usage is not real", mismatched, refs)
	case len(problems) > 0:
		f.Status = StatusWarn
		f.Summary = fmt.Sprintf("%d of %d streamed responses: %s", len(orderBad)+len(mismatched), n, problems[0])
	case noUsage > 0:
		f.Status = StatusWarn
		f.Summary = fmt.Sprintf("%d of %d streamed responses reported no usage", noUsage, n)
	case len(refs) == 0:
		f.Status, f.Summary = StatusPass, fmt.Sprintf("streamed responses follow the %s event order (no non-streamed request to compare usage with)", protocol)
	default:
		f.Status, f.Summary = StatusPass, fmt.Sprintf("streamed responses follow the %s event order and report the same input as non-streamed requests", protocol)
	}
	return f
}

// matchesAny reports whether in is within the channel_consistency spread of
// at least one reference count.
func matchesAny(in int64, refs []int64) bool {
	for _, r := range refs {
		if !spreadTooWide(min(in, r), max(in, r)) {
			return true
		}
	}
	return false
}

// streamOrderProblem describes the first violation of the protocol's stream
// contract, or returns "" when the stream is well-formed.
func streamOrderProblem(st *probe.Stream) string {
	switch {
	case st.JSON:
		return "the endpoint answered a stream request with a JSON body"
	case len(st.Events) == 0:
		return "the stream contains no events"
	case st.Protocol == probe.ProtocolAnthropic:
		return anthropicOrderProblem(st.Events)
	default:
		return openAIOrderProblem(st.Events)
	}
}

// anthropicEventTypes are the event types whose order is checked; ping and
// anything else may appear anywhere. Problems therefore never quote an event
// type the endpoint made up.
var anthropicEventTypes = map[string]bool{
	"message_start": true, "content_block_start": true, "content_block_delta": true,
	"content_block_stop": true, "message_delta": true, "message_stop": true,
}

// anthropicOrderProblem checks message_start, then one or more
// content_block_start / content_block_delta... / content_block_stop
// sequences, then message_delta and message_stop. ping may appear anywhere
// and unknown event types are ignored, as Anthropic documents.
func anthropicOrderProblem(events []string) string {
	var started, inBlock, delta, stopped bool
	blocks := 0
	for _, ev := range events {
		if !anthropicEventTypes[ev] {
			continue
		}
		if stopped {
			return fmt.Sprintf("%s after message_stop", ev)
		}
		if !started && ev != "message_start" {
			return fmt.Sprintf("%s before message_start", ev)
		}
		switch ev {
		case "message_start":
			if started {
				return "second message_start"
			}
			started = true
		case "content_block_start":
			switch {
			case delta:
				return "content_block_start after message_delta"
			case inBlock:
				return "content_block_start inside an open block"
			}
			inBlock = true
			blocks++
		case "content_block_delta", "content_block_stop":
			if !inBlock {
				return fmt.Sprintf("%s outside a content block", ev)
			}
			inBlock = ev == "content_block_delta"
		case "message_delta":
			switch {
			case inBlock:
				return "message_delta before content_block_stop"
			case blocks == 0:
				return "message_delta without any content block"
			}
			delta = true
		case "message_stop":
			if !delta {
				return "message_stop without message_delta"
			}
			stopped = true
		}
	}
	switch {
	case !started:
		return "no message_start"
	case inBlock:
		return "content block not closed"
	case !delta:
		return "no message_delta"
	case !stopped:
		return "no message_stop"
	}
	return ""
}

// openAIOrderProblem checks content chunks, usage in the final chunk when
// the stream reports usage at all, and a closing [DONE]. Usage repeated on
// earlier chunks is tolerated as long as the final chunk carries it.
func openAIOrderProblem(events []string) string {
	content, usage := false, false
	last := ""
	for i, ev := range events {
		if ev == "[DONE]" {
			if i != len(events)-1 {
				return "chunks after [DONE]"
			}
			continue
		}
		if ev == "chunk" || ev == "chunk+usage" {
			content = true
		}
		if ev == "usage" || ev == "chunk+usage" {
			usage = true
		}
		last = ev
	}
	switch {
	case !content:
		return "no content chunks"
	case usage && last == "chunk":
		return "usage did not arrive in the final chunk"
	case events[len(events)-1] != "[DONE]":
		return "the stream does not end with [DONE]"
	}
	return ""
}

// compactEvents joins events, collapsing repeats: "message_start,
// content_block_delta×12, ...".
func compactEvents(events []string) string {
	var parts []string
	for i := 0; i < len(events); {
		j := i
		for j < len(events) && events[j] == events[i] {
			j++
		}
		if j-i > 1 {
			parts = append(parts, fmt.Sprintf("%s×%d", events[i], j-i))
		} else {
			parts = append(parts, events[i])
		}
		i = j
	}
	return strings.Join(parts, ", ")
}
