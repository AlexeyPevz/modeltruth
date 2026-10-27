package checks

import (
	"fmt"
	"math/rand/v2"
)

// Probe names identify which prompt a sample answered.
const (
	ProbeTiny   = "tiny"
	ProbeMedium = "medium"
	// Task probes are named task_1, task_2, ... in the order of a run.
	probeTaskPrefix = "task_"
)

// TaskProbes is how many realistic task prompts one run sends.
const TaskProbes = 3

// answerWords are short words that tokenize to one or two tokens everywhere.
// Each run asks for a different one, so a relay cannot match a fixed probe
// string and special-case it; within a run every request is identical, which
// channel_consistency relies on.
var answerWords = []string{
	"ok", "yes", "done", "ready", "fine", "sure", "noted", "clear",
	"agreed", "right", "good", "true", "green", "north", "river", "stone",
}

func tinyPrompt(word string) string {
	return "Reply with the single word: " + word
}

// mediumStory is long enough that any real tokenizer reports well over len/6
// tokens for it, which lets UsageSanity detect under-reported usage.
const mediumStory = "A lighthouse keeper on a remote northern island kept a careful log of every ship that passed, the color of the evening sky, the direction of the wind, and the number of seabirds resting on the rocks below the tower. Over many years the notebooks filled a whole shelf. Visitors who came in summer liked to read the entries aloud, noticing how the handwriting changed with the seasons and how certain storms were described in a single calm line while quiet days received whole pages. When the light was finally automated, the notebooks were sent to a small museum on the mainland, where they still sit in a glass case near the door."

func mediumPrompt(word string) string {
	return mediumStory + " " + tinyPrompt(word)
}

// taskPrompts look like real work. Some relays serve a probe-like prompt
// clean and prepend their own system prompt only to requests that look like
// coding or reasoning work (seen on A6 channels in October 2026: "Reply with
// OK." billed as 13 tokens, a multiplication question as 4399). A run draws
// TaskProbes of them, so the set differs between runs.
var taskPrompts = []string{
	"What is 17*23? Answer with the number only.",
	"Write a Python function that returns the n-th Fibonacci number iteratively. Keep it under ten lines and do not explain it.",
	"My cat keeps scratching the sofa. Give me two short tips.",
	"Fix the bug in this JavaScript: `const total = items.reduce((a, b) => a + b.price);` Return only the corrected line.",
	"Write a SQL query that returns the ten most recent orders from a table named orders with a created_at column.",
	"Explain in one sentence what a race condition is.",
	"Translate to French: The meeting is moved to Thursday afternoon.",
	"A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show only the result.",
	"Refactor this Go snippet to return an error instead of panicking: `if err != nil { panic(err) }`",
	"Suggest a short name for a command line tool that compresses log files.",
	"Write a bash one-liner that counts the lines of every .go file in the current directory.",
	"What does HTTP status 429 mean? Answer in one short sentence.",
}

func taskProbe(i int) string {
	return fmt.Sprintf("%s%d", probeTaskPrefix, i+1)
}

func isTaskProbe(name string) bool {
	return len(name) > len(probeTaskPrefix) && name[:len(probeTaskPrefix)] == probeTaskPrefix
}

// drawTasks picks n distinct task prompts.
func drawTasks(n int) []string {
	n = min(n, len(taskPrompts))
	out := make([]string, 0, n)
	for _, i := range rand.Perm(len(taskPrompts))[:n] {
		out = append(out, taskPrompts[i])
	}
	return out
}
