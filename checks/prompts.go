package checks

// Probe names identify which fixed prompt a sample answered.
const (
	ProbeTiny   = "tiny"
	ProbeMedium = "medium"
)

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
