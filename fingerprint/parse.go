package fingerprint

import (
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Features are what one answer to a Prompt says about the model that wrote
// it. Words are normalized for case and punctuation only: synonyms and
// spellings ("USA" and "United States", "grey" and "gray") stay distinct,
// since which one a model prefers is part of its fingerprint.
type Features struct {
	Template string              `json:"template"`
	Seed     uint64              `json:"seed"`
	Words    map[Category]string `json:"words,omitempty"`
	// Numbers are the integers in range, in the order written.
	Numbers      []int `json:"numbers,omitempty"`
	WordsAsked   int   `json:"words_asked"`
	NumbersAsked int   `json:"numbers_asked"`
	// Rejected counts answers that were given but cannot be used: a word
	// that is not a single word or name, a category answered twice, an
	// integer out of range, a negative or fractional number.
	Rejected int `json:"rejected"`
}

// Recognized is how many features the answer yielded: one per word and one
// per integer.
func (f Features) Recognized() int {
	return len(f.Words) + len(f.Numbers)
}

// Limits on what a parse reads, so any input is handled in bounded time and
// memory.
const (
	maxAnswerBytes = 1 << 20
	maxNumbers     = 4 * maxNumberCount
	maxValueRunes  = 48
	maxValueWords  = 5
)

// listMarker is a bullet or enumeration at the start of a line.
var listMarker = regexp.MustCompile(`^(?:[-*•+]|\d{1,4}[.)])\s+`)

// explanation starts the part of an answer that comments on it: "Kyoto
// (Japan)", "Iceland - because...", "Paris, France".
var explanation = regexp.MustCompile(`\(| [-–—] |[,;]| because\b`)

// Parse reads the answer to p. It never fails: what it cannot read is left
// out, and Recognized, Rejected and the asked counts tell how complete the
// answer was. Each answer is expected on its own line as `label: answer`;
// list markers and Markdown emphasis around labels and values are ignored.
// The integer list may wrap over several lines and ends at the next label.
func Parse(p Prompt, answer string) Features {
	f := Features{
		Template:     p.Template,
		Seed:         p.Seed,
		Words:        map[Category]string{},
		WordsAsked:   len(p.Categories),
		NumbersAsked: max(p.Numbers.Count, 0),
	}
	labels := map[string]Category{}
	for _, c := range p.Categories {
		for _, l := range categoryTemplates[c].labels {
			labels[l] = c
		}
	}
	if len(answer) > maxAnswerBytes {
		answer = answer[:maxAnswerBytes]
	}
	answer = strings.ToValidUTF8(answer, "\uFFFD")

	// A prompt that asks only for integers needs no label to find them.
	inNumbers := len(p.Categories) == 0 && f.NumbersAsked > 0
	for _, line := range strings.Split(answer, "\n") {
		line = listMarker.ReplaceAllString(strings.TrimSpace(line), "")
		if key, value, ok := splitLabel(line); ok {
			if c, ok := labels[key]; ok {
				inNumbers = false
				if _, dup := f.Words[c]; dup {
					f.Rejected++
				} else if w, ok := normalizeWord(c, value); ok {
					f.Words[c] = w
				} else {
					f.Rejected++
				}
				continue
			}
			switch {
			case f.NumbersAsked > 0 && slices.Contains(numberLabels, key):
				inNumbers = true
				line = value
			case strings.ContainsFunc(key, unicode.IsLetter):
				// Another label ("Count: 2") is not part of the list.
				inNumbers = inNumbers && len(p.Categories) == 0
				continue
			}
		}
		if inNumbers {
			f.readNumbers(line, p.Numbers)
		}
	}
	return f
}

// splitLabel splits "label: value" and normalizes the label: "**Colour:**",
// "`color`" and "colour (a colour)" all give "colour".
func splitLabel(line string) (key, value string, ok bool) {
	i := strings.IndexAny(line, ":：")
	if i < 0 {
		return "", "", false
	}
	key, value = line[:i], line[i:]
	_, size := utf8.DecodeRuneInString(value)
	value = value[size:]
	if j := strings.IndexByte(key, '('); j >= 0 {
		key = key[:j]
	}
	key = strings.Join(strings.Fields(strings.ToLower(trimMarks(key))), " ")
	return key, value, key != ""
}

// trimMarks drops Markdown emphasis and surrounding punctuation, symbols and
// spaces.
func trimMarks(s string) string {
	s = strings.NewReplacer("*", "", "`", "", "_", "").Replace(s)
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r)
	})
}

// normalizeWord lowercases a value, removes punctuation around it and any
// comment after it, and checks that what is left can answer c.
func normalizeWord(c Category, value string) (string, bool) {
	value = strings.NewReplacer("*", "", "`", "", "_", "").Replace(value)
	if loc := explanation.FindStringIndex(value); loc != nil {
		value = value[:loc[0]]
	}
	words := strings.Fields(strings.ToLower(trimMarks(value)))
	w := strings.Join(words, " ")
	n := utf8.RuneCountInString(w)
	switch c {
	case Letter:
		r, _ := utf8.DecodeRuneInString(w)
		return w, n == 1 && unicode.IsLetter(r)
	case Digit:
		return w, n == 1 && w[0] >= '0' && w[0] <= '9'
	}
	if n == 0 || n > maxValueRunes || len(words) > maxValueWords || !strings.ContainsFunc(w, unicode.IsLetter) {
		return "", false
	}
	return w, true
}

func isDigit(b byte) bool { return b >= '0' && b <= '9' }

// readNumbers adds the integers written in s. Anything between them
// (commas, spaces, words) is a separator; negative and fractional numbers
// (-7, −7, 2.5, .5) are rejected.
func (f *Features) readNumbers(s string, spec NumberSpec) {
	for i := 0; i < len(s); {
		if !isDigit(s[i]) {
			i++
			continue
		}
		j := i
		for j < len(s) && isDigit(s[j]) {
			j++
		}
		// A minus sign right before the digits, and not between two numbers
		// as in "10-20", makes the number negative.
		before, size := utf8.DecodeLastRuneInString(s[:i])
		negative := strings.ContainsRune("-−", before) && (i == size || !isDigit(s[i-size-1]))
		fraction := (i > 0 && s[i-1] == '.') || (j+1 < len(s) && s[j] == '.' && isDigit(s[j+1]))
		digits := s[i:j]
		if fraction {
			for j++; j < len(s) && isDigit(s[j]); j++ {
			}
		}
		i = j
		n, err := strconv.Atoi(digits)
		if negative || fraction || err != nil || n < spec.Min || n > spec.Max || len(f.Numbers) >= maxNumbers {
			f.Rejected++
			continue
		}
		f.Numbers = append(f.Numbers, n)
	}
}
