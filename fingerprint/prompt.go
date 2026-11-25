// Package fingerprint builds the prompts of the tier layer, turns answers
// into features and scores them against reference answers. A prompt asks for "first-instinct" choices: one word for each
// of several categories and a list of integers. Wording and order are drawn
// from templates with a recorded seed, so a relay cannot match one fixed
// string, while a reference and a probe made from the same template version
// and seed are the same request.
package fingerprint

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
)

// TemplateVersion identifies the phrase lists below. Any change to them, or
// to how a seed picks from them, changes the prompts and must bump it:
// references are only comparable with probes of the same version.
const TemplateVersion = "fp1"

// Category is one word choice asked for in a prompt.
type Category string

// The categories of the 2026-10-26 pilot (docs/methodology.md).
const (
	Country Category = "country"
	Animal  Category = "animal"
	Colour  Category = "colour"
	Fruit   Category = "fruit"
	City    Category = "city"
	Bird    Category = "bird"
	Gem     Category = "gem"
	Element Category = "element"
	Month   Category = "month"
	Weekday Category = "weekday"
	Letter  Category = "letter"
	Digit   Category = "digit"
)

type categoryTemplate struct {
	// labels are the keys the answer line may start with; the prompt uses one
	// drawn per seed, the parser accepts all of them.
	labels []string
	// phrases must all admit the same answers: the seed changes the wording,
	// never which answers fit.
	phrases []string
}

var categoryTemplates = map[Category]categoryTemplate{
	Country: {[]string{"country"}, []string{"a country", "any country", "a country of the world"}},
	Animal:  {[]string{"animal"}, []string{"an animal", "any animal", "the name of an animal"}},
	Colour:  {[]string{"colour", "color"}, []string{"a colour", "any colour", "a single colour"}},
	Fruit:   {[]string{"fruit"}, []string{"a fruit", "any fruit", "a kind of fruit"}},
	City:    {[]string{"city"}, []string{"a city", "any city", "a city anywhere in the world"}},
	Bird:    {[]string{"bird"}, []string{"a bird", "any bird", "a kind of bird"}},
	Gem:     {[]string{"gem", "gemstone"}, []string{"a gemstone", "any gem", "a precious or semi-precious stone"}},
	Element: {[]string{"element"}, []string{"a chemical element", "any element of the periodic table", "an element from the periodic table"}},
	Month:   {[]string{"month"}, []string{"a month", "a month of the year", "any month"}},
	Weekday: {[]string{"weekday", "day"}, []string{"a day of the week", "any day of the week", "a day from Monday to Sunday"}},
	Letter:  {[]string{"letter"}, []string{"a letter of the English alphabet", "a single letter from A to Z", "any letter from A to Z"}},
	Digit:   {[]string{"digit"}, []string{"a digit from 0 to 9", "a single digit", "any digit"}},
}

// allCategories is the default order before shuffling.
var allCategories = []Category{Country, Animal, Colour, Fruit, City, Bird, Gem, Element, Month, Weekday, Letter, Digit}

// Categories returns every known category.
func Categories() []Category {
	return append([]Category(nil), allCategories...)
}

// NumberSpec describes the integer list: Count integers from Min to Max. A
// zero Count leaves the list out of the prompt.
type NumberSpec struct {
	Min   int `json:"min"`
	Max   int `json:"max"`
	Count int `json:"count"`
}

// maxNumberCount bounds the list so one answer stays within ordinary output
// limits.
const maxNumberCount = 1000

// Spec is what a prompt asks for. The seed varies only wording and order;
// everything in Spec is fixed, so it is part of what a reference records. The
// order of Categories does not matter.
type Spec struct {
	Categories []Category `json:"categories"`
	Numbers    NumberSpec `json:"numbers"`
}

// DefaultSpec asks for all 12 pilot categories and, following ModelTrace's
// design, 200 integers from 1 to 355: the pilot's 80 integers from 1 to 100
// were too uniform to separate tiers.
func DefaultSpec() Spec {
	return Spec{Categories: Categories(), Numbers: NumberSpec{Min: 1, Max: 355, Count: 200}}
}

func (s Spec) validate() error {
	seen := map[Category]bool{}
	for _, c := range s.Categories {
		if _, ok := categoryTemplates[c]; !ok {
			return fmt.Errorf("unknown category %q", c)
		}
		if seen[c] {
			return fmt.Errorf("category %q listed twice", c)
		}
		seen[c] = true
	}
	n := s.Numbers
	switch {
	case n.Count < 0 || n.Count > maxNumberCount:
		return fmt.Errorf("number count %d is outside 0..%d", n.Count, maxNumberCount)
	case n.Count > 0 && (n.Min < 0 || n.Min >= n.Max):
		return fmt.Errorf("number range %d..%d must be non-negative and not empty", n.Min, n.Max)
	case len(s.Categories) == 0 && n.Count == 0:
		return errors.New("spec asks for nothing")
	}
	return nil
}

// Prompt is one generated fingerprint prompt with everything needed to
// regenerate it and to parse its answer.
type Prompt struct {
	Template string `json:"template"`
	Seed     uint64 `json:"seed"`
	// Categories are in the order the prompt lists them.
	Categories []Category `json:"categories"`
	Numbers    NumberSpec `json:"numbers"`
	Text       string     `json:"text"`
}

// Spec returns what the prompt asks for: Generate(p.Spec(), p.Seed) gives p
// again under the same TemplateVersion.
func (p Prompt) Spec() Spec {
	return Spec{Categories: append([]Category(nil), p.Categories...), Numbers: p.Numbers}
}

var intros = []string{
	"Answer with the first thing that comes to mind. Do not explain or comment.",
	"Go with your first instinct for every item below. No explanations.",
	"Without thinking it over, give your first choice for each item. Do not add any commentary.",
	"Quick association test: reply with whatever comes to mind first, nothing else.",
}

var wordInstructions = []string{
	"For each item, write one line in the form `label: answer`, with a single word or name as the answer:",
	"Reply with one line per item, formatted as `label: answer`. Keep each answer to one word or name.",
	"Give one answer per line as `label: answer` (one word or name each):",
}

// numberLabels are the keys of the integer list line.
var numberLabels = []string{"numbers", "integers"}

var numberInstructions = []string{
	"On a line starting with `%[1]s:`, write %[2]d whole numbers between %[3]d and %[4]d, as they come to mind, separated by commas.",
	"Add a line `%[1]s:` followed by %[2]d integers from %[3]d to %[4]d, the first ones you think of, comma-separated.",
	"Write a line beginning with `%[1]s:` and list %[2]d integers in the range %[3]d-%[4]d that pop into your head, separated by commas.",
}

// seedStream separates fingerprint draws from any other use of the same seed.
const seedStream = 0x6670_315f_7465_6d70

// draw picks from template lists. It uses the PCG source directly, whose
// output Go specifies, so a seed gives the same prompt on every Go version.
type draw struct{ src *rand.PCG }

func (d draw) intn(n int) int { return int(d.src.Uint64() % uint64(n)) }

func (d draw) pick(list []string) string { return list[d.intn(len(list))] }

// Generate builds the prompt for spec and seed. The same spec and seed always
// give the same text; Prompt records both, plus TemplateVersion.
func Generate(spec Spec, seed uint64) (Prompt, error) {
	if err := spec.validate(); err != nil {
		return Prompt{}, err
	}
	d := draw{rand.NewPCG(seed, seedStream)}
	// Shuffle from the canonical order, so the order of spec.Categories does
	// not change the prompt and Prompt.Spec regenerates it.
	var cats []Category
	for _, c := range allCategories {
		if slices.Contains(spec.Categories, c) {
			cats = append(cats, c)
		}
	}
	for i := len(cats) - 1; i > 0; i-- {
		j := d.intn(i + 1)
		cats[i], cats[j] = cats[j], cats[i]
	}

	var words, numbers string
	if len(cats) > 0 {
		lines := []string{d.pick(wordInstructions)}
		for _, c := range cats {
			t := categoryTemplates[c]
			lines = append(lines, fmt.Sprintf("- %s: %s", d.pick(t.labels), d.pick(t.phrases)))
		}
		words = strings.Join(lines, "\n")
	}
	if n := spec.Numbers; n.Count > 0 {
		numbers = fmt.Sprintf(d.pick(numberInstructions), d.pick(numberLabels), n.Count, n.Min, n.Max)
	}
	parts := []string{d.pick(intros)}
	switch {
	case words == "":
		parts = append(parts, numbers)
	case numbers == "":
		parts = append(parts, words)
	case d.intn(2) == 0:
		parts = append(parts, words, numbers)
	default:
		parts = append(parts, numbers, words)
	}
	return Prompt{
		Template:   TemplateVersion,
		Seed:       seed,
		Categories: cats,
		Numbers:    spec.Numbers,
		Text:       strings.Join(parts, "\n\n"),
	}, nil
}
