package fingerprint

import (
	"reflect"
	"strings"
	"testing"
)

func TestGenerateSameSeedSamePrompt(t *testing.T) {
	a, err := Generate(DefaultSpec(), 42)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := Generate(DefaultSpec(), 42)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same seed gave different prompts:\n%s\n---\n%s", a.Text, b.Text)
	}
	if a.Template != TemplateVersion || a.Seed != 42 {
		t.Fatalf("template %q seed %d, want %q and 42", a.Template, a.Seed, TemplateVersion)
	}
}

func TestGenerateDifferentSeedsDiffer(t *testing.T) {
	seen := map[string]uint64{}
	firsts := map[Category]bool{}
	numbersFirst := 0
	for seed := uint64(0); seed < 200; seed++ {
		p, err := Generate(DefaultSpec(), seed)
		if err != nil {
			t.Fatal(err)
		}
		if prev, dup := seen[p.Text]; dup {
			t.Fatalf("seeds %d and %d gave the same prompt", prev, seed)
		}
		seen[p.Text] = seed
		firsts[p.Categories[0]] = true
		if strings.Index(p.Text, ":`") < strings.Index(p.Text, "label: answer") {
			numbersFirst++
		}
	}
	if len(firsts) < 6 {
		t.Errorf("only %d categories ever came first", len(firsts))
	}
	if numbersFirst == 0 || numbersFirst == 200 {
		t.Errorf("the integer list came first in %d of 200 prompts", numbersFirst)
	}
}

func TestGenerateAsksForEverything(t *testing.T) {
	spec := DefaultSpec()
	p, err := Generate(spec, 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Categories) != len(spec.Categories) || p.Numbers != spec.Numbers {
		t.Fatalf("prompt asks for %v and %+v, want %v and %+v", p.Categories, p.Numbers, spec.Categories, spec.Numbers)
	}
	for _, c := range p.Categories {
		found := false
		for _, l := range categoryTemplates[c].labels {
			found = found || strings.Contains(p.Text, "- "+l+": ")
		}
		if !found {
			t.Errorf("no line for %s in\n%s", c, p.Text)
		}
	}
	for _, s := range []string{"200", "1", "355"} {
		if !strings.Contains(p.Text, s) {
			t.Errorf("prompt does not mention %s:\n%s", s, p.Text)
		}
	}
}

func TestGenerateIgnoresSpecOrderAndRegenerates(t *testing.T) {
	spec := DefaultSpec()
	reversed := DefaultSpec()
	for i, j := 0, len(reversed.Categories)-1; i < j; i, j = i+1, j-1 {
		reversed.Categories[i], reversed.Categories[j] = reversed.Categories[j], reversed.Categories[i]
	}
	a, _ := Generate(spec, 9)
	b, _ := Generate(reversed, 9)
	if a.Text != b.Text {
		t.Fatal("the order of spec categories changed the prompt")
	}
	c, err := Generate(a.Spec(), a.Seed)
	if err != nil || !reflect.DeepEqual(a, c) {
		t.Fatalf("Generate(p.Spec(), p.Seed) did not give p back (err %v)", err)
	}
}

func TestGenerateSections(t *testing.T) {
	words, err := Generate(Spec{Categories: []Category{Gem, Bird}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(words.Text, "numbers") || strings.Contains(words.Text, "integers") {
		t.Errorf("words-only prompt mentions the integer list:\n%s", words.Text)
	}
	numbers, err := Generate(Spec{Numbers: NumberSpec{Min: 0, Max: 9, Count: 30}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(numbers.Text, "label: answer") || len(numbers.Categories) != 0 {
		t.Errorf("numbers-only prompt asks for words:\n%s", numbers.Text)
	}
}

// The template text is part of every reference: a change here without a new
// TemplateVersion would silently compare different prompts.
func TestGenerateGolden(t *testing.T) {
	p, err := Generate(Spec{Categories: []Category{Colour, Letter, Gem}, Numbers: NumberSpec{Min: 1, Max: 355, Count: 200}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	const want = "Answer with the first thing that comes to mind. Do not explain or comment.\n\n" +
		"On a line starting with `numbers:`, write 200 whole numbers between 1 and 355, as they come to mind, separated by commas.\n\n" +
		"Reply with one line per item, formatted as `label: answer`. Keep each answer to one word or name.\n" +
		"- colour: any colour\n" +
		"- letter: a letter of the English alphabet\n" +
		"- gemstone: any gem"
	if p.Text != want {
		t.Fatalf("prompt for seed 1 changed; bump TemplateVersion and update this test:\n%s", p.Text)
	}
}

func TestGenerateRejectsBadSpecs(t *testing.T) {
	for name, spec := range map[string]Spec{
		"empty":            {},
		"unknown category": {Categories: []Category{"planet"}},
		"duplicate":        {Categories: []Category{Gem, Gem}},
		"negative count":   {Categories: []Category{Gem}, Numbers: NumberSpec{Min: 1, Max: 9, Count: -1}},
		"too many numbers": {Numbers: NumberSpec{Min: 1, Max: 9, Count: maxNumberCount + 1}},
		"empty range":      {Numbers: NumberSpec{Min: 5, Max: 5, Count: 10}},
		"inverted range":   {Numbers: NumberSpec{Min: 9, Max: 1, Count: 10}},
		"negative range":   {Numbers: NumberSpec{Min: -5, Max: 5, Count: 10}},
	} {
		if _, err := Generate(spec, 1); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
