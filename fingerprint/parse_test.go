package fingerprint

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

func mustGenerate(t testing.TB, spec Spec, seed uint64) Prompt {
	t.Helper()
	p, err := Generate(spec, seed)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParseFullAnswer(t *testing.T) {
	p := mustGenerate(t, DefaultSpec(), 3)
	answer := `country: Iceland
animal: Otter
colour: Teal
fruit: Mango
city: Kyoto
bird: Heron
gem: Opal
element: Iron
month: March
weekday: Friday
letter: K
digit: 7
numbers: 17, 42, 7, 288, 355, 1, 199`
	f := Parse(p, answer)
	want := map[Category]string{
		Country: "iceland", Animal: "otter", Colour: "teal", Fruit: "mango", City: "kyoto", Bird: "heron",
		Gem: "opal", Element: "iron", Month: "march", Weekday: "friday", Letter: "k", Digit: "7",
	}
	if !reflect.DeepEqual(f.Words, want) {
		t.Errorf("words %v, want %v", f.Words, want)
	}
	if !reflect.DeepEqual(f.Numbers, []int{17, 42, 7, 288, 355, 1, 199}) {
		t.Errorf("numbers %v", f.Numbers)
	}
	if f.Template != TemplateVersion || f.Seed != 3 || f.WordsAsked != 12 || f.NumbersAsked != 200 {
		t.Errorf("template %q seed %d asked %d/%d", f.Template, f.Seed, f.WordsAsked, f.NumbersAsked)
	}
	if f.Recognized() != 19 || f.Rejected != 0 {
		t.Errorf("recognized %d rejected %d, want 19 and 0", f.Recognized(), f.Rejected)
	}
}

func TestParseSyntheticAnswers(t *testing.T) {
	spec := Spec{Categories: []Category{Country, Colour, City, Letter, Digit, Weekday}, Numbers: NumberSpec{Min: 1, Max: 100, Count: 10}}
	p := mustGenerate(t, spec, 5)
	for _, tc := range []struct {
		name     string
		answer   string
		words    map[Category]string
		numbers  []int
		rejected int
	}{
		{
			name:   "markdown and list markers",
			answer: "Sure! Here are my picks:\n\n1. **Country:** *Japan*\n2. `colour`: \"Blue\".\n- **City**: Paris!\n* letter: `Q`\n• Digit:  3\n+ day: Monday",
			words:  map[Category]string{Country: "japan", Colour: "blue", City: "paris", Letter: "q", Digit: "3", Weekday: "monday"},
		},
		{
			name:   "comments after the answer",
			answer: "country: Iceland (land of fire and ice)\ncity: Kyoto, Japan\ncolour: teal - calming\nweekday: Friday because weekend\nletter: Z;",
			words:  map[Category]string{Country: "iceland", City: "kyoto", Colour: "teal", Weekday: "friday", Letter: "z"},
		},
		{
			name:   "synonyms and spellings stay distinct",
			answer: "country: United States\ncolour: GRAY\ncity: New   York City\nweekday: Fri",
			words:  map[Category]string{Country: "united states", Colour: "gray", City: "new york city", Weekday: "fri"},
		},
		{
			name:   "label repeating the phrase, fullwidth colon",
			answer: "colour (a colour)： Crimson",
			words:  map[Category]string{Colour: "crimson"},
		},
		{
			name:     "invalid letter, digit and word values",
			answer:   "letter: AB\ndigit: seven\ncountry: 42\ncity: I would rather not pick a city right now\ncolour: !!!",
			words:    map[Category]string{},
			rejected: 5,
		},
		{
			name:     "a category answered twice keeps the first",
			answer:   "colour: red\ncolour: blue",
			words:    map[Category]string{Colour: "red"},
			rejected: 1,
		},
		{
			name:    "incomplete answer",
			answer:  "country: Chile\ncolour:",
			words:   map[Category]string{Country: "chile"},
			numbers: nil, rejected: 1,
		},
		{
			name:     "numbers wrapped over lines, then words",
			answer:   "numbers: 3, 14, 15,\n92, 65\n35 89\ncolour: green\n77",
			words:    map[Category]string{Colour: "green"},
			numbers:  []int{3, 14, 15, 92, 65, 35, 89},
			rejected: 0,
		},
		{
			name:     "out of range, negative, fractional and huge numbers",
			answer:   "integers: 0, 101, -5, 2.5, 99999999999999999999999, 50, 10-20, 7., .5, −7, -3",
			words:    map[Category]string{},
			numbers:  []int{50, 10, 20, 7},
			rejected: 8,
		},
		{
			name:     "numbers enumerated one per line",
			answer:   "Numbers:\n1. 64\n2. 12\n3) 100",
			words:    map[Category]string{},
			numbers:  []int{64, 12, 100},
			rejected: 0,
		},
		{
			name:     "another label ends the list",
			answer:   "numbers: 5, 6\nCount: 2\n7, 8\nNote: 9\ncolour: red",
			words:    map[Category]string{Colour: "red"},
			numbers:  []int{5, 6},
			rejected: 0,
		},
		{
			name:   "unlabeled numbers are not read when words were asked",
			answer: "17, 23, 42\nI cannot help with that.",
			words:  map[Category]string{},
		},
		{
			name:   "unknown labels and prose",
			answer: "planet: Mars\nanimal: otter\nHere you go: my answers below\n:::\n: :",
			words:  map[Category]string{},
		},
		{
			name:   "empty",
			answer: "",
			words:  map[Category]string{},
		},
		{
			name:    "windows line endings",
			answer:  "country: Chad\r\nnumbers: 1, 2\r\n",
			words:   map[Category]string{Country: "chad"},
			numbers: []int{1, 2},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := Parse(p, tc.answer)
			if !reflect.DeepEqual(f.Words, tc.words) {
				t.Errorf("words %v, want %v", f.Words, tc.words)
			}
			if !reflect.DeepEqual(f.Numbers, tc.numbers) {
				t.Errorf("numbers %v, want %v", f.Numbers, tc.numbers)
			}
			if f.Rejected != tc.rejected {
				t.Errorf("rejected %d, want %d", f.Rejected, tc.rejected)
			}
			if f.Recognized() != len(tc.words)+len(tc.numbers) {
				t.Errorf("recognized %d", f.Recognized())
			}
		})
	}
}

func TestParseNumbersOnlyPromptNeedsNoLabel(t *testing.T) {
	p := mustGenerate(t, Spec{Numbers: NumberSpec{Min: 1, Max: 355, Count: 5}}, 2)
	f := Parse(p, "Here you go:\n12 300 7\nCount: 3\n355, 400\ncountry: France 2")
	if !reflect.DeepEqual(f.Numbers, []int{12, 300, 7, 355}) || f.Rejected != 1 || len(f.Words) != 0 {
		t.Fatalf("numbers %v rejected %d words %v", f.Numbers, f.Rejected, f.Words)
	}
}

func TestParseOnlyAskedCategories(t *testing.T) {
	p := mustGenerate(t, Spec{Categories: []Category{Gem}}, 2)
	f := Parse(p, "gem: Ruby\ncountry: France\nnumbers: 1, 2, 3")
	if !reflect.DeepEqual(f.Words, map[Category]string{Gem: "ruby"}) || f.Numbers != nil {
		t.Fatalf("words %v numbers %v", f.Words, f.Numbers)
	}
}

func TestParseAcceptsEveryLabel(t *testing.T) {
	p := mustGenerate(t, DefaultSpec(), 11)
	for c, tpl := range categoryTemplates {
		value := map[Category]string{Letter: "x", Digit: "4"}[c]
		if value == "" {
			value = "word"
		}
		for _, l := range tpl.labels {
			if f := Parse(p, l+": "+value); f.Words[c] != value {
				t.Errorf("label %q not read as %s", l, c)
			}
		}
	}
}

func TestParseBoundsInput(t *testing.T) {
	p := mustGenerate(t, Spec{Numbers: NumberSpec{Min: 1, Max: 9, Count: 10}}, 1)
	f := Parse(p, "numbers: "+strings.Repeat("5 ", maxAnswerBytes))
	if len(f.Numbers) != maxNumbers {
		t.Fatalf("kept %d numbers, want the cap %d", len(f.Numbers), maxNumbers)
	}
}

func FuzzParse(f *testing.F) {
	for _, s := range []string{
		"", "country: Iceland\nnumbers: 1, 2, 3", "**Colour:** Teal", "letter: \xff", "numbers: -1.5.5--3",
		"digit: ٣", "city: (", "：：：", "1. 2. 3.", "numbers:\n" + strings.Repeat("9", 40),
	} {
		f.Add(s)
	}
	p, err := Generate(DefaultSpec(), 1)
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, answer string) {
		got := Parse(p, answer)
		if len(got.Words) > got.WordsAsked || got.Recognized() != len(got.Words)+len(got.Numbers) || got.Rejected < 0 {
			t.Fatalf("inconsistent features %+v", got)
		}
		for c, w := range got.Words {
			if !slices.Contains(p.Categories, c) || w == "" {
				t.Fatalf("bad word %s=%q", c, w)
			}
		}
		for _, n := range got.Numbers {
			if n < p.Numbers.Min || n > p.Numbers.Max {
				t.Fatalf("number %d out of range", n)
			}
		}
	})
}
