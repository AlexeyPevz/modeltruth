package fingerprint

import (
	"fmt"
	"math/rand/v2"
	"reflect"
	"testing"
)

// synthTier is a synthetic model: for each category it picks word fav with
// probability pWord and otherwise any of ten words; it picks an integer from
// [band, band+59] with probability pNumber and otherwise any from 1 to 355.
type synthTier struct {
	fav     int
	pWord   float64
	band    int
	pNumber float64
}

func (s synthTier) sample(r *rand.Rand, seed uint64) Features {
	f := Features{Template: TemplateVersion, Seed: seed, Words: map[Category]string{}, WordsAsked: len(allCategories), NumbersAsked: 200}
	for _, c := range allCategories {
		k := s.fav
		if r.Float64() >= s.pWord {
			k = r.IntN(10)
		}
		f.Words[c] = fmt.Sprintf("%s-%d", c, k)
	}
	for range f.NumbersAsked {
		if r.Float64() < s.pNumber {
			f.Numbers = append(f.Numbers, s.band+r.IntN(60))
		} else {
			f.Numbers = append(f.Numbers, 1+r.IntN(355))
		}
	}
	return f
}

func synthBank(r *rand.Rand, samples int, tiers map[string]synthTier, names ...string) []Reference {
	var bank []Reference
	for _, name := range names {
		ref := Reference{Model: name}
		for i := range samples {
			ref.Samples = append(ref.Samples, tiers[name].sample(r, uint64(i)))
		}
		bank = append(bank, ref)
	}
	return bank
}

var separable = map[string]synthTier{
	"opus":   {fav: 0, pWord: 0.6, band: 1, pNumber: 0.4},
	"sonnet": {fav: 1, pWord: 0.6, band: 101, pNumber: 0.4},
	"haiku":  {fav: 2, pWord: 0.6, band: 201, pNumber: 0.4},
}

func TestClassifyThreeSeparableTiers(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	names := []string{"opus", "sonnet", "haiku"}
	bank := synthBank(r, 8, separable, names...)
	for _, kinds := range []FeatureKind{AllFeatures, WordFeatures, NumberFeatures} {
		decisions := map[Decision]int{}
		for _, served := range names {
			for i := range 20 {
				probe := separable[served].sample(r, uint64(100+i))
				if kinds&WordFeatures == 0 {
					probe.Words, probe.WordsAsked = nil, 0
				}
				if kinds&NumberFeatures == 0 {
					probe.Numbers, probe.NumbersAsked = nil, 0
				}
				for _, claimed := range names {
					c := Classify(bank, claimed, probe)
					if len(c.Scores) != len(names) {
						t.Fatalf("scored %d models, want every model of the bank: %+v", len(c.Scores), c)
					}
					switch {
					case claimed == served && c.Decision == Fail:
						t.Fatalf("kinds %d: %s served as itself failed: %s", kinds, served, c.Reason)
					case claimed != served && c.Decision == Pass:
						t.Fatalf("kinds %d: %s served as %s passed: %s", kinds, served, claimed, c.Reason)
					case c.Decision == Fail && c.Scores[0].Model != served:
						t.Fatalf("kinds %d: fail names %s, served %s", kinds, c.Scores[0].Model, served)
					case c.Decision != Skip && (c.Scores[0].Posterior < MinPosterior || c.Margin < MinLogMargin || c.PairAccuracy < MinPairAccuracy):
						t.Fatalf("kinds %d: decided %s below the thresholds: %+v", kinds, c.Decision, c)
					}
					if claimed == served {
						decisions[c.Decision]++
					} else if c.Decision == Fail {
						decisions["caught"]++
					}
				}
			}
		}
		// 60 probes served as themselves, 120 served under another name.
		if decisions[Pass] < 54 || decisions["caught"] < 108 {
			t.Errorf("kinds %d: %d of 60 passed, %d of 120 substitutions caught", kinds, decisions[Pass], decisions["caught"])
		}
	}
}

func TestClassifyAbstainsOnInseparableModels(t *testing.T) {
	same := synthTier{fav: 3, pWord: 0.5, band: 150, pNumber: 0.3}
	tiers := map[string]synthTier{"sonnet-5": same, "sonnet-5.5": same, "opus": separable["opus"]}
	for seed := range uint64(5) {
		r := rand.New(rand.NewPCG(seed, 7))
		for _, names := range [][]string{{"sonnet-5", "sonnet-5.5"}, {"sonnet-5", "sonnet-5.5", "opus"}} {
			bank := synthBank(r, 8, tiers, names...)
			for i := range 40 {
				probe := same.sample(r, uint64(100+i))
				for _, claimed := range names {
					c := Classify(bank, claimed, probe)
					// Opus is told apart from both sonnets, so claiming it
					// may fail; the sonnets are only ever skipped.
					if c.Decision == Pass || (c.Decision == Fail && claimed != "opus") {
						t.Fatalf("bank %v, probe from a sonnet claimed %s: %s (%s)", names, claimed, c.Decision, c.Reason)
					}
				}
			}
		}
	}
}

func TestClassifyPassesNextToInseparablePair(t *testing.T) {
	same := synthTier{fav: 3, pWord: 0.5, band: 150, pNumber: 0.3}
	tiers := map[string]synthTier{"sonnet-5": same, "sonnet-5.5": same, "opus": separable["opus"]}
	r := rand.New(rand.NewPCG(3, 4))
	bank := synthBank(r, 8, tiers, "sonnet-5", "sonnet-5.5", "opus")
	passed := 0
	for i := range 20 {
		c := Classify(bank, "opus", separable["opus"].sample(r, uint64(100+i)))
		if c.Decision == Fail {
			t.Fatalf("opus served as itself failed: %s", c.Reason)
		}
		if c.Decision == Pass {
			passed++
		}
	}
	if passed < 18 {
		t.Errorf("%d of 20 opus probes passed", passed)
	}
}

func TestClassifySkips(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	bank := synthBank(r, 8, separable, "opus", "sonnet", "haiku")
	probe := separable["sonnet"].sample(r, 100)
	few := synthBank(r, MinReferenceSamples-1, separable, "opus")
	sparse := probe
	sparse.Words = map[Category]string{Country: probe.Words[Country], City: probe.Words[City]}
	sparse.Numbers = probe.Numbers[:20]
	otherTemplate := probe
	otherTemplate.Template = "fp0"
	for _, tc := range []struct {
		name    string
		bank    []Reference
		claimed string
		probe   Features
	}{
		{"empty bank", nil, "opus", probe},
		{"bank without samples", []Reference{{Model: "opus"}, {Model: "sonnet"}}, "opus", probe},
		{"claimed not in bank", bank, "opus-6", probe},
		{"claimed not in bank, probe matches a bank model", bank[1:], "opus", probe},
		{"claimed has too few references", append(bank[1:], few...), "opus", probe},
		{"only the claimed model", bank[:1], "opus", probe},
		{"too few features", bank, "opus", sparse},
		{"nothing recognized", bank, "opus", Features{Template: TemplateVersion, WordsAsked: 12, NumbersAsked: 200}},
		{"other template version", bank, "opus", otherTemplate},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := Classify(tc.bank, tc.claimed, tc.probe)
			if c.Decision != Skip || c.Reason == "" {
				t.Fatalf("got %s (%q), want skip with a reason", c.Decision, c.Reason)
			}
		})
	}
}

func TestClassifyClaimedNotInBankAlwaysSkips(t *testing.T) {
	r := rand.New(rand.NewPCG(8, 9))
	bank := synthBank(r, 8, separable, "sonnet", "haiku")
	for i := range 50 {
		for _, served := range []string{"opus", "sonnet", "haiku"} {
			if c := Classify(bank, "opus", separable[served].sample(r, uint64(i))); c.Decision != Skip {
				t.Fatalf("opus is not in the bank, %s probe gave %s: %s", served, c.Decision, c.Reason)
			}
		}
	}
}

func TestClassifyExcludesModelsWithFewReferences(t *testing.T) {
	r := rand.New(rand.NewPCG(10, 11))
	bank := append(synthBank(r, 8, separable, "opus", "sonnet"), synthBank(r, 2, separable, "haiku")...)
	c := Classify(bank, "opus", separable["opus"].sample(r, 100))
	if !reflect.DeepEqual(c.Excluded, []string{"haiku"}) || len(c.Scores) != 2 {
		t.Fatalf("excluded %v, scored %d", c.Excluded, len(c.Scores))
	}
}

func TestClassifyIgnoresOtherTemplateVersions(t *testing.T) {
	r := rand.New(rand.NewPCG(12, 13))
	bank := synthBank(r, 8, separable, "opus", "sonnet")
	// References of an older template for haiku, taken from opus answers:
	// were they compared, opus probes would split between opus and haiku.
	old := synthBank(r, 8, separable, "opus")[0]
	old.Model = "haiku"
	for i := range old.Samples {
		old.Samples[i].Template = "fp0"
	}
	c := Classify(append(bank, old), "opus", separable["opus"].sample(r, 100))
	if c.Decision != Pass || len(c.Scores) != 2 {
		t.Fatalf("got %s with %d scores: %s", c.Decision, len(c.Scores), c.Reason)
	}
}

func TestClassifyIsDeterministic(t *testing.T) {
	r := rand.New(rand.NewPCG(14, 15))
	bank := synthBank(r, 8, separable, "opus", "sonnet", "haiku")
	probe := separable["haiku"].sample(r, 100)
	first := Classify(bank, "opus", probe)
	for range 20 {
		if c := Classify(bank, "opus", probe); !reflect.DeepEqual(c, first) {
			t.Fatalf("results differ:\n%+v\n%+v", first, c)
		}
	}
}

func TestLeaveOneOut(t *testing.T) {
	r := rand.New(rand.NewPCG(16, 17))
	bank := synthBank(r, 8, separable, "opus", "sonnet", "haiku")
	for _, kinds := range []FeatureKind{AllFeatures, WordFeatures, NumberFeatures} {
		acc := LeaveOneOut(bank, kinds)
		if acc.Total != 24 || acc.Correct < 23 {
			t.Errorf("kinds %d: %d of %d correct, confusion %v", kinds, acc.Correct, acc.Total, acc.Confusion)
		}
	}

	same := synthTier{fav: 3, pWord: 0.5, band: 150, pNumber: 0.3}
	bank = synthBank(r, 10, map[string]synthTier{"a": same, "b": same}, "a", "b")
	acc := LeaveOneOut(bank, AllFeatures)
	sum := 0
	for _, row := range acc.Confusion {
		for _, n := range row {
			sum += n
		}
	}
	if acc.Total != 20 || sum != 20 || acc.Correct > 15 {
		t.Errorf("inseparable: %d of %d correct, confusion %v", acc.Correct, acc.Total, acc.Confusion)
	}

	if acc := LeaveOneOut(nil, AllFeatures); acc.Total != 0 {
		t.Errorf("empty bank: %+v", acc)
	}
	if acc := LeaveOneOut(bank[:1], AllFeatures); acc.Total != 0 {
		t.Errorf("one model: %+v", acc)
	}
}
