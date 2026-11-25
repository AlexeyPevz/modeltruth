package fingerprint

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// Thresholds of the tier decision (docs/methodology.md, "Layer 4: tier
// classifier"). They are deliberately strict: a wrong fail accuses a relay,
// a wrong skip only withholds a verdict.
const (
	// MinPosterior is the posterior the winning model needs before the
	// classifier decides anything.
	MinPosterior = 0.85
	// MinLogMargin is the log-likelihood lead, in nats, of the winner over the
	// runner-up: ln 10, a Bayes factor of 10 ("strong" on Jeffreys' scale).
	MinLogMargin = 2.302585092994046
	// MinPairAccuracy is the leave-one-out accuracy the bank must reach on the
	// two models a decision is about (winner and claimed model, or winner and
	// runner-up for a pass). Models the bank does not tell apart are a close
	// call however confident a single probe looks.
	MinPairAccuracy = 0.9
	// MinReferenceSamples is how many references a model needs to be a
	// candidate.
	MinReferenceSamples = 5
	// MinFeatureShare is the share of the asked words, or of the asked
	// integers, an answer must yield for that part to be scored.
	MinFeatureShare = 0.5
	// NumberTemperature turns the Hellinger distance between an answer's
	// integers and a model's reference integers into a log-score: -H / T.
	NumberTemperature = 0.025
	// laplaceAlpha is the pseudo-count of Laplace smoothing.
	laplaceAlpha = 1.0
)

// Reference is a model's reference answers, all taken in one request
// environment.
type Reference struct {
	Model   string     `json:"model"`
	Samples []Features `json:"samples"`
}

// Decision is the outcome of a tier classification. The values match the
// check statuses of the report.
type Decision string

const (
	Pass Decision = "pass"
	Fail Decision = "fail"
	Skip Decision = "skip"
)

// Score is one model's result for a probe.
type Score struct {
	Model string `json:"model"`
	// LogScore is the log-likelihood of the probe under the model, up to a
	// constant shared by all models.
	LogScore  float64 `json:"log_score"`
	Posterior float64 `json:"posterior"`
}

// Classification is the result of Classify.
type Classification struct {
	Decision Decision `json:"decision"`
	Reason   string   `json:"reason"`
	Claimed  string   `json:"claimed"`
	// Scores cover every candidate, best first. Empty when the classifier
	// stopped before scoring.
	Scores []Score `json:"scores,omitempty"`
	// Margin is LogScore of the first minus that of the second candidate.
	Margin float64 `json:"margin,omitempty"`
	// PairAccuracy is the leave-one-out accuracy of the bank on the pair the
	// decision is about, when it was computed.
	PairAccuracy float64 `json:"pair_accuracy,omitempty"`
	// Excluded are bank models with fewer than MinReferenceSamples usable
	// references; they are not candidates.
	Excluded []string `json:"excluded,omitempty"`
}

// FeatureKind selects the parts of an answer that are scored.
type FeatureKind int

const (
	WordFeatures FeatureKind = 1 << iota
	NumberFeatures
	AllFeatures = WordFeatures | NumberFeatures
)

// profile counts a model's reference answers.
type profile struct {
	model    string
	samples  []Features
	words    map[Category]map[string]int
	answered map[Category]int
	numbers  map[int]int
	total    int
}

func newProfile(model string) *profile {
	return &profile{model: model, words: map[Category]map[string]int{}, answered: map[Category]int{}, numbers: map[int]int{}}
}

// add adds f to the counts, or removes it when sign is -1.
func (p *profile) add(f Features, sign int) {
	for c, w := range f.Words {
		if p.words[c] == nil {
			p.words[c] = map[string]int{}
		}
		p.words[c][w] += sign
		p.answered[c] += sign
	}
	for _, n := range f.Numbers {
		p.numbers[n] += sign
	}
	p.total += sign * len(f.Numbers)
}

// profiles groups the samples of bank made from template by model, in order
// of first appearance. Samples of other template versions are not comparable
// and are left out.
func profiles(bank []Reference, template string) []*profile {
	var out []*profile
	byModel := map[string]*profile{}
	for _, r := range bank {
		for _, f := range r.Samples {
			if f.Template != template {
				continue
			}
			p := byModel[r.Model]
			if p == nil {
				p = newProfile(r.Model)
				byModel[r.Model] = p
				out = append(out, p)
			}
			p.samples = append(p.samples, f)
			p.add(f, 1)
		}
	}
	return out
}

// scorer holds what all candidates share: the vocabulary of each category and
// the integers seen anywhere, so every model is smoothed over the same
// outcomes.
type scorer struct {
	vocab   map[Category]map[string]bool
	support map[int]bool
}

func newScorer(ps []*profile) scorer {
	s := scorer{vocab: map[Category]map[string]bool{}, support: map[int]bool{}}
	for _, p := range ps {
		for c, ws := range p.words {
			if s.vocab[c] == nil {
				s.vocab[c] = map[string]bool{}
			}
			for w, n := range ws {
				if n > 0 {
					s.vocab[c][w] = true
				}
			}
		}
		for n, k := range p.numbers {
			if k > 0 {
				s.support[n] = true
			}
		}
	}
	return s
}

// usable reports which parts of f carry enough features to score against ps.
func usable(f Features, ps []*profile, kinds FeatureKind) FeatureKind {
	var use FeatureKind
	if kinds&WordFeatures != 0 && len(f.Words) > 0 && float64(len(f.Words)) >= MinFeatureShare*float64(f.WordsAsked) {
		use |= WordFeatures
	}
	if kinds&NumberFeatures != 0 && len(f.Numbers) > 0 && float64(len(f.Numbers)) >= MinFeatureShare*float64(f.NumbersAsked) &&
		!slices.ContainsFunc(ps, func(p *profile) bool { return p.total == 0 }) {
		use |= NumberFeatures
	}
	return use
}

// score returns the log-score of f under each profile: a Laplace-smoothed
// multinomial log-likelihood of the words, one independent choice per
// category, plus -H/NumberTemperature for the Hellinger distance H between
// the answer's integers and the model's smoothed reference integers.
func (s scorer) score(f Features, ps []*profile, use FeatureKind) []float64 {
	out := make([]float64, len(ps))
	if use&WordFeatures != 0 {
		for _, c := range allCategories {
			w, ok := f.Words[c]
			if !ok {
				continue
			}
			// One outcome more for a word no candidate gave.
			v := len(s.vocab[c]) + 1
			for i, p := range ps {
				out[i] += math.Log((float64(p.words[c][w]) + laplaceAlpha) / (float64(p.answered[c]) + laplaceAlpha*float64(v)))
			}
		}
	}
	if use&NumberFeatures != 0 {
		counts := map[int]int{}
		for _, n := range f.Numbers {
			counts[n]++
		}
		size := len(s.support)
		for n := range counts {
			if !s.support[n] {
				size++
			}
		}
		values := make([]int, 0, len(counts))
		for n := range counts {
			values = append(values, n)
		}
		sort.Ints(values)
		for i, p := range ps {
			// Bhattacharyya coefficient; outcomes the answer lacks add nothing.
			bc := 0.0
			for _, n := range values {
				q := (float64(p.numbers[n]) + laplaceAlpha) / (float64(p.total) + laplaceAlpha*float64(size))
				bc += math.Sqrt(float64(counts[n]) / float64(len(f.Numbers)) * q)
			}
			out[i] -= math.Sqrt(max(0, 1-bc)) / NumberTemperature
		}
	}
	return out
}

// posteriors turns log-scores into posteriors under a uniform prior.
func posteriors(logs []float64) []float64 {
	top := slices.Max(logs)
	out := make([]float64, len(logs))
	sum := 0.0
	for i, l := range logs {
		out[i] = math.Exp(l - top)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Classify scores probe against every model of bank and decides whether the
// claimed model answered it. Candidates are not narrowed by the claim. It
// fails only when another model wins with posterior at least MinPosterior,
// leads the runner-up by MinLogMargin and the bank separates it from the
// claimed model (MinPairAccuracy); it passes on the same conditions with the
// claimed model winning. Everything else is a skip: an empty bank, a claimed
// model the bank lacks or has too few references for, an answer with too few
// features, and close calls.
func Classify(bank []Reference, claimed string, probe Features) Classification {
	res := Classification{Decision: Skip, Claimed: claimed}
	all := profiles(bank, probe.Template)
	var ps []*profile
	ci := -1
	for _, p := range all {
		if len(p.samples) < MinReferenceSamples {
			res.Excluded = append(res.Excluded, p.model)
			if p.model == claimed {
				res.Reason = fmt.Sprintf("%d references for %s, need %d", len(p.samples), claimed, MinReferenceSamples)
				return res
			}
			continue
		}
		if p.model == claimed {
			ci = len(ps)
		}
		ps = append(ps, p)
	}
	switch {
	case len(all) == 0:
		res.Reason = fmt.Sprintf("bank has no references for template %q", probe.Template)
		return res
	case ci < 0:
		res.Reason = fmt.Sprintf("%s is not in the bank", claimed)
		return res
	case len(ps) < 2:
		res.Reason = "bank has no other model to compare with"
		return res
	}
	use := usable(probe, ps, AllFeatures)
	if use == 0 {
		res.Reason = fmt.Sprintf("answer yielded %d of %d words and %d of %d integers", len(probe.Words), probe.WordsAsked, len(probe.Numbers), probe.NumbersAsked)
		return res
	}

	s := newScorer(ps)
	logs := s.score(probe, ps, use)
	post := posteriors(logs)
	order := make([]int, len(ps))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logs[order[a]] > logs[order[b]] })
	for _, i := range order {
		res.Scores = append(res.Scores, Score{Model: ps[i].model, LogScore: logs[i], Posterior: post[i]})
	}
	win, second := order[0], order[1]
	res.Margin = logs[win] - logs[second]
	if post[win] < MinPosterior || res.Margin < MinLogMargin {
		res.Reason = fmt.Sprintf("close call: %s leads with posterior %.2f and margin %.2f", ps[win].model, post[win], res.Margin)
		return res
	}
	other := ci
	if win == ci {
		other = second
	}
	res.PairAccuracy = pairAccuracy(ps[win], ps[other], probe, use)
	if res.PairAccuracy < MinPairAccuracy {
		res.Reason = fmt.Sprintf("bank does not separate %s from %s: leave-one-out accuracy %.2f", ps[win].model, ps[other].model, res.PairAccuracy)
		return res
	}
	if win == ci {
		res.Decision = Pass
		res.Reason = fmt.Sprintf("%s wins with posterior %.2f", claimed, post[win])
	} else {
		res.Decision = Fail
		res.Reason = fmt.Sprintf("%s wins with posterior %.2f; %s has %.2f", ps[win].model, post[win], claimed, post[ci])
	}
	return res
}

// pairAccuracy is the leave-one-out accuracy of a bank holding only a and b,
// scored on the features in use and, of the words, only on the categories
// the probe answered. It is 0 unless references of both models were scored.
func pairAccuracy(a, b *profile, probe Features, use FeatureKind) float64 {
	var cats []Category
	for _, c := range allCategories {
		if _, ok := probe.Words[c]; ok {
			cats = append(cats, c)
		}
	}
	acc := leaveOneOut([]*profile{a, b}, use, cats)
	if acc.Confusion[a.model] == nil || acc.Confusion[b.model] == nil {
		return 0
	}
	return float64(acc.Correct) / float64(acc.Total)
}

// Accuracy is the result of a leave-one-out evaluation.
type Accuracy struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
	// Confusion counts, for each model, which model each of its samples was
	// attributed to. A tie counts as attributed to no model ("").
	Confusion map[string]map[string]int `json:"confusion"`
}

// LeaveOneOut classifies every sample of bank against the rest of the bank,
// by highest log-score over all models and with the parts of the answer
// selected by kinds, as the 2026-10-26 pilot did. Samples are only compared
// with samples of their own template version. Samples with too few features
// of the selected kinds, and the only sample of a model, are not counted.
func LeaveOneOut(bank []Reference, kinds FeatureKind) Accuracy {
	acc := Accuracy{Confusion: map[string]map[string]int{}}
	var templates []string
	for _, r := range bank {
		for _, f := range r.Samples {
			if !slices.Contains(templates, f.Template) {
				templates = append(templates, f.Template)
			}
		}
	}
	for _, t := range templates {
		a := leaveOneOut(profiles(bank, t), kinds, nil)
		acc.Correct += a.Correct
		acc.Total += a.Total
		for m, row := range a.Confusion {
			if acc.Confusion[m] == nil {
				acc.Confusion[m] = map[string]int{}
			}
			for k, n := range row {
				acc.Confusion[m][k] += n
			}
		}
	}
	return acc
}

// leaveOneOut scores each held-out sample on the words of cats only, or on
// all its words when cats is nil.
func leaveOneOut(ps []*profile, kinds FeatureKind, cats []Category) Accuracy {
	acc := Accuracy{Confusion: map[string]map[string]int{}}
	if len(ps) < 2 {
		return acc
	}
	for i, p := range ps {
		// Without its only reference a model has nothing to be recognized by.
		if len(p.samples) < 2 {
			continue
		}
		for _, f := range p.samples {
			p.add(f, -1)
			held := f
			if cats != nil {
				held.Words, held.WordsAsked = map[Category]string{}, len(cats)
				for _, c := range cats {
					if w, ok := f.Words[c]; ok {
						held.Words[c] = w
					}
				}
			}
			use := usable(held, ps, kinds)
			if use != 0 {
				logs := newScorer(ps).score(held, ps, use)
				got := ""
				best := slices.Max(logs)
				if j := slices.Index(logs, best); slices.Index(logs[j+1:], best) < 0 {
					got = ps[j].model
				}
				if acc.Confusion[p.model] == nil {
					acc.Confusion[p.model] = map[string]int{}
				}
				acc.Confusion[p.model][got]++
				acc.Total++
				if got == ps[i].model {
					acc.Correct++
				}
			}
			p.add(f, 1)
		}
	}
	return acc
}
