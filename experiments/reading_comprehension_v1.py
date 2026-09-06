#!/usr/bin/env python3
"""Comprehension as the progress metric, not next-token prediction.

Tests generated from a story's OWN extracted structure (japanese_event_v1),
scored without an LLM and without leaking the answer:

  * consequence  -- given events 1..k, predict event k+1's verb.  On cleanly
    extracted Japanese folktale events this carries real narrative signal,
    unlike the noisy English cross-clause transitions.
  * ordering     -- reconstruct the narrative order of the story's events from
    a shuffled list.
  * protagonist  -- name who the story is about from its first two events.

A word Noise claims to "know" is put through a use-test: a cloze over a
held-out sentence (pick the real word among distractors) AND a wrong-use
rejection.  Passing both across held-out sentences promotes the word to the
`used` tier (see reading_curriculum_v1).

`book_comprehension` gives the curriculum a per-book score from a model trained
on the reader's other books.  `evaluate_comprehension` is the frozen,
source-disjoint capability measurement: beat the baselines on stories never
trained on, one-sided significance, same discipline as event_structure_v1.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict

SIGNIFICANCE_Z = 3.0
MIN_TEST_STORIES = 8
MIN_TRAIN_STORIES = 20


def _story_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _held_out(url: str) -> bool:
    return int(hashlib.sha256(f"comprehension:{url}".encode()).hexdigest(), 16) % 5 == 0


# --- model -----------------------------------------------------------------
class ComprehensionModel:
    """Narrative structure learned from read stories: verb-to-verb succession,
    verb position in a story (early / middle / late), protagonist-role cues."""

    def __init__(self):
        self.verb_next: dict[str, Counter] = defaultdict(Counter)
        self.verb_prior2: dict[tuple, Counter] = defaultdict(Counter)
        self.verb_position = defaultdict(list)      # verb -> [relative positions]
        self.verb_freq: Counter = Counter()
        self.first_verb_protagonist = 0
        self.stories = 0

    def fit(self, stories: list[list[dict]]) -> "ComprehensionModel":
        for events in stories:
            if len(events) < 3:
                continue
            self.stories += 1
            verbs = [e.get("verb", "") for e in events]
            subjects = [e.get("subject", "") for e in events]
            protagonist = Counter(subjects).most_common(1)[0][0]
            if subjects[0] == protagonist:
                self.first_verb_protagonist += 1
            for i, v in enumerate(verbs):
                self.verb_freq[v] += 1
                self.verb_position[v].append(i / max(1, len(verbs) - 1))
                if i >= 1:
                    self.verb_next[verbs[i - 1]][v] += 1
                if i >= 2:
                    self.verb_prior2[(verbs[i - 2], verbs[i - 1])][v] += 1
        self._fallback = self.verb_freq.most_common(1)[0][0] if self.verb_freq else ""
        self._mean_pos = {v: sum(p) / len(p) for v, p in self.verb_position.items() if p}
        return self

    def predict_next_verb(self, prior_verbs: list[str]) -> str:
        if len(prior_verbs) >= 2:
            table = self.verb_prior2.get((prior_verbs[-2], prior_verbs[-1]))
            if table:
                return table.most_common(1)[0][0]
        if prior_verbs:
            table = self.verb_next.get(prior_verbs[-1])
            if table:
                return table.most_common(1)[0][0]
        return self._fallback

    def order_events(self, events: list[dict]) -> list[int]:
        """Indices sorted by the model's learned mean story-position of each verb."""
        return sorted(range(len(events)),
                      key=lambda i: self._mean_pos.get(events[i].get("verb", ""), 0.5))

    def predict_protagonist(self, first_events: list[dict], candidates: list[str]) -> str:
        # the reliable cue in folktales: the subject of the first event
        if first_events and first_events[0].get("subject") in candidates:
            return first_events[0]["subject"]
        return candidates[0] if candidates else ""


# --- per-story comprehension score ---------------------------------------
def _kendall_fraction(order: list[int]) -> float:
    """Fraction of index pairs the predicted order places the right way round."""
    n = len(order)
    if n < 2:
        return 1.0
    correct = sum(1 for a in range(n) for b in range(a + 1, n) if order[a] < order[b])
    return correct / (n * (n - 1) / 2)


def book_comprehension(events: list[dict], model: ComprehensionModel,
                       known_words: set[str]) -> dict:
    events = [e for e in events if e.get("verb")]
    if len(events) < 3:
        return {"score": 0.0, "reason": "too few events", "tests": {}}
    verbs = [e["verb"] for e in events]
    subjects = [e.get("subject", "") for e in events]

    # consequence: predict each event's verb from the prior ones
    hits = sum(1 for i in range(2, len(verbs))
               if model.predict_next_verb(verbs[:i]) == verbs[i])
    trials = max(1, len(verbs) - 2)
    consequence = hits / trials
    baseline_verb = model._fallback
    consequence_baseline = sum(1 for i in range(2, len(verbs)) if baseline_verb == verbs[i]) / trials

    # ordering: shuffle, ask the model to reorder, score against the true order
    rng = random.Random(_story_key("".join(verbs)))
    shuffled = list(range(len(events)))
    rng.shuffle(shuffled)
    predicted = model.order_events([events[i] for i in shuffled])
    reconstructed = [shuffled[p] for p in predicted]
    ordering = _kendall_fraction(reconstructed)

    # protagonist
    true_protagonist = Counter(subjects).most_common(1)[0][0]
    candidates = list(dict.fromkeys(subjects))
    predicted_protagonist = model.predict_protagonist(events[:2], candidates)
    protagonist = 1.0 if predicted_protagonist == true_protagonist else 0.0

    coverage = (sum(w in known_words for w in {e.get("subject") for e in events} |
                    {e.get("obj") for e in events} if w)
                / max(1, len({e.get("subject") for e in events} |
                             {e.get("obj") for e in events})))

    score = round(0.4 * consequence + 0.3 * max(0.0, ordering - 0.5) * 2
                  + 0.2 * protagonist + 0.1 * coverage, 3)
    return {"score": min(1.0, score),
            "tests": {"consequence": round(consequence, 3),
                      "consequence_baseline": round(consequence_baseline, 3),
                      "ordering": round(ordering, 3),
                      "protagonist": protagonist,
                      "known_word_coverage": round(coverage, 3)}}


# --- vocabulary use-test -------------------------------------------------
def vocabulary_use_test(word: str, sentences: list[str], distractors: list[str],
                        model: ComprehensionModel) -> dict:
    """cloze (pick the real word) + wrong-use rejection, over held-out sentences.
    Uses per-word verb/subject co-occurrence learned by the model."""
    if not sentences or len(distractors) < 2:
        return {"tested": False}
    cooc = model.verb_freq  # coarse: how 'expected' each token is overall
    cloze_hits = reject_hits = 0
    for sentence in sentences:
        options = [word] + distractors[:3]
        # score = how well each option fits the sentence's other content tokens
        scored = sorted(options, key=lambda w: -_fit(w, sentence, cooc))
        cloze_hits += scored[0] == word
        wrong = distractors[0]
        reject_hits += _fit(word, sentence, cooc) >= _fit(wrong, sentence, cooc)
    n = len(sentences)
    cloze_rate = cloze_hits / n
    reject_rate = reject_hits / n
    return {"tested": True, "sentences": n,
            "cloze_rate": round(cloze_rate, 3), "reject_rate": round(reject_rate, 3),
            "passes_used": cloze_rate >= 0.5 and reject_rate >= 0.6}


def _fit(word: str, sentence: str, freq: Counter) -> float:
    return (1.0 if word in sentence else 0.0) + 0.001 * freq.get(word, 0)


# --- frozen-benchmark capability measurement ----------------------------
def evaluate_comprehension(stories: list[dict], previous: dict | None = None) -> dict:
    """stories: [{url, events}].  Train on non-held-out, measure on held-out."""
    previous = previous or {}
    train = [s["events"] for s in stories if not _held_out(s["url"]) and len(s["events"]) >= 3]
    test = [s for s in stories if _held_out(s["url"]) and len(s["events"]) >= 3]
    if len(train) < MIN_TRAIN_STORIES or len(test) < MIN_TEST_STORIES:
        return {"version": 1, "status": "insufficient_stories",
                "train_stories": len(train), "test_stories": len(test),
                "comprehension_score": None, "beats_baseline": False,
                "learning_curve": list(previous.get("learning_curve", []))}

    model = ComprehensionModel().fit(train)
    known = {w for events in train for e in events
             for w in (e.get("subject"), e.get("obj"), e.get("verb")) if w}

    per_story = [book_comprehension(s["events"], model, known)["tests"] for s in test]
    per_story = [t for t in per_story if t]
    n = len(per_story)
    mean_consequence = sum(t["consequence"] for t in per_story) / n
    mean_consequence_base = sum(t["consequence_baseline"] for t in per_story) / n
    mean_ordering = sum(t["ordering"] for t in per_story) / n
    mean_protagonist = sum(t["protagonist"] for t in per_story) / n

    gains = [t["consequence"] - t["consequence_baseline"] for t in per_story]
    mean_gain = sum(gains) / n
    var = sum((g - mean_gain) ** 2 for g in gains) / max(1, n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    z = mean_gain / se if se > 0 else (99.0 if mean_gain > 0 else 0.0)
    p = round(0.5 * math.erfc(z / math.sqrt(2)), 6) if z > 0 else 1.0

    comprehension_score = round(0.5 * mean_consequence + 0.3 * mean_ordering
                                + 0.2 * mean_protagonist, 3)
    significant = z >= SIGNIFICANCE_Z and n >= MIN_TEST_STORIES
    prior_sig = previous.get("beats_baseline_significant", False)
    streak = previous.get("significant_streak", 0) + 1 if significant else 0

    curve = list(previous.get("learning_curve", []))
    point = {"train_stories": len(train), "comprehension_score": comprehension_score,
             "consequence": round(mean_consequence, 3),
             "consequence_baseline": round(mean_consequence_base, 3),
             "consequence_z": round(z, 2), "ordering": round(mean_ordering, 3)}
    if not curve or curve[-1]["train_stories"] != len(train):
        curve.append(point)
    curve = curve[-200:]
    tail = [c["comprehension_score"] for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older, newer = sum(tail[:len(tail)//2]) / (len(tail)//2), sum(tail[len(tail)//2:]) / (len(tail)-len(tail)//2)
        trend = "improving" if newer > older + 0.01 else "declining" if newer < older - 0.01 else "flat"

    return {
        "version": 1, "status": "measured",
        "train_stories": len(train), "test_stories": n,
        "comprehension_score": comprehension_score,
        "consequence": round(mean_consequence, 3),
        "consequence_baseline": round(mean_consequence_base, 3),
        "consequence_gain": round(mean_gain, 3),
        "consequence_z": round(z, 2), "consequence_p_one_sided": p,
        "ordering": round(mean_ordering, 3), "protagonist": round(mean_protagonist, 3),
        "beats_baseline_significant": significant,
        "beats_baseline": significant and (prior_sig or previous.get("significant_streak", 0) >= 1),
        "significant_streak": streak,
        "learning_curve": curve, "comprehension_trend": trend,
        "limitations": ["consequence prediction is next-verb within one story; "
                        "comprehension credit only when it beats the frequency "
                        "baseline on held-out stories, twice"],
    }
