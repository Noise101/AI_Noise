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
import json
import math
import random
from collections import Counter, defaultdict

SIGNIFICANCE_Z = 3.0
MIN_TEST_STORIES = 8
MIN_TRAIN_STORIES = 20


EVAL_REGIME = "frozen_snapshot_v1"
SIGNIFICANT_TRAIN_GROWTH = 1.4       # training must grow this much for another
                                    # "independent" significant measurement to count


def _story_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _collection(url: str) -> str:
    """Group a multi-part source (「イソップ童話集/きつねとつる」, an Aozora author's
    files directory) so its parts never straddle train and test.  A standalone
    work -- including a bare wiki page like /wiki/桃太郎, where the parent is only
    the generic /wiki mount -- is its own collection.

    parts == ['https:', '', host, seg1, seg2, ...]; real path segments start at
    index 3.  A collection needs >= 2 directory segments above the leaf, so
    /wiki/Title (one dir: "wiki") stays standalone while /wiki/Collection/Title
    and /cards/NNN/files/xxx.html group on their parent.
    """
    base = url.split("#")[0].split("?")[0].rstrip("/")
    parts = base.split("/")
    if len(parts[3:]) >= 3:
        return "/".join(parts[:-1])
    return base


def _held_out(url: str) -> bool:
    """Hold out whole COLLECTIONS, not individual URLs: an Aozora author's works
    share a collection, so a per-URL split would leak almost every author across
    train and test (and starve training).  A standalone work is its own
    collection, so this stays a ~1/5 split there."""
    key = _collection(url)
    return int(hashlib.sha256(f"comprehension:{key}".encode()).hexdigest(), 16) % 5 == 0


def _canon(events: list) -> list:
    out = []
    for e in events:
        if isinstance(e, dict):
            out.append({"subject": e.get("subject", ""), "verb": e.get("verb", ""),
                        "obj": e.get("obj", "")})
        else:
            out.append({"subject": e[0] if len(e) > 0 else "",
                        "verb": e[1] if len(e) > 1 else "",
                        "obj": e[2] if len(e) > 2 else ""})
    return out


def _fingerprint(snapshot: list) -> str:
    """Hash URL *and event content* (canonical JSON): a snapshot whose events
    were altered no longer matches even if the URL set is unchanged."""
    payload = json.dumps(
        sorted(({"url": s["url"], "events": _canon(s["events"])} for s in snapshot),
               key=lambda s: s["url"]),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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

    from japanese_retell_v1 import retelling_coherence
    # a book whose extracted structure does not read as Japanese must not
    # graduate on ordering/coverage cues alone -- gate the score by coherence
    coherence = retelling_coherence(events)
    score = round(0.4 * consequence + 0.3 * max(0.0, ordering - 0.5) * 2
                  + 0.2 * protagonist + 0.1 * coverage, 3)
    score = round(min(1.0, score) * (0.5 + 0.5 * coherence), 3)
    return {"score": score,
            "tests": {"consequence": round(consequence, 3),
                      "consequence_baseline": round(consequence_baseline, 3),
                      "ordering": round(ordering, 3),
                      "protagonist": protagonist,
                      "coherence": coherence,
                      "known_word_coverage": round(coverage, 3)}}


# --- vocabulary use-test -------------------------------------------------
def _shuffled(options: list[str], salt: str) -> list[str]:
    """Deterministic order from a per-question salt -- the answer is not first."""
    return sorted(options, key=lambda w: hashlib.sha256(f"{salt}|{w}".encode()).hexdigest())


def vocabulary_use_test(word: str, events: list[dict], distractors: list[str],
                        model: ComprehensionModel,
                        cooccurrence: "dict[str, Counter] | None" = None,
                        url: str = "") -> dict:
    """Cloze + wrong-use rejection over held-out EVENTS.

    `events` are heuristic (subject, obj, verb, sentence) events in which `word`
    is one of the three slots.  The slot is BLANKED, the other two slots are the
    context, and each candidate is scored by its co-occurrence with that context
    -- context tokens and the co-occurrence table are the SAME event-token unit.
    The candidate order is a deterministic shuffle; the target must score
    STRICTLY above the best distractor AND have positive context evidence.
    """
    slots = ("subject", "obj", "verb")
    trials = [e for e in events if word in (e.get(s) for s in slots)]
    if len(trials) < 2 or len(distractors) < 2 or not cooccurrence:
        return {"tested": False, "reason": "need >=2 events, >=2 distractors, a co-occurrence table"}
    candidates = [word] + [d for d in distractors[:3] if d != word]
    if len(candidates) < 2:
        return {"tested": False, "reason": "need >=1 distinct distractor"}
    cloze_hits = reject_hits = 0
    for e in trials:
        target_slot = next(s for s in slots if e.get(s) == word)
        context = [e.get(s) for s in slots if s != target_slot and e.get(s)]
        salt = f"{url}|{e.get('sentence', '')}|{word}"
        ordered = _shuffled(candidates, salt)
        scores = {c: _context_fit(c, context, cooccurrence) for c in ordered}
        best_distractor = max((scores[c] for c in ordered if c != word), default=0.0)
        won = scores[word] > best_distractor and scores[word] > 0
        cloze_hits += won
        # wrong-use rejection: score the target against a FOREIGN context (the
        # context of a different trial).  A word whose fit is real context
        # sensitivity -- not raw frequency -- should NOT also win there.
        foreign = trials[(trials.index(e) + 1) % len(trials)]
        fslot = next((s for s in slots if foreign.get(s) == word), None)
        fcontext = [foreign.get(s) for s in slots if s != fslot and foreign.get(s)]
        fscores = {c: _context_fit(c, fcontext, cooccurrence) for c in ordered}
        fbest = max((fscores[c] for c in ordered if c != word), default=0.0)
        reject_hits += won and not (fscores[word] > fbest and fscores[word] > 0)
    n = len(trials)
    cloze_rate = cloze_hits / n
    reject_rate = reject_hits / n
    return {"tested": True, "trials": n,
            "cloze_rate": round(cloze_rate, 3), "reject_rate": round(reject_rate, 3),
            "passes_used": cloze_rate >= 0.6 and reject_rate >= 0.6}


def _context_fit(word: str, context: list[str], cooccurrence: "dict[str, Counter]") -> float:
    row = cooccurrence.get(word)
    if not row:
        return 0.0
    return sum(row.get(c, 0) for c in context) / (1 + sum(row.values()))


def build_cooccurrence(stories: list[dict]) -> "dict[str, Counter]":
    """word -> Counter(other content words seen in the same event), from events."""
    from collections import defaultdict
    table: "dict[str, Counter]" = defaultdict(Counter)
    for story in stories:
        for e in story.get("events", []):
            toks = [t for t in (e.get("subject"), e.get("obj"), e.get("verb")) if t]
            for a in toks:
                for b in toks:
                    if a != b:
                        table[a][b] += 1
    return dict(table)


# --- frozen-benchmark capability measurement ----------------------------
def evaluate_comprehension(stories: list[dict], previous: dict | None = None) -> dict:
    """stories: [{url, events}].  The held-out test set is a SNAPSHOT frozen the
    first time it is large enough -- new books only ever grow the training side.
    """
    previous = previous or {}
    regime_ok = previous.get("eval_regime") == EVAL_REGIME
    # an eval-regime change never inherits the old regime's streak / confirmation
    # / last-significant-train: the new method's first measurement starts fresh.
    streak_prev = previous if regime_ok else {}
    snap = previous.get("test_snapshot") if regime_ok else None
    # sticky: once the frozen snapshot replaced a legacy (non-frozen) evaluation
    # it stays flagged, so status keeps showing that the migration happened
    migrated = bool(previous.get("snapshot_migrated")) or (bool(previous) and not regime_ok)

    train = [s["events"] for s in stories
             if not _held_out(s["url"]) and len(s["events"]) >= 3]

    if snap and regime_ok:
        test = [{"url": s["url"], "events": [tuple(e) if isinstance(e, list) else e
                                             for e in s["events"]]} for s in snap]
    else:
        candidate_test = [s for s in stories if _held_out(s["url"]) and len(s["events"]) >= 3]
        if len(train) < MIN_TRAIN_STORIES or len(candidate_test) < MIN_TEST_STORIES:
            return {"version": 2, "status": "insufficient_stories", "eval_regime": EVAL_REGIME,
                    "train_stories": len(train), "test_stories": len(candidate_test),
                    "comprehension_score": None, "beats_baseline": False,
                    "learning_curve": list(streak_prev.get("learning_curve", []))}
        test = candidate_test           # freeze it now
        snap = [{"url": s["url"], "events": s["events"]} for s in test]
        migrated = migrated or bool(previous)   # replaced a legacy evaluation

    held_urls = {s["url"] for s in snap}
    held_cols = {_collection(u) for u in held_urls}
    # a book that is now on the held-out list must never be in training either
    train = [s["events"] for s in stories
             if s["url"] not in held_urls and _collection(s["url"]) not in held_cols
             and len(s["events"]) >= 3]
    if len(train) < MIN_TRAIN_STORIES:
        return {"version": 2, "status": "insufficient_stories", "eval_regime": EVAL_REGIME,
                "train_stories": len(train), "test_stories": len(test),
                "comprehension_score": None, "beats_baseline": False,
                "test_snapshot": snap,
                "learning_curve": list(streak_prev.get("learning_curve", []))}

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
    prior_sig = streak_prev.get("beats_baseline_significant", False)
    # re-measuring the SAME frozen snapshot is not an independent replication --
    # the streak only advances when training has meaningfully grown since the
    # last significant measurement
    last_sig_train = streak_prev.get("last_significant_train", 0)
    grew = len(train) >= last_sig_train * SIGNIFICANT_TRAIN_GROWTH
    streak = (streak_prev.get("significant_streak", 0) + 1) if (significant and grew) \
        else (streak_prev.get("significant_streak", 0) if significant else 0)

    curve = list(streak_prev.get("learning_curve", []))
    point = {"train_stories": len(train), "eval_regime": EVAL_REGIME,
             "comprehension_score": comprehension_score,
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

    beats = significant and (prior_sig or streak_prev.get("significant_streak", 0) >= 1) and streak >= 2
    return {
        "version": 2, "status": "measured", "eval_regime": EVAL_REGIME,
        "snapshot_migrated": migrated,
        "regime_reset_from": previous.get("eval_regime") if (previous and not regime_ok) else None,
        "train_stories": len(train), "test_stories": n,
        "snapshot_stories": len(snap), "snapshot_fingerprint": _fingerprint(snap),
        "test_snapshot": snap,
        "comprehension_score": comprehension_score,
        "consequence": round(mean_consequence, 3),
        "consequence_baseline": round(mean_consequence_base, 3),
        "consequence_gain": round(mean_gain, 3),
        "consequence_z": round(z, 2), "consequence_p_one_sided": p,
        "ordering": round(mean_ordering, 3), "protagonist": round(mean_protagonist, 3),
        "beats_baseline_significant": significant,
        "beats_baseline": beats,
        "significant_streak": streak,
        "last_significant_train": len(train) if significant else last_sig_train,
        "learning_curve": curve, "comprehension_trend": trend,
        "limitations": ["consequence prediction is next-verb within one story; "
                        "credit only when it beats the frequency baseline on the "
                        "FROZEN held-out snapshot, at two different training sizes"],
    }
