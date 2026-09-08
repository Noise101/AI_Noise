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
import time
from collections import Counter, defaultdict

import japanese_benchmark_v1 as jb

SIGNIFICANCE_Z = 3.0
MIN_TEST_STORIES = 8
MIN_TRAIN_STORIES = 20


EVAL_REGIME = "tiered_frozen_v3"     # v3: learned back-off + proper-scoring consequence
SCORING_VERSION = 3                  # consequence is now predictive probability, not a 0/1 hit
BASELINE_DEFINITION = "per_story_unigram_next_verb_probability"
BENCH_SALT = "comprehension:tiered:v1"
MODEL_CONFIG = "learned_backoff(tri+bi+uni)+position+first_verb_protagonist"
SIGNIFICANT_TRAIN_GROWTH = 1.4       # selection must stay significant across this
                                    # much train growth before a final may open


def _story_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


_collection = jb.collection
_canon = jb.canonical_events
_fingerprint = jb.fingerprint


def _held_out(url: str) -> bool:
    """Legacy per-collection ~1/5 split -- the tiered benchmark uses jb.Tiers now;
    kept for callers / tests that still reason about a single held-out set."""
    return int(hashlib.sha256(f"comprehension:{jb.collection(url)}".encode()).hexdigest(), 16) % 5 == 0


def forbidden_training_collections(previous: dict | None, stories: list | None = None,
                                   ever_trained_collections=None, cycle: int = 0) -> set:
    """Collections that must stay OUT of any model / RNN training for this
    benchmark: everything the tiering puts in a non-train tier, plus every frozen
    selection / final / reserve collection."""
    previous = previous or {}
    prev = previous if previous.get("eval_regime") == EVAL_REGIME else {}
    t = jb.Tiers(stories or [], BENCH_SALT, prev,
                 ever_trained_collections=ever_trained_collections, cycle=cycle)
    return set(t.forbidden_train_collections)


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
        # learned interpolation weights for the trigram / bigram / unigram
        # back-off (fit on a held-out slice of the training stories).
        self.lam = (1 / 3, 1 / 3, 1 / 3)

    def fit(self, stories: list[list[dict]]) -> "ComprehensionModel":
        seqs = []
        for events in stories:
            if len(events) < 3:
                continue
            self.stories += 1
            verbs = [e.get("verb", "") for e in events]
            subjects = [e.get("subject", "") for e in events]
            protagonist = Counter(subjects).most_common(1)[0][0]
            if subjects[0] == protagonist:
                self.first_verb_protagonist += 1
            seqs.append(verbs)
            for i, v in enumerate(verbs):
                self.verb_freq[v] += 1
                self.verb_position[v].append(i / max(1, len(verbs) - 1))
                if i >= 1:
                    self.verb_next[verbs[i - 1]][v] += 1
                if i >= 2:
                    self.verb_prior2[(verbs[i - 2], verbs[i - 1])][v] += 1
        self._fallback = self.verb_freq.most_common(1)[0][0] if self.verb_freq else ""
        self._mean_pos = {v: sum(p) / len(p) for v, p in self.verb_position.items() if p}
        self._vocab_size = max(1, len(self.verb_freq))
        self._uni_total = max(1, sum(self.verb_freq.values()))
        self._fit_backoff(seqs)
        return self

    # --- learned back-off language model over verbs ------------------------
    def _components(self, p2: str, p1: str, v: str) -> tuple[float, float, float]:
        """Trigram, bigram, unigram probabilities of `v` (add-1 smoothed uni)."""
        tri = self.verb_prior2.get((p2, p1))
        p_tri = tri[v] / sum(tri.values()) if tri and sum(tri.values()) else 0.0
        bi = self.verb_next.get(p1)
        p_bi = bi[v] / sum(bi.values()) if bi and sum(bi.values()) else 0.0
        p_uni = (self.verb_freq.get(v, 0) + 1) / (self._uni_total + self._vocab_size)
        return p_tri, p_bi, p_uni

    def _fit_backoff(self, seqs: list[list[str]]) -> None:
        if len(seqs) < 4:
            return
        cut = max(1, len(seqs) // 4)
        val = [(vb[i - 2], vb[i - 1], vb[i]) for vb in seqs[:cut]
               for i in range(2, len(vb))]
        if not val:
            return
        # gradient ascent on validation log-likelihood over softmax(logits)
        logits = [0.0, 0.0, 0.0]
        lr = 0.5
        for _ in range(120):
            m = max(logits)
            ex = [math.exp(x - m) for x in logits]
            s = sum(ex)
            lam = [e / s for e in ex]
            grad = [0.0, 0.0, 0.0]
            for p2, p1, v in val:
                comps = self._components(p2, p1, v)
                mix = lam[0] * comps[0] + lam[1] * comps[1] + lam[2] * comps[2]
                if mix <= 0:
                    continue
                for k in range(3):
                    # d/dlogit_k of log(sum lam_j c_j), lam = softmax(logits)
                    dlam_k = lam[k] * (comps[k] - (lam[0] * comps[0] + lam[1] * comps[1] + lam[2] * comps[2]))
                    grad[k] += dlam_k / mix
            for k in range(3):
                logits[k] += lr * grad[k] / len(val)
        m = max(logits)
        ex = [math.exp(x - m) for x in logits]
        s = sum(ex)
        self.lam = tuple(e / s for e in ex)

    def verb_prob(self, prior_verbs: list[str], v: str) -> float:
        p2 = prior_verbs[-2] if len(prior_verbs) >= 2 else ""
        p1 = prior_verbs[-1] if prior_verbs else ""
        c = self._components(p2, p1, v)
        return self.lam[0] * c[0] + self.lam[1] * c[1] + self.lam[2] * c[2]

    def unigram_prob(self, v: str) -> float:
        return (self.verb_freq.get(v, 0) + 1) / (self._uni_total + self._vocab_size)

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

    # consequence: predict each event's verb from the prior ones.
    #  - `consequence` / `consequence_baseline`: the 0/1 argmax hit rate the
    #    curriculum's per-book score uses (unchanged).
    #  - `consequence_prob` / `_baseline`: the probability the LEARNED back-off
    #    model / the unigram assign to the true next verb -- a proper scoring
    #    rule that moves smoothly with training (the capability benchmark).
    trials = max(1, len(verbs) - 2)
    hits = sum(1 for i in range(2, len(verbs))
               if model.predict_next_verb(verbs[:i]) == verbs[i])
    consequence = hits / trials
    baseline_verb = model._fallback
    consequence_baseline = sum(1 for i in range(2, len(verbs)) if baseline_verb == verbs[i]) / trials
    probs = [model.verb_prob(verbs[:i], verbs[i]) for i in range(2, len(verbs))]
    uni = [model.unigram_prob(verbs[i]) for i in range(2, len(verbs))]
    consequence_prob = sum(probs) / trials
    consequence_prob_baseline = sum(uni) / trials

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
                      "consequence_prob": round(consequence_prob, 4),
                      "consequence_prob_baseline": round(consequence_prob_baseline, 4),
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


# --- tiered frozen-benchmark capability measurement --------------------
def comprehension_training_fingerprint(train_stories: list[dict]) -> dict:
    """Full, auditable identity of the ComprehensionModel's training data:
    URL *and canonical event content* of every train story, plus parser /
    scoring / baseline / model config.  Detects parser changes, event-content
    changes and training-source changes -- not just a changed URL list."""
    from japanese_event_v1 import PARSER_VERSION
    srcs = sorted(({"url": s["url"], "events": _canon(s["events"])} for s in train_stories),
                  key=lambda s: s["url"])
    src_payload = json.dumps(srcs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    identity = {
        "regime": EVAL_REGIME, "scoring_version": SCORING_VERSION,
        "parser_version": PARSER_VERSION, "baseline_definition": BASELINE_DEFINITION,
        "model_config": MODEL_CONFIG,
    }
    return {
        "identity": identity,
        "identity_fingerprint": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16],
        "training_set_fingerprint": hashlib.sha256(src_payload.encode()).hexdigest()[:16],
        "training_source_count": len(srcs),
        "training_urls": [s["url"] for s in srcs],
    }


def _comprehension_model_fingerprint(train_stories: list[dict]) -> str:
    tf = comprehension_training_fingerprint(train_stories)
    return hashlib.sha256(
        (tf["identity_fingerprint"] + ":" + tf["training_set_fingerprint"]).encode()
    ).hexdigest()[:16]


def _measure(test_stories: list[dict], model: "ComprehensionModel", known: set) -> dict | None:
    per = [book_comprehension(s["events"], model, known)["tests"] for s in test_stories]
    per = [t for t in per if t]
    n = len(per)
    if not n:
        return None
    # the capability signal is the LEARNED model's predictive probability of the
    # true next verb vs the unigram's -- a proper scoring rule (SCORING_VERSION 3)
    mc = sum(t["consequence_prob"] for t in per) / n
    mcb = sum(t["consequence_prob_baseline"] for t in per) / n
    mo = sum(t["ordering"] for t in per) / n
    mp = sum(t["protagonist"] for t in per) / n
    gains = [t["consequence_prob"] - t["consequence_prob_baseline"] for t in per]
    mg = sum(gains) / n
    var = sum((g - mg) ** 2 for g in gains) / max(1, n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    z = mg / se if se > 0 else (99.0 if mg > 0 else 0.0)
    p = round(0.5 * math.erfc(z / math.sqrt(2)), 6) if z > 0 else 1.0
    # hit-rate view kept for the dashboard (not the significance signal)
    hit = sum(t["consequence"] for t in per) / n
    lift = max(0.0, min(1.0, (mc - mcb) / max(mcb, 1e-6)))   # relative gain over unigram
    return {"n": n, "consequence": round(mc, 4), "consequence_baseline": round(mcb, 4),
            "consequence_gain": round(mg, 4), "z": round(z, 2), "p_one_sided": p,
            "consequence_hit_rate": round(hit, 3),
            "ordering": round(mo, 3), "protagonist": round(mp, 3),
            "comprehension_score": round(0.5 * lift + 0.3 * mo + 0.2 * mp, 3),
            "significant": z >= SIGNIFICANCE_Z and n >= MIN_TEST_STORIES}


def evaluate_comprehension(stories: list[dict], previous: dict | None = None,
                           ever_trained_collections=None, cycle: int = 0) -> dict:
    """Tiered frozen benchmark (re-audit #6).

    SELECTION: frozen once, measured every cycle -- diagnostic learning curve +
    an ANCHOR-based streak (P1-3).  FINAL: a candidate checkpoint is registered
    only at streak 2 with meaningful new training (P1-4); its one-shot final runs
    once against that frozen checkpoint; a pass becomes a permanent
    `confirmed_checkpoint`.  Ordinary continued training never burns the reserve.
    """
    previous = previous or {}
    regime_ok = previous.get("eval_regime") == EVAL_REGIME
    prev = previous if regime_ok else {}
    from japanese_event_v1 import PARSER_VERSION

    tiers = jb.Tiers(stories, BENCH_SALT, prev,
                     ever_trained_collections=ever_trained_collections, cycle=cycle)
    train_stories = tiers.train_stories
    train_events = [s["events"] for s in train_stories]
    base = {"version": 4, "eval_regime": EVAL_REGIME,
            "regime_reset_from": previous.get("eval_regime") if (previous and not regime_ok) else None,
            "snapshot_migrated": tiers.selection_migrated or (bool(previous) and not regime_ok),
            **tiers.report_fields(),
            "selection_snapshot": tiers.selection_snapshot,
            "reserve_snapshot": tiers.reserve_snapshot,
            "learning_curve": list(prev.get("learning_curve", []))}

    if not tiers.selection_frozen or len(train_events) < MIN_TRAIN_STORIES:
        return {**base, "status": "insufficient_selection_stories", "beats_baseline": False,
                "capability_confirmed": False, "comprehension_score": None,
                "capability_pending_reason": tiers.selection_insufficient_reason or "not_enough_train",
                "train_stories": len(train_events)}

    model = ComprehensionModel().fit(train_events)
    known = {w for events in train_events for e in events
             for w in (e.get("subject"), e.get("obj"), e.get("verb")) if w}
    ctf = comprehension_training_fingerprint(train_stories)
    model_fp = _comprehension_model_fingerprint(train_stories)

    sel = _measure(tiers.selection_snapshot, model, known)
    if sel is None:
        return {**base, "status": "insufficient_selection_stories", "beats_baseline": False,
                "capability_confirmed": False, "comprehension_score": None,
                "train_stories": len(train_events),
                "capability_pending_reason": "no scorable selection stories"}
    sel_measurements = prev.get("selection_measurements", 0) + 1
    streak_state = jb.advance_selection_streak(prev.get("selection_streak"),
                                               sel["significant"], len(train_events), model_fp)
    sel_sig_streak = streak_state["streak"]

    # ---- candidate checkpoint + one-shot final (P1-4) ----
    candidates = [dict(c) for c in prev.get("candidate_checkpoints", [])]
    if (jb.should_register_candidate(candidates, sel_sig_streak, len(train_events))
            and tiers.disjoint):
        nxt = tiers.next_unopened_final()
        cand = {"registered_at_train": len(train_events), "registered_at_cycle": cycle,
                "model_fingerprint": model_fp,
                "comprehension_training_fingerprint": ctf,
                "identity": ctf["identity"], "significance_z": SIGNIFICANCE_Z,
                "selection_result": sel, "selection_fingerprint": tiers.selection_fingerprint,
                "final_result": None}
        if nxt:
            fin = _measure(nxt["snapshot"], model, known)
            cand.update(tier=nxt["tier"], final_snapshot_fingerprint=nxt["fingerprint"],
                        final_snapshot_urls=nxt["urls"], final_result=fin,
                        confirmed_at=(cycle if (fin or {}).get("significant") else None),
                        evaluated_at=time.time())
            tiers.record_final(nxt)
        else:
            cand.update(tier=None, final_result=None,
                        note="no unopened final/reserve snapshot available")
        candidates = candidates + [cand]

    cap = jb.capability_view(candidates, model_fp)
    standing = candidates[-1] if candidates else None
    if not candidates:
        final_status = ("registerable_next_cycle" if sel_sig_streak >= 2
                        else "awaiting_selection_streak_2")
    elif standing and standing.get("final_result") is None:
        final_status = "candidate_registered_final_snapshot_unavailable"
    elif cap["confirmed_ever"]:
        final_status = ("confirmed_current_model" if cap["confirmed_current_model"]
                        else "confirmed_earlier_checkpoint_model_since_changed")
    else:
        final_status = "final_below_threshold"

    curve = list(prev.get("learning_curve", []))
    point = {"train_stories": len(train_events), "eval_regime": EVAL_REGIME, "cycle": cycle,
             "selection_measurement": sel_measurements, "model_fingerprint": model_fp,
             "comprehension_score": sel["comprehension_score"],
             "consequence": sel["consequence"], "consequence_baseline": sel["consequence_baseline"],
             "consequence_z": sel["z"], "ordering": sel["ordering"],
             "selection_significant": sel["significant"], "selection_streak": sel_sig_streak}
    if not curve or curve[-1]["train_stories"] != len(train_events):
        curve.append(point)
    curve = curve[-200:]
    tail = [c["comprehension_score"] for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older, newer = sum(tail[:len(tail)//2]) / (len(tail)//2), sum(tail[len(tail)//2:]) / (len(tail)-len(tail)//2)
        trend = "improving" if newer > older + 0.01 else "declining" if newer < older - 0.01 else "flat"

    beats = cap["confirmed_ever"]
    return {
        **base, "status": "measured",
        "train_stories": len(train_events),
        "model_fingerprint": model_fp,
        "comprehension_training_fingerprint": ctf,
        # selection (diagnostic, NOT capability)
        "selection": sel,
        "selection_measurements": sel_measurements,
        "selection_streak": streak_state,
        "selection_significant_streak": sel_sig_streak,
        "selection_next_streak_train": streak_state.get("next_streak_train"),
        # candidate checkpoints + one-shot finals
        "candidate_checkpoints": candidates,
        "final_opened_count": sum(1 for c in candidates if c.get("final_result") is not None),
        "final_query_budget": jb.FINAL_QUERY_BUDGET,
        "final_status": final_status,
        "final_result": (standing or {}).get("final_result"),
        "final_preconditions": ({k: standing[k] for k in
                                 ("registered_at_train", "model_fingerprint", "identity",
                                  "significance_z", "selection_fingerprint",
                                  "final_snapshot_fingerprint")
                                 if k in standing} if standing else None),
        # capability
        "capability_confirmed": beats,
        "capability_confirmed_ever": cap["confirmed_ever"],
        "capability_confirmed_current_model": cap["confirmed_current_model"],
        "confirmed_checkpoints": cap["confirmed_checkpoints"],
        "beats_baseline": beats,
        "capability_pending_reason": (None if beats else
            "awaiting selection streak 2 (anchor-based)" if sel_sig_streak < 2 else
            "candidate registered, no unopened final snapshot" if final_status ==
                "candidate_registered_final_snapshot_unavailable" else
            "final below threshold" if final_status == "final_below_threshold" else
            "candidate registerable next cycle"),
        # compat fields
        "comprehension_score": sel["comprehension_score"],
        "consequence": sel["consequence"], "consequence_baseline": sel["consequence_baseline"],
        "consequence_gain": sel["consequence_gain"], "consequence_z": sel["z"],
        "consequence_p_one_sided": sel["p_one_sided"],
        "ordering": sel["ordering"], "protagonist": sel["protagonist"],
        "beats_baseline_significant": sel["significant"],
        "test_stories": sel["n"], "snapshot_fingerprint": tiers.selection_fingerprint,
        "snapshot_stories": len(tiers.selection_snapshot),
        "significant_streak": sel_sig_streak,
        "learning_curve": curve, "comprehension_trend": trend,
        "limitations": ["SELECTION is diagnostic only.  Capability = a candidate "
                        "checkpoint (registered at anchor-streak 2 with meaningful new "
                        "training) passing its ONE-SHOT unopened final.  A confirmed "
                        "checkpoint is permanent; the current model is separately "
                        "reported as (not) itself re-confirmed."],
    }
