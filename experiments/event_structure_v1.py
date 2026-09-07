#!/usr/bin/env python3
"""Within-event structure learning on real reading material.

Replaces world_model_v51 / association_learning_v33 / causal_experiment_v28 /
representation_learning_v31.  Those four all predicted the *next* event from the
current one; measured on this corpus (~2,400 parsed clauses, ~290 coherent
adjacencies) that target carries no learnable signal -- a verb-bigram is worse
than the frequency baseline, and every context key is seen with exactly one
outcome, so a conditional frequency table degenerates to memorised lookup.

This module predicts structure *inside* one event instead:

  * verb_cloze        -- rank the observed verb among all known verbs given its
                         subject and object.  (selectional preference)
  * event_plausibility -- score a real event above a corrupted one.  Corruption
                         is pluggable; the shipped default only swaps the verb.

Both beat the frequency baseline on a source-disjoint holdout (5-fold CV:
verb_cloze +4.5pt top-1, event_plausibility +6.8pt), where next-event did not.

Decision replay, not full state replay: train_and_evaluate recomputes every
number from the raw audit each call.  `emitted_events` records WHEN/WHY a
benchmark locked or the selected model changed -- not the evaluation numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path


VERSION = 1

# --- frozen-benchmark constants -------------------------------------------------
BENCHMARK_REGIME = "collection_disjoint_within_event_v1"
BENCHMARK_SALT = "event-structure-benchmark:v1"
MIN_BENCHMARK_COLLECTIONS = 20
MINIMUM_EVALUATION_TOTAL = 15           # per selection / final split
MINIMUM_TRAIN_EVENTS = 200
FINAL_QUERY_BUDGET = 5
FINAL_TRAIN_GROWTH_FACTOR = 2

# --- significance gate --------------------------------------------------------
# Recalibrated for this redesign.  The family is {C0 baseline, C1 smoothed,
# C2 tiny-net} x {verb_cloze, event_plausibility} -- C2's slot is reserved now
# so adding it later does not move the bar.  C0 is the baseline every candidate
# is tested against, not itself a selectable hypothesis, but it keeps its slot
# in the divisor so the correction is stable across phases.
FAMILY_ALPHA = 0.05
MODEL_SLOTS = 3
TASK_COUNT = 2
SELECTION_ALPHA = FAMILY_ALPHA / (MODEL_SLOTS * TASK_COUNT)          # 0.05/6
FINAL_ALPHA = FAMILY_ALPHA / (MODEL_SLOTS * TASK_COUNT * FINAL_QUERY_BUDGET)

# --- C1 model constants (validated on live data, see module docstring) -------
DIRICHLET = 0.05        # per-feature add-k smoothing
GLOBAL_WEIGHT = 0.3     # weight on the global verb prior vs the argument cues
# Exact subject/object strings only.  Suffix-backoff features were measured to
# raise coverage 0.73 -> 0.93 but dilute the signal (cloax +2.2pt vs +4.5pt);
# object-class prediction was measured at -1.7pt vs baseline and is not a task.

WORD = re.compile(r"[A-Za-z']+")
TASKS = ("verb_cloze", "event_plausibility")


def source_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def collection_key(url: str) -> str:
    """Keep pages from one work/anthology on the same side of a split.

    Byte-for-byte the rule world_model_v51 used, so a benchmark frozen there is
    grouped the same way here."""
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.unquote(parsed.path).rstrip("/")
    marker = "/wiki/"
    if marker in path:
        title = path.split(marker, 1)[1]
        title = title.rsplit("/", 1)[0] if "/" in title else title
        return f"{parsed.netloc}/wiki/{title}"
    parent = path.rsplit("/", 1)[0] if path.count("/") > 1 else path
    return f"{parsed.netloc}{parent}"


# --- events -----------------------------------------------------------------
def iter_events(verified_experience: dict) -> list[tuple[str, str, str, str]]:
    """(subject, verb, object, source_url) for every parsed simple clause with a
    subject and a verb.  Object may be empty."""
    events = []
    for sequence in verified_experience.get("sequences", []):
        url = sequence.get("source_url")
        if not url:
            continue
        for raw in sequence.get("events", []):
            parts = raw.split("|", 2)
            if len(parts) != 3:
                continue
            subject, verb, obj = (part.strip().lower() for part in parts)
            if subject and verb:
                events.append((subject, verb, obj, url))
    return events


def arg_features(word: str, role: str) -> list[str]:
    """The cues C1 conditions a verb on.  Exact string only (see DIRICHLET note)."""
    return [f"{role}={word}"] if word else []


# --- corruption registry (event_plausibility negatives) ---------------------
def corrupt_verb(event: tuple[str, str, str, str], vocab: list[str],
                 weights: list[int], rng: random.Random) -> tuple | None:
    subject, verb, obj, url = event
    for _ in range(8):
        candidate = rng.choices(vocab, weights=weights, k=1)[0]
        if candidate != verb:
            return (subject, candidate, obj, url)
    return None


def corrupt_object(event: tuple[str, str, str, str], vocab: list[str],
                   weights: list[int], rng: random.Random) -> tuple | None:
    # Registered for the planned extension; not in DEFAULT_CORRUPTERS yet.
    subject, verb, obj, url = event
    for _ in range(8):
        candidate = rng.choices(vocab, weights=weights, k=1)[0]
        if candidate != obj:
            return (subject, verb, candidate, url)
    return None


CORRUPTERS = {"verb_swap": corrupt_verb, "object_swap": corrupt_object}
DEFAULT_CORRUPTERS = ("verb_swap",)


# --- models ----------------------------------------------------------------
class FrequencyBaseline:
    """C0.  Predicts the globally most common verb; for plausibility, prefers
    the more frequent of the two verbs."""

    model_id = "frequency_baseline"

    def __init__(self, train: list[tuple[str, str, str, str]]):
        self.verb_freq = Counter(verb for _, verb, _, _ in train)
        self.fallback = self.verb_freq.most_common(1)[0][0] if self.verb_freq else None

    def predict_verb(self, subject: str, obj: str) -> str | None:
        return self.fallback

    def score_event(self, subject: str, verb: str, obj: str) -> float:
        return float(self.verb_freq.get(verb, 0))

    def covers(self, subject: str, obj: str) -> bool:
        return False


class SmoothedArgumentModel:
    """C1.  P(verb | subject, object) as a smoothed product of the per-argument
    co-occurrence distributions and the global verb prior."""

    model_id = "smoothed_argument"

    def __init__(self, train: list[tuple[str, str, str, str]]):
        self.feature_verb: dict[str, Counter] = defaultdict(Counter)
        self.verb_freq: Counter = Counter()
        for subject, verb, obj, _ in train:
            self.verb_freq[verb] += 1
            for feature in arg_features(subject, "s") + arg_features(obj, "o"):
                self.feature_verb[feature][verb] += 1
        self.total = sum(self.verb_freq.values())
        self.vocab = sorted(self.verb_freq)
        self.fallback = self.verb_freq.most_common(1)[0][0] if self.verb_freq else None

    def _log_global(self, verb: str) -> float:
        return math.log((self.verb_freq.get(verb, 0) + 1) / (self.total + len(self.vocab) + 1))

    def score_event(self, subject: str, verb: str, obj: str) -> float:
        score = GLOBAL_WEIGHT * self._log_global(verb)
        for feature in arg_features(subject, "s") + arg_features(obj, "o"):
            counter = self.feature_verb.get(feature)
            if counter:
                score += math.log((counter.get(verb, 0) + DIRICHLET) /
                                  (sum(counter.values()) + DIRICHLET * len(self.vocab)))
        return score

    def covers(self, subject: str, obj: str) -> bool:
        return any(feature in self.feature_verb
                   for feature in arg_features(subject, "s") + arg_features(obj, "o"))

    def predict_verb(self, subject: str, obj: str) -> str | None:
        if not self.vocab:
            return None
        return max(self.vocab, key=lambda verb: self.score_event(subject, verb, obj))

    def rank_verb(self, subject: str, obj: str) -> list[str]:
        return sorted(self.vocab, key=lambda verb: -self.score_event(subject, verb, obj))


CANDIDATE_MODELS = (SmoothedArgumentModel,)   # C2 (tiny net) joins here in phase 3


# --- evaluation ----------------------------------------------------------------
def _sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2 ** discordant)


def _collection_sign(per_collection: "dict[str, list[int]]") -> tuple[int, int]:
    """A collection wins if the model's net correct-vs-baseline is positive
    within it, loses if negative.  Events inside one collection are correlated,
    so significance is a sign test over COLLECTIONS, not events."""
    wins = sum(1 for net in per_collection.values() if net[0] - net[1] > 0)
    losses = sum(1 for net in per_collection.values() if net[0] - net[1] < 0)
    return wins, losses


def _summarise(correct: int, baseline: int, total: int, covered: int,
               wins: int, losses: int,
               per_collection: "dict[str, list[int]] | None" = None) -> dict:
    coll_wins, coll_losses = _collection_sign(per_collection or {})
    return {"correct": correct, "baseline_correct": baseline, "total": total,
            "coverage": round(covered / total, 4) if total else 0.0,
            "lift": correct - baseline,
            "paired_wins": wins, "paired_losses": losses,
            "collection_wins": coll_wins, "collection_losses": coll_losses,
            "evaluated_collections": len(per_collection or {}),
            # significance is over collections; the event-level sign test stays
            # for reference only
            "one_sided_sign_p": round(_sign_p(coll_wins, coll_losses), 6),
            "event_level_sign_p": round(_sign_p(wins, losses), 6)}


def evaluate_cloze(model, baseline: FrequencyBaseline,
                   events: list[tuple[str, str, str, str]]) -> tuple[dict, list[dict]]:
    correct = base_correct = covered = wins = losses = 0
    per_collection: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trials = []
    for subject, verb, obj, url in events:
        predicted = model.predict_verb(subject, obj)
        base = baseline.predict_verb(subject, obj)
        hit, base_hit = predicted == verb, base == verb
        correct += hit
        base_correct += base_hit
        wins += hit and not base_hit
        losses += base_hit and not hit
        per_collection[collection_key(url)][0] += hit
        per_collection[collection_key(url)][1] += base_hit
        covered += model.covers(subject, obj)
        trials.append({"source_id": source_key(url), "subject": subject, "object": obj,
                       "predicted": predicted, "observed": verb, "baseline": base,
                       "correct": hit, "baseline_correct": base_hit})
    return _summarise(correct, base_correct, len(events), covered, wins, losses,
                      per_collection), trials


def evaluate_plausibility(model, baseline: FrequencyBaseline,
                          events: list[tuple[str, str, str, str]],
                          corrupters: tuple[str, ...], seed: str) -> tuple[dict, list[dict]]:
    vocab = list(baseline.verb_freq)
    weights = [baseline.verb_freq[v] for v in vocab]
    obj_vocab = sorted({obj for _, _, obj, _ in events if obj})
    correct = base_correct = covered = wins = losses = 0
    per_collection: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trials = []
    for subject, verb, obj, url in events:
        rng = random.Random(f"{seed}:{subject}|{verb}|{obj}|{url}")
        for name in corrupters:
            pool, pool_weights = ((vocab, weights) if name == "verb_swap"
                                  else (obj_vocab, [1] * len(obj_vocab)))
            corrupted = CORRUPTERS[name](
                (subject, verb, obj, url), pool, pool_weights, rng)
            if corrupted is None:
                continue
            _, bad_verb, bad_obj, _ = corrupted
            real = model.score_event(subject, verb, obj)
            fake = model.score_event(*corrupted[:3])
            base_real = baseline.score_event(subject, verb, obj)
            base_fake = baseline.score_event(corrupted[0], bad_verb, bad_obj)
            hit = real > fake                      # ties count as a miss
            base_hit = base_real > base_fake
            correct += hit
            base_correct += base_hit
            wins += hit and not base_hit
            losses += base_hit and not hit
            per_collection[collection_key(url)][0] += hit
            per_collection[collection_key(url)][1] += base_hit
            covered += model.covers(subject, obj)
            trials.append({"source_id": source_key(url), "corrupter": name,
                           "real": f"{subject}|{verb}|{obj}",
                           "corrupted": f"{corrupted[0]}|{bad_verb}|{bad_obj}",
                           "correct": hit, "baseline_correct": base_hit})
    return _summarise(correct, base_correct, len(trials), covered, wins, losses,
                      per_collection), trials


def passes_gain_gate(evaluation: dict, alpha: float) -> bool:
    total = evaluation.get("total", 0)
    required_lift = max(3, round(total * 0.03))
    return (total >= MINIMUM_EVALUATION_TOTAL
            and evaluation.get("lift", 0) >= required_lift
            and evaluation.get("coverage", 0.0) >= 0.1
            and evaluation.get("one_sided_sign_p", 1.0) <= alpha)


# --- frozen benchmark --------------------------------------------------------
def _balanced_collection_split(events: list[tuple[str, str, str, str]],
                               benchmark_collections: set[str]) -> set[str]:
    """Whole collections to the selection side, greedily balancing event count."""
    counts: Counter = Counter()
    for _, _, _, url in events:
        key = collection_key(url)
        if key in benchmark_collections:
            counts[key] += 1
    selection, totals = set(), [0, 0]
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        side = 0 if totals[0] <= totals[1] else 1
        if side == 0:
            selection.add(key)
        totals[side] += count
    return selection


def _candidate_benchmark(events: list[tuple[str, str, str, str]]) -> dict:
    collections_seen: Counter = Counter(collection_key(url) for _, _, _, url in events)
    ranked = sorted(collections_seen, key=lambda key: hashlib.sha256(
        f"{BENCHMARK_SALT}:{key}".encode()).hexdigest())
    take = min(len(ranked), max(MIN_BENCHMARK_COLLECTIONS, len(ranked) // 3))
    benchmark_collections = set(ranked[:take])
    selection_collections = _balanced_collection_split(events, benchmark_collections)
    selection = final = train = 0
    for _, _, _, url in events:
        key = collection_key(url)
        if key not in benchmark_collections:
            train += 1
        elif key in selection_collections:
            selection += 1
        else:
            final += 1
    ready = (selection >= MINIMUM_EVALUATION_TOTAL and final >= MINIMUM_EVALUATION_TOTAL
             and train >= MINIMUM_TRAIN_EVENTS and len(benchmark_collections) >= MIN_BENCHMARK_COLLECTIONS)
    return {"ready": ready, "eligible_collection_count": len(collections_seen),
            "benchmark_collections": sorted(benchmark_collections),
            "selection_collections": sorted(selection_collections),
            "candidate_selection_events": selection, "candidate_final_events": final,
            "candidate_train_events": train}


def choose_benchmark(events, previous: dict) -> dict:
    """Return the frozen split, locking it the first time it is ready.  Once
    locked the evaluation events themselves are a SNAPSHOT -- not re-derived from
    current data -- so the held-out set cannot grow as its collections accrue
    more events."""
    locked = (previous or {}).get("benchmark", {})
    if locked.get("selection_regime") == BENCHMARK_REGIME and locked.get("benchmark_collections"):
        result = {"ready": True, "locked": True,
                  "benchmark_collections": locked["benchmark_collections"],
                  "selection_collections": locked["selection_collections"],
                  "eligible_collection_count": locked.get("eligible_collection_count", 0)}
        snap_sel = locked.get("selection_event_snapshot")
        snap_fin = locked.get("final_event_snapshot")
        if snap_sel is not None and snap_fin is not None:
            result["selection_event_snapshot"] = [tuple(e) for e in snap_sel]
            result["final_event_snapshot"] = [tuple(e) for e in snap_fin]
        return result
    candidate = _candidate_benchmark(events)
    candidate["locked"] = False
    return candidate


# --- entry point -------------------------------------------------------------
def _empty_selected(task: str = "none") -> dict:
    return {"task": task, "model_id": "frequency_baseline", "correct": 0,
            "baseline_correct": 0, "total": 0, "coverage": 0.0, "lift": 0}


def train_and_evaluate(verified_experience: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    events = iter_events(verified_experience)
    benchmark = choose_benchmark(events, previous)
    emitted_events: list[dict] = []

    if not benchmark["ready"]:
        return _postponed_report(events, benchmark, previous)

    if not previous.get("benchmark", {}).get("locked"):
        emitted_events.append({
            "event_type": "benchmark_locked", "before": {"locked": False},
            "after": {"locked": True,
                      "collection_count": len(benchmark["benchmark_collections"])},
            "reason": "candidate benchmark's selection/final split and training pool "
                      "cleared the event-count readiness gate"})

    bench_set = set(benchmark["benchmark_collections"])
    selection_set = set(benchmark["selection_collections"])
    # training may grow (the point is to compare learning speed); the held-out
    # evaluation events are frozen the first time they are captured
    train_events = [e for e in events if collection_key(e[3]) not in bench_set]
    if benchmark.get("final_event_snapshot") is not None:
        selection_events = benchmark["selection_event_snapshot"]
        final_events = benchmark["final_event_snapshot"]
    else:
        selection_events = [e for e in events if collection_key(e[3]) in selection_set]
        final_events = [e for e in events
                        if collection_key(e[3]) in bench_set and collection_key(e[3]) not in selection_set]

    baseline = FrequencyBaseline(train_events)
    plaus_seed = hashlib.sha256(
        "\n".join(sorted(benchmark["benchmark_collections"])).encode()).hexdigest()[:16]

    evaluations: list[dict] = []
    fitted: dict[str, object] = {}
    for model_class in CANDIDATE_MODELS:
        model = model_class(train_events)
        fitted[model.model_id] = model
        for task in TASKS:
            if task == "verb_cloze":
                evaluation, _ = evaluate_cloze(model, baseline, selection_events)
            else:
                evaluation, _ = evaluate_plausibility(
                    model, baseline, selection_events, DEFAULT_CORRUPTERS, plaus_seed)
            evaluations.append({"task": task, "model_id": model.model_id,
                                "selection": evaluation, **evaluation})

    finalists = [item for item in evaluations
                 if passes_gain_gate(item["selection"], SELECTION_ALPHA)]
    candidate = max(finalists, key=lambda item: (item["selection"]["lift"],
                    item["selection"]["correct"], item["selection"]["coverage"]),
                    default=None)

    final_history = list(previous.get("final_attempt_history", []))
    last_final_train = final_history[-1].get("training_events", 0) if final_history else 0
    milestone_reached = (not final_history
                         or len(train_events) >= max(1, last_final_train) * FINAL_TRAIN_GROWTH_FACTOR)
    budget_left = len(final_history) < FINAL_QUERY_BUDGET

    selected = None
    final_attempt = None
    if candidate and milestone_reached and budget_left:
        model = fitted[candidate["model_id"]]
        if candidate["task"] == "verb_cloze":
            final_eval, _ = evaluate_cloze(model, baseline, final_events)
        else:
            final_eval, _ = evaluate_plausibility(
                model, baseline, final_events, DEFAULT_CORRUPTERS, plaus_seed)
        final_attempt = {"task": candidate["task"], "model_id": candidate["model_id"],
                         "evaluation": final_eval, "training_events": len(train_events),
                         "query_index": len(final_history) + 1}
        final_history = final_history + [final_attempt]
        if passes_gain_gate(final_eval, FINAL_ALPHA):
            selected = {"task": candidate["task"], "model_id": candidate["model_id"],
                        "selection": candidate["selection"], "final": final_eval,
                        **final_eval}
    elif candidate and final_history:
        # No fresh final query this cycle (milestone not reached, or budget
        # spent). A model that already cleared the final gate stays selected on
        # its last recorded final result rather than flapping back to the
        # baseline while it waits for the next milestone.
        last = final_history[-1]
        if (last.get("task") == candidate["task"] and last.get("model_id") == candidate["model_id"]
                and passes_gain_gate(last.get("evaluation", {}), FINAL_ALPHA)):
            selected = {"task": candidate["task"], "model_id": candidate["model_id"],
                        "selection": candidate["selection"], "final": last["evaluation"],
                        "stale_final": True, **last["evaluation"]}

    best_candidate = max(evaluations, key=lambda item: (item["selection"]["lift"],
                         item["selection"]["coverage"]), default=None)
    learning_curve = list(previous.get("learning_curve", []))
    best_sel = (best_candidate or {}).get("selection", {})
    point = {"training_events": len(train_events),
             "task": (best_candidate or {}).get("task"),
             "model_id": (best_candidate or {}).get("model_id"),
             "accuracy": (round(best_sel["correct"] / best_sel["total"], 4)
                          if best_sel.get("total") else 0.0),
             "lift": best_sel.get("lift", 0),
             "coverage": best_sel.get("coverage", 0.0),
             "one_sided_sign_p": best_sel.get("one_sided_sign_p", 1.0),
             "gate_passed": bool(selected)}
    if not learning_curve or learning_curve[-1]["training_events"] != point["training_events"]:
        learning_curve.append(point)
    learning_curve = learning_curve[-200:]
    learning_curve_trend = classify_trend(learning_curve, "lift", window=10, min_delta=1)

    selected_model_id = f"{selected['task']}:{selected['model_id']}" if selected else "frequency_baseline"
    old_model_id = previous.get("selected_model_id")
    if old_model_id and old_model_id != selected_model_id:
        emitted_events.append({"event_type": "selected_model_changed",
                               "before": {"selected_model_id": old_model_id},
                               "after": {"selected_model_id": selected_model_id},
                               "reason": "frozen final-split evaluation changed the ranking"})

    counterexamples = _counterexamples(fitted, baseline, selection_events, best_candidate)
    target = None
    if not selected and counterexamples:
        worst = counterexamples[0]
        target = {"seed": " ".join(w for w in (worst.get("subject"), worst.get("object")) if w)
                  or "simple action story",
                  "reason": "seek independent events with this argument pattern",
                  "failure_pattern": f"{worst.get('subject')}|?|{worst.get('object')}"}

    if selected:
        selection_status = "accepted_final_gain"
    elif candidate and len(final_history) >= FINAL_QUERY_BUDGET:
        selection_status = "final_query_budget_exhausted_no_confirmed_gain"
    elif candidate and not milestone_reached:
        selection_status = "awaiting_training_growth_for_next_final_query"
    elif candidate:
        selection_status = "candidate_failed_final_query"
    else:
        selection_status = "no_model_beats_corrected_selection_baseline"

    return {
        "version": VERSION,
        "benchmark": {"locked": True, "status": "ready", "selection_regime": BENCHMARK_REGIME,
                      "benchmark_collections": benchmark["benchmark_collections"],
                      "selection_collections": benchmark["selection_collections"],
                      "eligible_collection_count": benchmark["eligible_collection_count"],
                      "collection_count": len(benchmark["benchmark_collections"]),
                      "source_count": len({e[3] for e in selection_events + final_events}),
                      "selection_events": len(selection_events),
                      "final_events": len(final_events),
                      # the frozen evaluation events themselves (captured once)
                      "selection_event_snapshot": [list(e) for e in selection_events],
                      "final_event_snapshot": [list(e) for e in final_events],
                      "fingerprint": plaus_seed},
        "training": {"events": len(train_events),
                     "source_count": len({e[3] for e in train_events}),
                     "verb_vocabulary": len(baseline.verb_freq)},
        "selected": selected,
        "selected_model_id": selected_model_id,
        "selection_status": selection_status,
        "selected_evaluation": selected or _empty_selected(),
        "best_rejected_candidate": None if selected else best_candidate,
        "evaluations": evaluations,
        "final_attempt": final_attempt,
        "final_attempt_history": final_history[-50:],
        "final_query_budget": FINAL_QUERY_BUDGET,
        "final_queries_used": len(final_history),
        "counterexamples": counterexamples[:200],
        "next_learning_target": target,
        "learning_curve": learning_curve,
        "learning_curve_trend": learning_curve_trend,
        "corrupters": list(DEFAULT_CORRUPTERS),
        "emitted_events": emitted_events,
        "invariants": ["benchmark_collections_are_locked", "benchmark_never_trains",
                       "selection_and_final_are_collection_disjoint",
                       "familywise_error_is_bonferroni_corrected",
                       "final_holdout_queries_are_finite_milestone_gated_and_corrected",
                       "baseline_must_be_beaten_on_selection_and_final"],
        "limitations": ["within-event structure only; next-event / causal succession "
                        "carries no measurable signal in this corpus",
                        "parsed clauses are heuristic observations, not semantic truth"],
    }


def _postponed_report(events, benchmark: dict, previous: dict) -> dict:
    return {
        "version": VERSION,
        "benchmark": {"locked": False, "status": "insufficient_benchmark_events",
                      "selection_regime": BENCHMARK_REGIME,
                      "eligible_collection_count": benchmark.get("eligible_collection_count", 0),
                      "candidate_selection_events": benchmark.get("candidate_selection_events", 0),
                      "candidate_final_events": benchmark.get("candidate_final_events", 0),
                      "candidate_train_events": benchmark.get("candidate_train_events", 0),
                      "minimum_selection_events": MINIMUM_EVALUATION_TOTAL,
                      "minimum_final_events": MINIMUM_EVALUATION_TOTAL,
                      "minimum_train_events": MINIMUM_TRAIN_EVENTS,
                      "collection_count": 0, "source_count": 0,
                      "selection_events": 0, "final_events": 0, "fingerprint": None},
        "training": {"events": len(events), "source_count": len({e[3] for e in events}),
                     "verb_vocabulary": len({e[1] for e in events})},
        "selected": None, "selected_model_id": "frequency_baseline",
        "selection_status": "benchmark_not_ready",
        "selected_evaluation": _empty_selected(),
        "best_rejected_candidate": None, "evaluations": [],
        "final_attempt": None, "final_attempt_history": list(previous.get("final_attempt_history", [])),
        "final_query_budget": FINAL_QUERY_BUDGET, "final_queries_used": 0,
        "counterexamples": [],
        "next_learning_target": {"seed": "simple animal story",
                                 "reason": "collect independent eligible collections before evaluation"},
        "learning_curve": list(previous.get("learning_curve", [])),
        "learning_curve_trend": previous.get("learning_curve_trend", "insufficient_data"),
        "corrupters": list(DEFAULT_CORRUPTERS), "emitted_events": [],
        "invariants": ["benchmark deliberately postponed until its split would carry enough events"],
        "limitations": ["no evaluation until the frozen benchmark is ready"],
    }


def _counterexamples(fitted, baseline, selection_events, best_candidate) -> list[dict]:
    if not best_candidate or best_candidate["task"] != "verb_cloze":
        return []
    model = fitted[best_candidate["model_id"]]
    _, trials = evaluate_cloze(model, baseline, selection_events)
    misses = [t for t in trials if not t["correct"]]
    order = Counter((t["subject"], t["object"], t["predicted"], t["observed"]) for t in misses)
    ranked = sorted(misses, key=lambda t: -order[(t["subject"], t["object"],
                                                  t["predicted"], t["observed"])])
    return ranked


def classify_trend(points: list[dict], key: str, window: int = 10, min_delta: float = 0.0) -> str:
    """Mean of the older half of the window vs the newer half -> improving /
    flat / declining / insufficient_data.  Coarse observability signal only; it
    never feeds the significance gates."""
    tail = [point.get(key, 0) for point in points[-window:] if point.get(key) is not None]
    if len(tail) < 4:
        return "insufficient_data"
    middle = len(tail) // 2
    delta = sum(tail[middle:]) / (len(tail) - middle) - sum(tail[:middle]) / middle
    if delta > min_delta:
        return "improving"
    if delta < -min_delta:
        return "declining"
    return "flat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".local")
    args = parser.parse_args()
    verified = json.loads((args.runtime / "verified-experience.json").read_text(encoding="utf-8"))
    output = args.runtime / "event-structure.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    report = train_and_evaluate(verified, previous)
    report.pop("emitted_events", None)
    output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8")
    print(json.dumps({"selection_status": report["selection_status"],
                      "selected_model_id": report["selected_model_id"],
                      "benchmark_locked": report["benchmark"]["locked"],
                      "trend": report["learning_curve_trend"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
