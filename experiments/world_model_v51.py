#!/usr/bin/env python3
"""A source-held-out, stateful experience model for real reading material.

This is deliberately symbolic and inspectable.  It does not claim that a parsed
sentence is understood: every frame retains its source sentence and confidence.
"""

from __future__ import annotations

import hashlib
import math
import re
import urllib.parse
from collections import Counter, defaultdict

from narrative_event_v29 import NarrativeEventExtractor, WORD


NEGATION = {"not", "never", "no", "neither", "nor", "without", "cannot", "can't"}
GOAL_WORDS = {"want", "wanted", "wish", "wished", "hope", "hoped", "try", "tried",
              "intend", "intended", "seek", "sought", "need", "needed"}
RESULT_MARKERS = {"so", "therefore", "thus", "hence", "consequently"}
CONTRAST_MARKERS = {"but", "however", "although", "though", "yet"}
STATE_AUX = {"am", "are", "is", "was", "were", "be", "been", "being", "have", "has", "had"}
STOP_VALUE = {"a", "an", "the", "to", "of", "and", "or", "but", "that", "this", "his",
              "her", "their", "its", "very", "then", "there"}
NARRATIVE_STEMS = ("fabl", "fairy", "folk", "tale", "aesop", "animal")
NARRATIVE_CHARACTERS = {"mouse", "fox", "lion", "hare", "wolf", "boy", "girl"}
REFERENCE_HINTS = {"dictionary", "encyclopaedia", "encyclopedia", "britannica", "journal",
                   "notes", "history"}
CONTRADICTORY_STATE_GROUPS = (
    frozenset({"hungry", "full", "fed", "sated", "satisfied"}),
    frozenset({"tired", "rested", "refreshed", "energetic"}),
    frozenset({"afraid", "scared", "frightened", "brave", "calm", "fearless"}),
    frozenset({"sad", "unhappy", "miserable", "happy", "glad", "cheerful"}),
    frozenset({"cold", "cool", "warm", "hot"}),
    frozenset({"lost", "safe", "found"}),
    frozenset({"thirsty", "quenched"}),
    frozenset({"awake", "asleep"}),
    frozenset({"sick", "ill", "healthy", "well", "better"}),
    frozenset({"poor", "rich", "wealthy"}),
    frozenset({"weak", "strong"}),
    frozenset({"empty", "full"}),
    frozenset({"alive", "dead"}),
)
BENCHMARK_REGIME = "collection_disjoint_preregistered_v2"
MIN_BENCHMARK_GROUPS = 20
FAMILY_ALPHA = .05
# Shared with passes_gain_gate: an evaluation (or a training pool) below this
# many predictive pairs is too small to trust, whether it's a selection/final
# holdout or the training side of the collection-disjoint split.
MINIMUM_EVALUATION_TOTAL = 15


def source_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def collection_key(url: str) -> str:
    """Keep pages from one work/anthology on the same side of a split."""
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.unquote(parsed.path).rstrip("/")
    marker = "/wiki/"
    if marker in path:
        title = path.split(marker, 1)[1]
        title = title.rsplit("/", 1)[0] if "/" in title else title
        return f"{parsed.netloc}/wiki/{title}"
    # Generic test/local sources may encode a collection as the parent path.
    parent = path.rsplit("/", 1)[0] if path.count("/") > 1 else path
    return f"{parsed.netloc}{parent}"


def narrative_source(url: str) -> bool:
    tokens = re.findall(r"[a-z]+", urllib.parse.unquote(url).lower())
    return (not set(tokens) & REFERENCE_HINTS
            and (bool(set(tokens) & NARRATIVE_CHARACTERS)
                 or any(token.startswith(hint) for token in tokens for hint in NARRATIVE_STEMS)))


def narrative_sequence(sequence: dict) -> bool:
    frames = sequence.get("frames", [])
    actors = Counter(frame.get("actor") for frame in frames if frame.get("actor"))
    actions = {frame.get("action") for frame in frames if frame.get("action")}
    return len(frames) >= 4 and bool(actors) and actors.most_common(1)[0][1] >= 3 and len(actions) >= 2


def _states_conflict(first: str, second: str) -> bool:
    """True only when two conditions are mutually exclusive (e.g. hungry vs. full)."""
    return first != second and any(first in group and second in group
                                   for group in CONTRADICTORY_STATE_GROUPS)


def _value_after(words: list[str], index: int) -> str | None:
    segment = words[index + 1:index + 6]
    if segment[:2] == ["no", "longer"]:
        segment = segment[2:]
    elif segment[:3] == ["not", "any", "more"]:
        segment = segment[3:]
    return next((word for word in segment
                 if word not in STOP_VALUE and word not in NEGATION), None)


def parse_frame(sentence: str, recent_actor: str | None = None) -> dict | None:
    """Extract an auditable state/goal/action/result frame from a simple clause."""
    parsed = NarrativeEventExtractor("developmental_grounded_24").extract(
        sentence, recent_actor)
    words = [word.lower() for word in WORD.findall(sentence)]
    if parsed.accepted and parsed.event:
        actor, action, obj = parsed.event.subject, parsed.event.action, parsed.event.object
        confidence = parsed.quality
    else:
        # State-only clauses are observations even when they contain no action.
        copula = next((index for index, word in enumerate(words) if word in STATE_AUX), None)
        if copula is None or copula == 0:
            return None
        actor_candidates = [word for word in words[:copula] if word not in STOP_VALUE]
        value = _value_after(words, copula)
        if not actor_candidates or not value:
            return None
        actor = recent_actor if actor_candidates[-1] in {"he", "she", "it", "they"} else actor_candidates[-1]
        if not actor:
            return None
        action, obj, confidence = "state", value, .8
    states, goals = [], []
    for index, word in enumerate(words):
        if word in STATE_AUX:
            value = _value_after(words, index)
            if value and value != action:
                states.append(value)
        if word in GOAL_WORDS:
            value = _value_after(words, index)
            if value:
                goals.append(value)
    # An infinitive following the observed action is a prospective goal, not its result.
    try:
        action_index = words.index(action)
    except ValueError:
        action_index = -1
    if action_index >= 0 and "to" in words[action_index + 1:]:
        to_index = words.index("to", action_index + 1)
        value = _value_after(words, to_index)
        if value:
            goals.append(value)
    polarity = "negative" if any(word in NEGATION for word in words) else "positive"
    discourse = ("result" if any(word in RESULT_MARKERS for word in words) else
                 "contrast" if any(word in CONTRAST_MARKERS for word in words) else "continuation")
    state_polarity = "negative" if any(word in NEGATION for word in words) else "positive"
    return {"actor": actor, "states": sorted(set(states))[:3],
            "state_updates": [{"slot": "condition", "value": value,
                               "polarity": state_polarity} for value in sorted(set(states))[:3]],
            "goals": sorted(set(goals))[:3],
            "action": action, "object": obj.split("_")[0] if obj else "none",
            "polarity": polarity, "discourse": discourse,
            "sentence": sentence, "parser_confidence": confidence}


def build_sequences(audit: dict) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in audit.get("records", {}).values():
        if item.get("curriculum_admitted") is True and item.get("source_url") and item.get("sentence"):
            grouped[item["source_url"]].append(item)
    sequences = {}
    for url, records in grouped.items():
        frames, recent_actor = [], None
        # Non-conflicting conditions (e.g. "hungry" and "brave") accumulate; only an
        # explicit negation or a mutually exclusive condition removes an existing one.
        actor_states: dict[str, list[str]] = defaultdict(list)
        actor_goals: dict[str, list[str]] = defaultdict(list)
        for item in sorted(records, key=lambda row: row.get("source_position", 0)):
            frame = parse_frame(item["sentence"], recent_actor)
            if frame:
                frame["position"] = item.get("source_position", 0)
                superseded = []
                current = actor_states[frame["actor"]]
                for update in frame.get("state_updates", []):
                    value = update["value"]
                    if update["polarity"] == "negative":
                        if value in current:
                            current.remove(value)
                            superseded.append(value)
                    else:
                        for existing in list(current):
                            if _states_conflict(existing, value):
                                current.remove(existing)
                                superseded.append(existing)
                        if value not in current:
                            current.append(value)
                actor_goals[frame["actor"]] = list(dict.fromkeys(
                    actor_goals[frame["actor"]] + frame["goals"]))[-4:]
                frame["known_states"] = list(current)
                frame["superseded_states"] = superseded
                frame["active_goals"] = list(actor_goals[frame["actor"]])
                frames.append(frame)
                recent_actor = frame["actor"]
            else:
                recent_actor = None
        if len(frames) >= 3:
            sequences[url] = {"source_url": url, "source_id": source_key(url), "frames": frames}
    return sequences


def classify_trend(points: list[dict], key: str, window: int = 10, min_delta: float = 0.0) -> str:
    """A lightweight improving/flat/declining read on the tail of a learning curve.

    Compares the mean of the older half of the window against the newer half,
    rather than just the two endpoints, so a single noisy point can't flip the
    verdict. Not a substitute for the fixed-benchmark significance gates --
    this is a coarse trend for humans and update_autonomy_state to look at
    alongside those, not a pass/fail criterion on its own.
    """
    tail = [point.get(key, 0) for point in points[-window:] if point.get(key) is not None]
    if len(tail) < 4:
        return "insufficient_data"
    middle = len(tail) // 2
    older_avg = sum(tail[:middle]) / middle
    newer_avg = sum(tail[middle:]) / (len(tail) - middle)
    delta = newer_avg - older_avg
    if delta > min_delta:
        return "improving"
    if delta < -min_delta:
        return "declining"
    return "flat"


def examples_from(sequences: dict[str, dict], sources: set[str]) -> list[tuple[list[dict], dict, str]]:
    examples = []
    for url in sorted(sources):
        frames = sequences.get(url, {}).get("frames", [])
        for index in range(1, len(frames)):
            # Actor changes are narrative adjacency, not an actor-state transition.
            if frames[index - 1]["actor"] != frames[index]["actor"]:
                continue
            examples.append((frames[max(0, index - 2):index], frames[index], url))
    return examples


def balanced_source_split(examples: list[tuple[list[dict], dict, str]]) -> set[str]:
    """Divide a benchmark's whole sources between its selection and final halves,
    keeping each source's examples together (never split across the two)."""
    counts = Counter(url for _, _, url in examples)
    selection, totals = set(), [0, 0]
    for url, count in sorted(counts.items(), key=lambda item: (-item[1], source_key(item[0]))):
        side = 0 if totals[0] <= totals[1] else 1
        if side == 0:
            selection.add(url)
        totals[side] += count
    return selection


def _candidate_benchmark(sequences: dict[str, dict], minimum_total: int,
                         minimum_train_examples: int) -> dict:
    """Build the ranked candidate benchmark and judge readiness by whether its
    selection/final split, and the remaining training pool, actually carry
    enough predictive pairs -- not by how many collections exist. A handful of
    long collections can satisfy this before thirty short ones would, and
    thirty very short ones might still not."""
    groups: dict[str, list[str]] = defaultdict(list)
    for url, sequence in sequences.items():
        if narrative_source(url) and narrative_sequence(sequence):
            groups[collection_key(url)].append(url)
    empty = {"ready": False, "benchmark_urls": [], "eligible_collection_count": len(groups),
             "selection_examples": 0, "final_examples": 0, "train_examples": 0}
    if not groups:
        return empty
    ranked_groups = sorted(groups, key=lambda group: hashlib.sha256(
        f"fixed-world-benchmark:{group}".encode()).hexdigest())
    selected_groups = set(ranked_groups[:min(40, max(MIN_BENCHMARK_GROUPS,
                                                len(ranked_groups) // 3))])
    benchmark_urls = sorted(url for group in selected_groups for url in groups[group])
    train_sources = {url for group in ranked_groups if group not in selected_groups
                     for url in groups[group]}
    raw_benchmark = examples_from(sequences, set(benchmark_urls))
    selection_sources = balanced_source_split(raw_benchmark)
    selection_examples = sum(1 for _, _, url in raw_benchmark if url in selection_sources)
    final_examples = len(raw_benchmark) - selection_examples
    train_examples = len(examples_from(sequences, train_sources))
    ready = (selection_examples >= minimum_total and final_examples >= minimum_total
            and train_examples >= minimum_train_examples)
    return {"ready": ready, "benchmark_urls": benchmark_urls,
            "eligible_collection_count": len(groups), "selection_examples": selection_examples,
            "final_examples": final_examples, "train_examples": train_examples}


def choose_benchmark_sources(sequences: dict[str, dict], previous: dict | None = None,
                             minimum_total: int = MINIMUM_EVALUATION_TOTAL,
                             minimum_train_examples: int = MINIMUM_EVALUATION_TOTAL) -> list[str]:
    """Freeze an eligible collection-disjoint set, or postpone without fallback."""
    previous = previous or {}
    prior_benchmark = previous.get("benchmark", {})
    locked = prior_benchmark.get("source_urls")
    if locked and prior_benchmark.get("selection_regime") == BENCHMARK_REGIME:
        return list(locked)
    candidate = _candidate_benchmark(sequences, minimum_total, minimum_train_examples)
    return candidate["benchmark_urls"] if candidate["ready"] else []


def context_key(history: list[dict], mode: str) -> str:
    current = history[-1]
    if mode.startswith("shape"):
        components = ["has_state:" + str(bool(current.get("known_states", current["states"]))),
                      "has_goal:" + str(bool(current.get("active_goals", current["goals"]))),
                      f"polarity:{current['polarity']}", f"discourse:{current['discourse']}"]
        if "history" in mode:
            prior = history[-2] if len(history) > 1 else None
            components.append("prior_shape:" + ("state" if prior and prior.get("states") else
                                                  "goal" if prior and prior.get("goals") else
                                                  "action" if prior else "none"))
        return "|".join(components)
    components = [f"action:{current['action']}"]
    if "state" in mode:
        values = current.get("known_states", current["states"])
        components.append("state:" + (values[-1] if values else "none"))
    if "goal" in mode:
        values = current.get("active_goals", current["goals"])
        components.append("goal:" + (values[-1] if values else "none"))
    if "result" in mode:
        components.extend((f"polarity:{current['polarity']}", f"discourse:{current['discourse']}"))
    if "history" in mode:
        components.append("previous:" + (history[-2]["action"] if len(history) > 1 else "none"))
    return "|".join(components)


MODES = ("action", "action_state", "action_goal", "action_state_goal",
         "action_state_goal_result", "action_state_goal_result_history",
         "shape", "shape_history")

# The fixed benchmark's final split may be re-queried a small, bounded number of
# times -- once when it first unlocks and again each time the training set doubles
# -- so an early, data-starved miss is not a permanent verdict.  Every one of these
# milestone queries is Bonferroni-corrected against by FINAL_QUERY_BUDGET.
FINAL_QUERY_BUDGET = 5
FINAL_TRAIN_GROWTH_FACTOR = 2


def passes_gain_gate(evaluation: dict, minimum_total: int = MINIMUM_EVALUATION_TOTAL,
                     alpha: float = FAMILY_ALPHA / (len(MODES) * 2)) -> bool:
    required = max(3, (evaluation.get("total", 0) + 9) // 10)
    return (evaluation.get("total", 0) >= minimum_total
            and evaluation.get("lift", 0) >= required
            and evaluation.get("coverage", 0.0) >= .1
            and evaluation.get("one_sided_sign_p", 1.0) <= alpha)


def train_and_evaluate(audit: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    sequences = build_sequences(audit)
    benchmark_urls = choose_benchmark_sources(sequences, previous)
    benchmark = set(benchmark_urls)
    benchmark_groups = {collection_key(url) for url in benchmark}
    train_sources = {url for url in sequences if collection_key(url) not in benchmark_groups}
    train = examples_from(sequences, train_sources)
    frozen_examples = previous.get("benchmark_examples")

    if not benchmark_urls:
        # choose_benchmark_sources already judged readiness by example count; redo
        # that (cheap) computation here purely to report what it found.
        candidate = _candidate_benchmark(sequences, MINIMUM_EVALUATION_TOTAL,
                                         MINIMUM_EVALUATION_TOTAL)
        return {"version": 51,
                "benchmark": {"locked": False, "status": "insufficient_benchmark_examples",
                    "selection_regime": BENCHMARK_REGIME, "source_urls": [], "source_count": 0,
                    "eligible_collection_count": candidate["eligible_collection_count"],
                    "candidate_selection_examples": candidate["selection_examples"],
                    "candidate_final_examples": candidate["final_examples"],
                    "candidate_train_examples": candidate["train_examples"],
                    "minimum_selection_examples": MINIMUM_EVALUATION_TOTAL,
                    "minimum_final_examples": MINIMUM_EVALUATION_TOTAL,
                    "minimum_train_examples": MINIMUM_EVALUATION_TOTAL,
                    "selection_examples": 0, "final_examples": 0, "examples": 0,
                    "fingerprint": None},
                "training": {"source_count": len(sequences),
                             "examples": len(examples_from(sequences, set(sequences))),
                             "source_urls": sorted(sequences)},
                "benchmark_examples": [], "selected_mode": "frequency_baseline",
                "selection_status": "benchmark_not_ready",
                "selected_evaluation": {"model_id": "frequency_baseline", "task": "none",
                    "mode": "frequency_baseline", "correct": 0, "baseline_correct": 0,
                    "total": 0, "coverage": 0.0, "lift": 0},
                "evaluations": [], "reusable_rules": [], "counterexamples": [],
                "next_learning_target": {"seed": "simple animal story",
                    "reason": "collect independent eligible narrative collections before evaluation"},
                "revision_history": list(previous.get("revision_history", []))[-200:],
                "learning_curve": list(previous.get("learning_curve", [])),
                "learning_curve_trend": previous.get("learning_curve_trend", "insufficient_data"),
                "invariants": ["no_non_narrative_benchmark_fallback",
                    "minimum_training_examples_preserved", "collection_disjoint_split"],
                "limitations": ["benchmark deliberately postponed until its selection/final split "
                                "and remaining training pool would each carry enough examples"]}

    if (frozen_examples and previous.get("benchmark", {}).get("selection_regime")
            == BENCHMARK_REGIME):
        legacy = [(item["history"], item["outcome"], item["source_url"])
                  for item in frozen_examples]
        migrated_selection = balanced_source_split(legacy)
        locked = [(item["history"], item["outcome"], item["source_url"],
                   item.get("split") or ("selection" if item["source_url"] in migrated_selection
                                          else "final"))
                  for item in frozen_examples]
    else:
        raw_benchmark = examples_from(sequences, benchmark)
        selection_sources = balanced_source_split(raw_benchmark)
        locked = [(history, outcome, url,
                   "selection" if url in selection_sources else "final")
                  for history, outcome, url in raw_benchmark]
    selection_test = [(history, outcome, url) for history, outcome, url, split in locked
                      if split == "selection"]
    final_test = [(history, outcome, url) for history, outcome, url, split in locked
                  if split == "final"]

    def outcome_label(frame: dict, task: str) -> str:
        if task == "exact_action":
            return frame["action"]
        return "|".join(("state" if frame.get("states") else "no_state",
                         "goal" if frame.get("goals") else "no_goal",
                         frame.get("polarity", "positive"),
                         frame.get("discourse", "continuation")))

    evaluations, models = [], {}

    def evaluate(rules: dict, fallback: str | None, task: str,
                 examples: list[tuple[list[dict], dict, str]], mode: str) -> tuple[dict, list[dict]]:
        correct = baseline = covered = wins = losses = 0
        trials = []
        for history, outcome, url in examples:
            observed = outcome_label(outcome, task)
            key = context_key(history, mode)
            predicted = rules[key].most_common(1)[0][0] if key in rules else fallback
            correct += predicted == observed
            baseline += fallback == observed
            wins += predicted == observed and fallback != observed
            losses += predicted != observed and fallback == observed
            covered += key in rules
            trials.append({"source_id": source_key(url), "context": key,
                           "predicted": predicted, "observed": observed,
                           "baseline": fallback, "correct": predicted == observed,
                           "baseline_correct": fallback == observed,
                           "query_terms": list(dict.fromkeys(
                               [frame["action"] for frame in history]
                               + [value for frame in history for value in
                                  frame.get("known_states", frame.get("states", []))]
                               + [value for frame in history for value in
                                  frame.get("active_goals", frame.get("goals", []))]))[:4]})
        total = len(examples)
        discordant = wins + losses
        sign_p = (sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) /
                  (2 ** discordant) if discordant else 1.0)
        return ({"correct": correct, "baseline_correct": baseline, "total": total,
                 "coverage": round(covered / total, 4) if total else 0.0,
                 "lift": correct - baseline, "paired_wins": wins, "paired_losses": losses,
                 "one_sided_sign_p": round(sign_p, 6)}, trials)

    for task in ("exact_action", "experience_transition"):
        global_outcomes = Counter(outcome_label(outcome, task) for _, outcome, _ in train)
        fallback = global_outcomes.most_common(1)[0][0] if global_outcomes else None
        for mode in MODES:
            rules: dict[str, Counter[str]] = defaultdict(Counter)
            for history, outcome, _ in train:
                rules[context_key(history, mode)][outcome_label(outcome, task)] += 1
            selection, selection_trials = evaluate(rules, fallback, task, selection_test, mode)
            model_id = f"{task}:{mode}"
            evaluations.append({"model_id": model_id, "task": task, "mode": mode,
                                "selection": selection, **selection})
            models[model_id] = {"rules": rules, "selection_trials": selection_trials}
    finalists = [item for item in evaluations if passes_gain_gate(item["selection"])]
    candidate = max(finalists, key=lambda item: (item["selection"]["lift"],
                    item["selection"]["correct"], item["selection"]["coverage"],
                    -MODES.index(item["mode"]), item["task"] == "exact_action"), default=None)
    prior_final = previous.get("final_attempt")
    final_history = list(previous.get("final_attempt_history", []))
    if prior_final and not final_history:
        # Migrate a pre-existing single "one-time" attempt into the bounded history.
        final_history = [prior_final]
    final_attempt = prior_final
    final_alpha = FAMILY_ALPHA / (len(MODES) * 2 * FINAL_QUERY_BUDGET)
    last_final_train = final_history[-1].get("training_examples", 0) if final_history else 0
    milestone_reached = (not final_history
                         or len(train) >= max(1, last_final_train) * FINAL_TRAIN_GROWTH_FACTOR)
    budget_left = len(final_history) < FINAL_QUERY_BUDGET
    selected = None
    if candidate and milestone_reached and budget_left:
        final, final_trials = evaluate(models[candidate["model_id"]]["rules"],
            Counter(outcome_label(outcome, candidate["task"]) for _, outcome, _ in train
                    ).most_common(1)[0][0] if train else None,
            candidate["task"], final_test, candidate["mode"])
        final_attempt = {"model_id": candidate["model_id"], "evaluation": final,
                         "trials": final_trials,
                         "training_examples": len(train),
                         "query_index": len(final_history) + 1,
                         "training_fingerprint": hashlib.sha256("\n".join(sorted(train_sources)).encode()
                                                                ).hexdigest()[:16]}
        final_history = final_history + [final_attempt]
    if candidate and final_attempt and final_attempt.get("model_id") == candidate["model_id"]:
        if passes_gain_gate(final_attempt.get("evaluation", {}), alpha=final_alpha):
            selected = {**candidate, "final": final_attempt["evaluation"],
                        **final_attempt["evaluation"]}
    best_candidate = max(evaluations, key=lambda item: (item["selection"]["lift"],
                         item["selection"]["coverage"]), default=None)
    # Track the best candidate's accuracy/lift/coverage against training size even
    # when it never clears the strict significance gates: a single pass/fail p-value
    # hides whether more data is trending toward or away from a real signal.
    best_selection = (best_candidate or {}).get("selection", {})
    learning_curve = list(previous.get("learning_curve", []))
    curve_point = {"training_examples": len(train),
                   "model_id": (best_candidate or {}).get("model_id"),
                   "accuracy": (round(best_selection["correct"] / best_selection["total"], 4)
                               if best_selection.get("total") else 0.0),
                   "lift": best_selection.get("lift", 0),
                   "coverage": best_selection.get("coverage", 0.0),
                   "one_sided_sign_p": best_selection.get("one_sided_sign_p", 1.0),
                   "gate_passed": bool(selected)}
    if not learning_curve or learning_curve[-1]["training_examples"] != curve_point["training_examples"]:
        learning_curve.append(curve_point)
    learning_curve = learning_curve[-200:]
    learning_curve_trend = classify_trend(learning_curve, "lift", window=10, min_delta=1)
    selected_mode = selected["model_id"] if selected else "frequency_baseline"
    old_mode = previous.get("selected_mode")
    revisions = list(previous.get("revision_history", []))
    if old_mode and old_mode != selected_mode:
        revisions.append({"before": old_mode, "after": selected_mode,
                          "reason": "fixed unseen-source evaluation changed model ranking",
                          "benchmark_fingerprint": hashlib.sha256(
                              "\n".join(benchmark_urls).encode()).hexdigest()[:16]})
    diagnostic_id = selected_mode if selected else (best_candidate or {}).get("model_id")
    chosen_model = models.get(diagnostic_id, {"rules": {}, "selection_trials": []})
    reusable = []
    for key, outcomes in (chosen_model["rules"].items() if selected else []):
        prediction, support = outcomes.most_common(1)[0]
        if support >= 3 and support / sum(outcomes.values()) >= .7:
            reusable.append({"context": key, "prediction": prediction, "support": support,
                             "confidence": round(support / sum(outcomes.values()), 4)})
    counterexamples = [trial for trial in chosen_model["selection_trials"] if not trial["correct"]]
    patterns = Counter((trial["context"], trial["predicted"], trial["observed"])
                       for trial in counterexamples)
    target_history = dict(previous.get("target_history", {}))
    ranked_patterns = sorted(patterns, key=lambda pattern: (
        target_history.get("|".join(str(value) for value in pattern), 0),
        -patterns[pattern], hashlib.sha256(repr(pattern).encode()).hexdigest()))
    target = None
    if not selected and ranked_patterns:
        pattern = ranked_patterns[0]
        exemplar = next(trial for trial in counterexamples
                        if (trial["context"], trial["predicted"], trial["observed"]) == pattern)
        pattern_key = "|".join(str(value) for value in pattern)
        target_history[pattern_key] = target_history.get(pattern_key, 0) + 1
        target = {"seed": " ".join(exemplar.get("query_terms", [])) or "simple action story",
                  "reason": "seek a rotated independent boundary case for a frequent failure",
                  "failure_pattern": pattern_key, "failure_count": patterns[pattern]}
    queries_used = len(final_history)
    if selected:
        selection_status = "accepted_final_gain"
    elif candidate and queries_used >= FINAL_QUERY_BUDGET:
        selection_status = "final_holdout_query_budget_exhausted_no_confirmed_gain"
    elif candidate and not milestone_reached:
        selection_status = "awaiting_training_growth_for_next_final_query"
    elif candidate:
        selection_status = "candidate_failed_final_query"
    else:
        selection_status = "no_model_beats_corrected_selection_baseline"
    return {"version": 51,
            "benchmark": {"locked": True, "status": "ready", "source_urls": benchmark_urls,
                          "selection_regime": BENCHMARK_REGIME,
                          "collection_count": len(benchmark_groups),
                          "source_count": len(benchmark_urls),
                          "selection_examples": len(selection_test),
                          "final_examples": len(final_test),
                          "examples": len(selection_test) + len(final_test),
                          "fingerprint": hashlib.sha256("\n".join(benchmark_urls).encode()).hexdigest()[:16]},
            "training": {"source_count": len(train_sources), "examples": len(train),
                         "source_urls": sorted(train_sources)},
            "benchmark_examples": [
                {"history": history, "outcome": outcome, "source_url": url, "split": split}
                for history, outcome, url, split in locked],
            "frame_schema": ["actor", "states", "goals", "action", "object", "polarity", "discourse"],
            "selected_mode": selected_mode,
            "selection_status": selection_status,
            "best_rejected_candidate": None if selected else best_candidate,
            "selected_evaluation": selected or {"model_id": "frequency_baseline",
                "task": "none", "mode": "frequency_baseline",
                "correct": 0, "baseline_correct": 0, "total": 0, "coverage": 0.0, "lift": 0},
            "final_attempt": final_attempt,
            "final_attempt_history": final_history[-50:],
            "final_query_budget": FINAL_QUERY_BUDGET,
            "final_queries_used": queries_used,
            "evaluations": evaluations, "reusable_rules": reusable[:2000],
            "counterexamples": counterexamples[-1000:],
            "counterexample_patterns": [{"pattern": "|".join(str(value) for value in pattern),
                 "count": count} for pattern, count in patterns.most_common(100)],
            "target_history": target_history, "next_learning_target": target,
            "revision_history": revisions[-200:],
            "learning_curve": learning_curve, "learning_curve_trend": learning_curve_trend,
            "invariants": ["benchmark_sources_are_locked", "benchmark_examples_are_frozen",
                           "benchmark_sources_never_train", "selection_and_final_are_disjoint",
                           "whole_collection_split", "familywise_error_is_bonferroni_corrected",
                           "final_holdout_queries_are_finite_milestone_gated_and_bonferroni_corrected",
                           "baseline_must_be_beaten_twice", "paired_improvement_must_pass_sign_test_twice",
                           "original_sentence_retained"],
            "limitations": ["frames remain heuristic observations, not semantic truth",
                            "only explicit simple-clause experiences receive predictive credit"]}
