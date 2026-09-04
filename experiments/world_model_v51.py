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


def source_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


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


def _value_after(words: list[str], index: int) -> str | None:
    return next((word for word in words[index + 1:index + 5]
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
    return {"actor": actor, "states": sorted(set(states))[:3], "goals": sorted(set(goals))[:3],
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
        actor_states: dict[str, list[str]] = defaultdict(list)
        actor_goals: dict[str, list[str]] = defaultdict(list)
        for item in sorted(records, key=lambda row: row.get("source_position", 0)):
            frame = parse_frame(item["sentence"], recent_actor)
            if frame:
                frame["position"] = item.get("source_position", 0)
                actor_states[frame["actor"]] = list(dict.fromkeys(
                    actor_states[frame["actor"]] + frame["states"]))[-4:]
                actor_goals[frame["actor"]] = list(dict.fromkeys(
                    actor_goals[frame["actor"]] + frame["goals"]))[-4:]
                frame["known_states"] = list(actor_states[frame["actor"]])
                frame["active_goals"] = list(actor_goals[frame["actor"]])
                frames.append(frame)
                recent_actor = frame["actor"]
            else:
                recent_actor = None
        if len(frames) >= 3:
            sequences[url] = {"source_url": url, "source_id": source_key(url), "frames": frames}
    return sequences


def choose_benchmark_sources(sequences: dict[str, dict], previous: dict | None = None) -> list[str]:
    """Freeze an evaluation set once; later observations can never migrate into it."""
    previous = previous or {}
    locked = previous.get("benchmark", {}).get("source_urls")
    if locked:
        return list(locked)
    eligible = [url for url, sequence in sequences.items()
                if narrative_source(url) and narrative_sequence(sequence)]
    pool = eligible if len(eligible) >= 10 else list(sequences)
    ranked = sorted(pool, key=lambda url: hashlib.sha256(
        f"fixed-world-benchmark:{url}".encode()).hexdigest())
    return ranked[:min(40, max(20, len(ranked) // 3))]


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


def passes_gain_gate(evaluation: dict, minimum_total: int = 15) -> bool:
    required = max(3, (evaluation.get("total", 0) + 9) // 10)
    return (evaluation.get("total", 0) >= minimum_total
            and evaluation.get("lift", 0) >= required
            and evaluation.get("coverage", 0.0) >= .1
            and evaluation.get("one_sided_sign_p", 1.0) <= .1)


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


def train_and_evaluate(audit: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    sequences = build_sequences(audit)
    benchmark_urls = choose_benchmark_sources(sequences, previous)
    benchmark = set(benchmark_urls)
    train_sources = set(sequences) - benchmark
    train = examples_from(sequences, train_sources)
    frozen_examples = previous.get("benchmark_examples")

    def balanced_source_split(examples: list[tuple[list[dict], dict, str]]) -> set[str]:
        counts = Counter(url for _, _, url in examples)
        selection, totals = set(), [0, 0]
        for url, count in sorted(counts.items(), key=lambda item: (-item[1], source_key(item[0]))):
            side = 0 if totals[0] <= totals[1] else 1
            if side == 0:
                selection.add(url)
            totals[side] += count
        return selection

    if frozen_examples:
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
            final, final_trials = evaluate(rules, fallback, task, final_test, mode)
            model_id = f"{task}:{mode}"
            evaluations.append({"model_id": model_id, "task": task, "mode": mode,
                                "selection": selection, "final": final, **final})
            models[model_id] = {"rules": rules, "trials": final_trials,
                                "selection_trials": selection_trials}
    finalists = [item for item in evaluations if passes_gain_gate(item["selection"])]
    candidate = max(finalists, key=lambda item: (item["selection"]["lift"],
                    item["selection"]["correct"], item["selection"]["coverage"],
                    -MODES.index(item["mode"]), item["task"] == "exact_action"), default=None)
    selected = candidate if candidate and passes_gain_gate(candidate["final"]) else None
    best_candidate = max(evaluations, key=lambda item: (item["selection"]["lift"],
                         item["final"]["lift"], item["selection"]["coverage"]), default=None)
    selected_mode = selected["model_id"] if selected else "frequency_baseline"
    old_mode = previous.get("selected_mode")
    revisions = list(previous.get("revision_history", []))
    if old_mode and old_mode != selected_mode:
        revisions.append({"before": old_mode, "after": selected_mode,
                          "reason": "fixed unseen-source evaluation changed model ranking",
                          "benchmark_fingerprint": hashlib.sha256(
                              "\n".join(benchmark_urls).encode()).hexdigest()[:16]})
    diagnostic_id = selected_mode if selected else (best_candidate or {}).get("model_id")
    chosen_model = models.get(diagnostic_id, {"rules": {}, "trials": []})
    reusable = []
    for key, outcomes in (chosen_model["rules"].items() if selected else []):
        prediction, support = outcomes.most_common(1)[0]
        if support >= 3 and support / sum(outcomes.values()) >= .7:
            reusable.append({"context": key, "prediction": prediction, "support": support,
                             "confidence": round(support / sum(outcomes.values()), 4)})
    counterexamples = [trial for trial in chosen_model["trials"] if not trial["correct"]]
    return {"version": 51,
            "benchmark": {"locked": True, "source_urls": benchmark_urls,
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
            "selection_status": "accepted_fixed_benchmark_gain" if selected else "no_model_beats_fixed_baseline",
            "best_rejected_candidate": None if selected else best_candidate,
            "selected_evaluation": selected or {"model_id": "frequency_baseline",
                "task": "none", "mode": "frequency_baseline",
                "correct": evaluations[0]["baseline_correct"] if evaluations else 0,
                "baseline_correct": evaluations[0]["baseline_correct"] if evaluations else 0,
                "total": len(final_test), "coverage": 1.0 if final_test else 0.0, "lift": 0},
            "evaluations": evaluations, "reusable_rules": reusable[:2000],
            "counterexamples": counterexamples[-1000:],
            "next_learning_target": ({"seed": " ".join(
                counterexamples[0].get("query_terms", [])) or "simple action story",
                "reason": "seek an independent boundary case for the best rejected world model"}
                if not selected and counterexamples else None),
            "revision_history": revisions[-200:],
            "invariants": ["benchmark_sources_are_locked", "benchmark_examples_are_frozen",
                           "benchmark_sources_never_train", "selection_and_final_are_disjoint",
                           "whole_source_split", "baseline_must_be_beaten_twice",
                           "paired_improvement_must_pass_sign_test_twice",
                           "original_sentence_retained"],
            "limitations": ["frames remain heuristic observations, not semantic truth",
                            "only explicit simple-clause experiences receive predictive credit"]}
