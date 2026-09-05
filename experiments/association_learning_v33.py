#!/usr/bin/env python3
"""Learn revisable associations from audited experience without causal credit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from representation_learning_v31 import learn_frequency_bands, learn_frequency_classes, parts


STOP = {"the", "a", "an", "to", "of", "and", "in", "on", "at", "for", "with",
        "he", "she", "it", "they", "his", "her", "their", "that", "this"}


def event_parts(event: str) -> tuple[str, str, list[str]]:
    values = event.split("|", 2)
    return (values[0] if values else "", values[1] if len(values) > 1 else "",
            values[2].split("_") if len(values) > 2 else [])


def features(event: str) -> list[str]:
    subject, action, objects = event_parts(event)
    result = []
    if subject:
        result.append(f"subject:{subject}")
    if action:
        result.append(f"action:{action}")
    result.extend(f"object:{item}" for item in objects if item not in STOP and len(item) >= 3)
    return result[:5]


def outcome_action(event: str) -> str:
    return event_parts(event)[1]


def source_held_out(source: str) -> bool:
    return hashlib.sha256(f"association-source:{source}".encode()).digest()[0] % 5 == 0


def held_out(prior: str, outcome: str, sources: list[str] | None = None) -> bool:
    """Hold out by whole source when source attribution is available (mirrors
    experience_rule_learning_v50's _source_holdout). Falls back to the legacy
    pair-hash split when it isn't -- e.g. the standalone CLI's
    global-language-memory input, which predates source attribution."""
    if sources:
        return source_held_out(min(sources))
    return hashlib.sha256(f"association:{prior}->{outcome}".encode()).digest()[0] % 5 == 0


# Hierarchical backoff, mirroring world_model_v51's: an ADDITIONAL candidate
# representation competing under the same selection logic as exact_action /
# learned_structural_class / learned_structural_bands below, not a change to
# how any of those three are themselves computed. This module votes across
# several cues per prediction rather than looking up one fixed context key, so
# "backing off" means dropping thin cues from the vote instead of walking a
# fixed mode chain: subject:/object: cues are drawn from an effectively open
# vocabulary and are exactly the ones found sparse in the audit that requested
# this (support=1 for 473/720 = 65.7% of predictive_associations); action:
# alone is the coarsest, densest cue (a small, bounded verb vocabulary) and is
# the natural floor to fall back on before the global majority.
MIN_CUE_SUPPORT = 3


def predict_with_backoff(prior: str, links: dict[str, Counter[str]],
                         fallback: str | None) -> tuple[str | None, str, list[str]]:
    """Vote across cues with enough of their own support to trust; if none
    clear MIN_CUE_SUPPORT, fall back to the action-only cue alone (even if
    thin); if that cue was never seen either, fall back to the global
    majority. Returns (prediction, resolved_level, cues_used) so callers can
    record which granularity actually produced the prediction."""
    cues = features(prior)
    scores = Counter()
    used = []
    for cue in cues:
        table = links.get(cue)
        if table and sum(table.values()) >= MIN_CUE_SUPPORT:
            scores.update(table)
            used.append(cue)
    if scores:
        return scores.most_common(1)[0][0], "cue_vote", used
    action_cue = next((cue for cue in cues if cue.startswith("action:")), None)
    if action_cue and links.get(action_cue):
        return links[action_cue].most_common(1)[0][0], "action_only_backoff", [action_cue]
    return fallback, "global_fallback", []


def classify_trend(points: list[dict], key: str, window: int = 10, min_delta: float = 0.0) -> str:
    """A lightweight improving/flat/declining read on the tail of a learning curve.

    Compares the mean of the older half of the window against the newer half, so
    a single noisy point can't flip the verdict. See world_model_v51.classify_trend
    for the same helper; kept as an independent copy here so this module stays
    runnable on its own.
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


class AssociationLearner:
    def __init__(self, transitions: dict[str, dict[str, int]],
                 event_counts: dict[str, int] | None = None,
                 previous: dict | None = None,
                 transition_sources: dict[str, dict[str, list[str]]] | None = None):
        self.transitions = transitions
        self.event_counts = event_counts or {}
        self.previous = previous or {}
        self.transition_sources = transition_sources or {}

    def structural_edges(self) -> list[dict]:
        edges: Counter[tuple[str, str, str]] = Counter()
        for event, count in self.event_counts.items():
            subject, action, objects = event_parts(event)
            if subject and action:
                edges[(f"subject:{subject}", f"action:{action}", "agent_action")] += count
            useful = [item for item in objects if item not in STOP and len(item) >= 3][:3]
            for item in useful:
                if action:
                    edges[(f"action:{action}", f"object:{item}", "action_object")] += count
                if subject:
                    left, right = sorted((f"subject:{subject}", f"object:{item}"))
                    edges[(left, right, "scene_cooccurrence")] += count
        for prior, outcomes in self.transitions.items():
            for outcome, count in outcomes.items():
                edges[(f"event:{prior}", f"event:{outcome}", "temporal_successor")] += count
        result = [{"source": left, "target": right, "kind": kind, "support": support,
                   "strength": round(1 - math.exp(-support / 3), 4)}
                  for (left, right, kind), support in edges.items()]
        return sorted(result, key=lambda item: (-item["strength"], -item["support"],
                                                item["kind"], item["source"], item["target"]))[:5000]

    def run(self) -> dict:
        train, test = [], []
        for prior, outcomes in self.transitions.items():
            for outcome, count in outcomes.items():
                sources = self.transition_sources.get(prior, {}).get(outcome, [])
                (test if held_out(prior, outcome, sources) else train).append((prior, outcome, count))
        baseline = Counter()
        links: dict[str, Counter[str]] = defaultdict(Counter)
        contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
        for prior, outcome, count in train:
            action = outcome_action(outcome)
            if not action:
                continue
            baseline[action] += count
            for cue in features(prior):
                links[cue][action] += count
                contexts[(cue, action)].add(prior)
        fallback = baseline.most_common(1)[0][0] if baseline else None
        feedback: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        correct = baseline_correct = total = covered = 0
        predictions = []
        for prior, outcome, count in test:
            observed = outcome_action(outcome)
            scores = Counter()
            for cue in features(prior):
                scores.update(links.get(cue, {}))
            predicted = scores.most_common(1)[0][0] if scores else fallback
            used = [cue for cue in features(prior) if links.get(cue, {}).get(predicted, 0)]
            success = predicted == observed
            for cue in used:
                feedback[(cue, predicted)]["success" if success else "failure"] += count
            correct += count * success
            baseline_correct += count * (fallback == observed)
            total += count
            covered += count * bool(used)
            predictions.append({"prior": prior, "prediction": predicted, "observed": observed,
                                "cues": used, "correct": success, "count": count})
        # Hierarchical backoff: an additional candidate representation, evaluated
        # on the exact same held-out pairs, competing for selection alongside
        # exact_action/learned_structural_class/learned_structural_bands below.
        backoff_correct = backoff_baseline_correct = backoff_total = backoff_covered = 0
        backoff_predictions = []
        for prior, outcome, count in test:
            observed = outcome_action(outcome)
            prediction, resolved_level, used_cues = predict_with_backoff(prior, links, fallback)
            success = prediction == observed
            backoff_correct += count * success
            backoff_baseline_correct += count * (fallback == observed)
            backoff_total += count
            backoff_covered += count * (resolved_level != "global_fallback")
            backoff_predictions.append({"prior": prior, "prediction": prediction,
                                        "observed": observed, "resolved_level": resolved_level,
                                        "cues": used_cues, "correct": success, "count": count})
        backoff_evaluation = {
            "accuracy": round(backoff_correct / backoff_total, 4) if backoff_total else 0.0,
            "baseline_accuracy": round(backoff_baseline_correct / backoff_total, 4)
                                if backoff_total else 0.0,
            "correct": backoff_correct, "baseline_correct": backoff_baseline_correct,
            "total": backoff_total,
            "coverage": round(backoff_covered / backoff_total, 4) if backoff_total else 0.0,
            "representation": "cue_hierarchy_backoff"}
        base_rate = baseline_correct / total if total else 0.0
        predictive = []
        for cue, outcomes in links.items():
            for outcome, support in outcomes.items():
                result = feedback[(cue, outcome)]
                successes, failures = result["success"], result["failure"]
                tested = successes + failures
                accuracy = successes / tested if tested else None
                status = ("reinforced" if tested >= 2 and accuracy > base_rate else
                          "weakened" if failures >= 2 and accuracy <= base_rate else "tentative")
                reliability = ((successes + 1) / (tested + 2)) if tested else 0.5
                strength = (1 - math.exp(-support / 3)) * reliability
                predictive.append({"cue": cue, "associated_outcome": outcome,
                                   "support": support,
                                   "independent_contexts": len(contexts[(cue, outcome)]),
                                   "prediction_successes": successes,
                                   "prediction_failures": failures,
                                   "tested_accuracy": None if accuracy is None else round(accuracy, 4),
                                   "strength": round(strength, 4), "status": status})
        predictive.sort(key=lambda item: (-item["strength"], -item["support"],
                                           item["cue"], item["associated_outcome"]))
        evaluation = {"accuracy": round(correct / total, 4) if total else 0.0,
                      "baseline_accuracy": round(base_rate, 4), "correct": correct,
                      "baseline_correct": baseline_correct, "total": total,
                      "coverage": round(covered / total, 4) if total else 0.0}
        action_classes, threshold = learn_frequency_classes(train)
        class_links: dict[str, Counter[str]] = defaultdict(Counter)
        class_baseline = Counter()
        for prior, outcome, count in train:
            prior_class = action_classes.get(parts(prior)[1], "rare")
            outcome_class = action_classes.get(parts(outcome)[1], "rare")
            class_links[prior_class][outcome_class] += count
            class_baseline[outcome_class] += count
        class_fallback = class_baseline.most_common(1)[0][0] if class_baseline else None
        class_correct = class_baseline_correct = class_total = 0
        class_predictions = []
        for prior, outcome, count in test:
            prior_class = action_classes.get(parts(prior)[1], "rare")
            observed_class = action_classes.get(parts(outcome)[1], "rare")
            prediction = (class_links[prior_class].most_common(1)[0][0]
                          if class_links.get(prior_class) else class_fallback)
            class_correct += count * (prediction == observed_class)
            class_baseline_correct += count * (class_fallback == observed_class)
            class_total += count
            class_predictions.append({"prior": prior, "prediction": prediction,
                                      "observed": observed_class, "baseline": class_fallback,
                                      "correct": prediction == observed_class,
                                      "baseline_correct": class_fallback == observed_class,
                                      "count": count})
        structural_evaluation = {
            "accuracy": round(class_correct / class_total, 4) if class_total else 0.0,
            "baseline_accuracy": round(class_baseline_correct / class_total, 4) if class_total else 0.0,
            "correct": class_correct, "baseline_correct": class_baseline_correct,
            "total": class_total, "coverage": 1.0 if class_total else 0.0,
            "representation": "learned_action_frequency_class",
            "learned_frequency_threshold": threshold}
        action_bands, band_thresholds = learn_frequency_bands(train)
        band_links: dict[str, Counter[str]] = defaultdict(Counter)
        band_baseline = Counter()
        for prior, outcome, count in train:
            prior_band = action_bands.get(parts(prior)[1], "rare")
            outcome_band = action_bands.get(parts(outcome)[1], "rare")
            band_links[prior_band][outcome_band] += count
            band_baseline[outcome_band] += count
        band_fallback = band_baseline.most_common(1)[0][0] if band_baseline else None
        band_correct = band_baseline_correct = band_total = 0
        band_predictions = []
        for prior, outcome, count in test:
            prior_band = action_bands.get(parts(prior)[1], "rare")
            observed_band = action_bands.get(parts(outcome)[1], "rare")
            prediction = (band_links[prior_band].most_common(1)[0][0]
                          if band_links.get(prior_band) else band_fallback)
            band_correct += count * (prediction == observed_band)
            band_baseline_correct += count * (band_fallback == observed_band)
            band_total += count
            band_predictions.append({"prior": prior, "prediction": prediction,
                                     "observed": observed_band, "baseline": band_fallback,
                                     "correct": prediction == observed_band,
                                     "baseline_correct": band_fallback == observed_band,
                                     "count": count})
        band_evaluation = {
            "accuracy": round(band_correct / band_total, 4) if band_total else 0.0,
            "baseline_accuracy": round(band_baseline_correct / band_total, 4) if band_total else 0.0,
            "correct": band_correct, "baseline_correct": band_baseline_correct,
            "total": band_total, "coverage": 1.0 if band_total else 0.0,
            "representation": "learned_action_frequency_bands",
            "learned_frequency_band_thresholds": list(band_thresholds)}
        candidates = [(correct - baseline_correct, "exact_action", evaluation),
                      (class_correct - class_baseline_correct, "learned_structural_class",
                       structural_evaluation),
                      (band_correct - band_baseline_correct, "learned_structural_bands",
                       band_evaluation),
                      (backoff_correct - backoff_baseline_correct, "cue_hierarchy_backoff",
                       backoff_evaluation)]
        _, selected_mode, selected_evaluation = max(candidates, key=lambda item: (item[0], item[1]))
        selected_predictions = ({"exact_action": predictions,
                                 "learned_structural_class": class_predictions,
                                 "learned_structural_bands": band_predictions,
                                 "cue_hierarchy_backoff": backoff_predictions}[selected_mode])
        # Track the selected representation's accuracy/lift/coverage against
        # training size: a single "correct == baseline_correct" snapshot can't
        # show whether more data is drifting toward or away from real signal.
        training_size = sum(count for _, _, count in train)
        learning_curve = list(self.previous.get("learning_curve", []))
        curve_point = {"training_examples": training_size, "selected_mode": selected_mode,
                       "accuracy": selected_evaluation.get("accuracy", 0.0),
                       "baseline_accuracy": selected_evaluation.get("baseline_accuracy", 0.0),
                       "lift": (selected_evaluation.get("correct", 0)
                                - selected_evaluation.get("baseline_correct", 0)),
                       "coverage": selected_evaluation.get("coverage", 0.0)}
        if not learning_curve or learning_curve[-1]["training_examples"] != training_size:
            learning_curve.append(curve_point)
        learning_curve = learning_curve[-200:]
        learning_curve_trend = classify_trend(learning_curve, "lift", window=10, min_delta=1)
        return {"version": 33,
                "method": "associations learned on 80%; predictions corrected on unseen 20%",
                "structural_associations": self.structural_edges(),
                "predictive_associations": predictive[:5000],
                "evaluation": evaluation, "predictions": predictions[:1000],
                "structural_evaluation": structural_evaluation,
                "band_evaluation": band_evaluation,
                "backoff_evaluation": backoff_evaluation,
                "selected_mode": selected_mode, "selected_evaluation": selected_evaluation,
                "selected_predictions": selected_predictions[:1000],
                "reinforced": sum(item["status"] == "reinforced" for item in predictive),
                "weakened": sum(item["status"] == "weakened" for item in predictive),
                "learning_curve": learning_curve, "learning_curve_trend": learning_curve_trend,
                "warning": "association guides recall and prediction; it is not causal evidence"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/global-language-memory.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/association-memory.json")
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    previous = (json.loads(args.output.read_text(encoding="utf-8"))
               if args.output.exists() else {})
    report = AssociationLearner(memory.get("quality_event_transitions", {}),
                                memory.get("quality_event_counts", {}), previous).run()
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["evaluation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
