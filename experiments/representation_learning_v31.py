#!/usr/bin/env python3
"""Select event abstractions by unseen prediction, not a supplied semantic dictionary."""

from __future__ import annotations

import hashlib
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parts(event: str) -> tuple[str, str, str]:
    values = (event.split("|", 2) + ["", ""])[:3]
    return values[0], values[1], values[2]


def held_out(prior: str, outcome: str) -> bool:
    return hashlib.sha256(f"{prior}->{outcome}".encode()).digest()[0] % 5 == 0


def common_prefix(left: str, right: str) -> str:
    result = []
    for a, b in zip(left, right):
        if a != b:
            break
        result.append(a)
    return "".join(result)


def learn_form_families(actions: set[str]) -> dict[str, str]:
    """Create candidates from observed spelling overlap; holdout decides whether they survive."""
    parent = {action: action for action in actions}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left in sorted(actions):
        for right in sorted(actions):
            if left >= right:
                continue
            prefix = common_prefix(left, right)
            if len(prefix) >= 4 and len(prefix) / max(len(left), len(right)) >= 0.55:
                a, b = find(left), find(right)
                parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for action in actions:
        groups[find(action)].append(action)
    mapping = {}
    for members in groups.values():
        label = common_prefix(min(members), max(members)).rstrip("aeiou") or min(members)
        for member in members:
            mapping[member] = label
    return mapping


def abstract_event(event: str, scheme: str, families: dict[str, str] | None = None,
                   action_classes: dict[str, str] | None = None) -> str:
    subject, action, obj = parts(event)
    object_head = obj.split("_")[-1] if obj else "none"
    if scheme == "surface":
        return event
    if scheme == "role_action":
        return f"agent|{action}|object"
    if scheme == "role_action_object":
        return f"agent|{action}|{object_head}"
    if scheme == "learned_form_family":
        return f"agent|{(families or {}).get(action, action)}|object"
    if scheme in {"learned_frequency_class", "learned_frequency_bands",
                  "learned_relational_class"}:
        return f"agent|{(action_classes or {}).get(action, 'rare')}|object"
    raise ValueError(f"unknown representation scheme: {scheme}")


def learn_frequency_classes(train: list[tuple[str, str, int]]) -> tuple[dict[str, str], int]:
    """Select a compression threshold on an inner holdout, never on the final test."""
    inner_train, inner_test = [], []
    for prior, outcome, count in train:
        key = hashlib.sha256(f"frequency-inner:{prior}->{outcome}".encode()).digest()[0]
        (inner_test if key % 5 == 0 else inner_train).append((prior, outcome, count))

    def action_counts(items):
        counts = Counter()
        for prior, outcome, count in items:
            counts[parts(prior)[1]] += count
            counts[parts(outcome)[1]] += count
        return counts

    def score(threshold: int) -> tuple[int, int]:
        counts = action_counts(inner_train)
        category = lambda event: "common" if counts[parts(event)[1]] >= threshold else "rare"
        choices, outcomes = defaultdict(Counter), Counter()
        for prior, outcome, count in inner_train:
            choices[category(prior)][category(outcome)] += count
            outcomes[category(outcome)] += count
        fallback = outcomes.most_common(1)[0][0] if outcomes else None
        correct = baseline = 0
        for prior, outcome, count in inner_test:
            observed = category(outcome)
            predicted = (choices[category(prior)].most_common(1)[0][0]
                         if choices.get(category(prior)) else fallback)
            correct += count * (predicted == observed)
            baseline += count * (fallback == observed)
        return correct - baseline, correct

    thresholds = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    threshold = max(thresholds, key=lambda item: (score(item), -item))
    counts = action_counts(train)
    return {action: ("common" if count >= threshold else "rare")
            for action, count in counts.items()}, threshold


def learn_frequency_bands(train: list[tuple[str, str, int]]) -> tuple[dict[str, str], tuple[int, int]]:
    """Learn rare/mid/high action bands on an inner holdout."""
    inner_train, inner_test = [], []
    for prior, outcome, count in train:
        key = hashlib.sha256(f"frequency-bands-inner:{prior}->{outcome}".encode()).digest()[0]
        (inner_test if key % 5 == 0 else inner_train).append((prior, outcome, count))

    def counts(items):
        result = Counter()
        for prior, outcome, count in items:
            result[parts(prior)[1]] += count
            result[parts(outcome)[1]] += count
        return result

    def score(low: int, high: int) -> tuple[int, int]:
        frequency = counts(inner_train)
        category = lambda event: ("high" if frequency[parts(event)[1]] >= high else
                                  ("mid" if frequency[parts(event)[1]] >= low else "rare"))
        choices, outcomes = defaultdict(Counter), Counter()
        for prior, outcome, count in inner_train:
            choices[category(prior)][category(outcome)] += count
            outcomes[category(outcome)] += count
        fallback = outcomes.most_common(1)[0][0] if outcomes else None
        correct = baseline = 0
        for prior, outcome, count in inner_test:
            observed = category(outcome)
            predicted = (choices[category(prior)].most_common(1)[0][0]
                         if choices.get(category(prior)) else fallback)
            correct += count * (predicted == observed)
            baseline += count * (fallback == observed)
        return correct - baseline, correct

    values = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
    low, high = max(((low, high) for low in values for high in values if low < high),
                    key=lambda pair: (score(*pair), -pair[0], -pair[1]))
    frequency = counts(train)
    mapping = {action: ("high" if count >= high else ("mid" if count >= low else "rare"))
               for action, count in frequency.items()}
    return mapping, (low, high)


def learn_relational_action_classes(train: list[tuple[str, str, int]]) -> dict[str, str]:
    """Induce functional action classes from what actions lead to, never from frequency alone."""
    effects: dict[str, Counter[str]] = defaultdict(Counter)
    for prior, outcome, count in train:
        prior_action = parts(prior)[1]
        outcome_action = parts(outcome)[1]
        if prior_action and outcome_action:
            effects[prior_action][outcome_action] += count
    result = {}
    for action, outcomes in effects.items():
        support = sum(outcomes.values())
        dominant, dominant_count = outcomes.most_common(1)[0]
        # Weak one-off relations stay distinct; repeated consequences may form a shared class.
        result[action] = (f"leads_to:{dominant}" if support >= 3
                          and dominant_count / support >= 0.5 else f"action:{action}")
    return result


def evaluate_representations(transitions: dict[str, dict[str, int]]) -> dict:
    train, test = [], []
    for prior, outcomes in transitions.items():
        for outcome, count in outcomes.items():
            (test if held_out(prior, outcome) else train).append((prior, outcome, count))
    train_actions = {parts(event)[1] for prior, outcome, _ in train for event in (prior, outcome)}
    families = learn_form_families(train_actions)
    action_classes, frequency_threshold = learn_frequency_classes(train)
    action_bands, frequency_band_thresholds = learn_frequency_bands(train)
    relational_classes = learn_relational_action_classes(train)
    schemes = ("surface", "role_action", "role_action_object", "learned_form_family",
               "learned_frequency_class", "learned_frequency_bands", "learned_relational_class")
    evaluations = []
    for scheme in schemes:
        scheme_classes = (action_bands if scheme == "learned_frequency_bands" else
                          relational_classes if scheme == "learned_relational_class" else
                          action_classes)
        choices: dict[str, Counter[str]] = defaultdict(Counter)
        global_outcomes = Counter()
        for prior, outcome, count in train:
            context = abstract_event(prior, scheme, families, scheme_classes)
            result = abstract_event(outcome, scheme, families, scheme_classes)
            choices[context][result] += count
            global_outcomes[result] += count
        fallback = global_outcomes.most_common(1)[0][0] if global_outcomes else None
        correct = baseline_correct = total = 0
        for prior, outcome, count in test:
            context = abstract_event(prior, scheme, families, scheme_classes)
            observed = abstract_event(outcome, scheme, families, scheme_classes)
            prediction = choices[context].most_common(1)[0][0] if choices.get(context) else fallback
            correct += count * (prediction == observed)
            baseline_correct += count * (fallback == observed)
            total += count
        evaluations.append({"scheme": scheme, "correct": correct,
                            "baseline_correct": baseline_correct, "total": total,
                            "accuracy": round(correct / total, 4) if total else 0.0,
                            "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
                            "coverage": round(sum(count for prior, _, count in test
                                if abstract_event(prior, scheme, families, scheme_classes) in choices) / total, 4)
                                if total else 0.0})
    surface = next(item for item in evaluations if item["scheme"] == "surface")
    eligible = [item for item in evaluations if item["total"] >= 20
                and item["accuracy"] >= item["baseline_accuracy"] + 0.02
                and item["accuracy"] >= surface["accuracy"] + 0.02]
    selected = max(eligible, key=lambda item: (item["accuracy"], item["coverage"],
                                                -schemes.index(item["scheme"])), default=surface)
    return {"method": "deterministic 80/20 holdout; abstractions learned on train only",
            "selected_scheme": selected["scheme"], "selection_status":
            "accepted_predictive_abstraction" if selected is not surface else "no_abstraction_beats_surface",
            "evaluations": evaluations, "learned_form_families": families,
            "learned_action_classes": action_classes,
            "learned_frequency_threshold": frequency_threshold,
            "learned_action_bands": action_bands,
            "learned_frequency_band_thresholds": list(frequency_band_thresholds),
            "learned_relational_action_classes": relational_classes,
            "warning": "a compact spelling family is retained only when unseen prediction improves"}


def transform_transitions(transitions: dict[str, dict[str, int]], report: dict) -> dict[str, dict[str, int]]:
    scheme = report.get("selected_scheme", "surface")
    families = report.get("learned_form_families", {})
    action_classes = (report.get("learned_action_bands", {}) if scheme == "learned_frequency_bands"
                      else report.get("learned_relational_action_classes", {})
                      if scheme == "learned_relational_class"
                      else report.get("learned_action_classes", {}))
    transformed: dict[str, dict[str, int]] = {}
    for prior, outcomes in transitions.items():
        context = abstract_event(prior, scheme, families, action_classes)
        bucket = transformed.setdefault(context, {})
        for outcome, count in outcomes.items():
            result = abstract_event(outcome, scheme, families, action_classes)
            bucket[result] = bucket.get(result, 0) + count
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/global-language-memory.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/representation-memory.json")
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    report = evaluate_representations(memory.get("quality_event_transitions", {}))
    report["selected_evaluation"] = next(item for item in report["evaluations"]
                                         if item["scheme"] == report["selected_scheme"])
    report["revisions"] = []
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps({"selected_scheme": report["selected_scheme"],
                      "selection_status": report["selection_status"],
                      **report["selected_evaluation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
