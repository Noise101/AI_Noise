#!/usr/bin/env python3
"""Turn audited event sequences into revisable structural predictions.

This mechanism does not supply meanings.  It proposes reusable rules from training
observations, tests them on deterministic holdout transitions, diagnoses failures,
and keeps failed rules as counterexamples rather than rewriting history.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict


def parts(event: str) -> tuple[str, str, str]:
    values = (event.split("|", 2) + ["", ""])[:3]
    return values[0], values[1], values[2]


def holdout(prior: str, outcome: str) -> bool:
    return hashlib.sha256(f"revision:{prior}->{outcome}".encode()).digest()[0] % 5 == 0


def structural(event: str, keep_object: bool = True) -> str:
    _, action, obj = parts(event)
    object_head = obj.split("_")[-1] if obj else "none"
    return f"agent|{action}|{object_head if keep_object else 'object'}"


def mismatch_kind(predicted: str | None, observed: str) -> str:
    if predicted is None:
        return "missing_rule"
    _, predicted_action, predicted_object = parts(predicted)
    _, observed_action, observed_object = parts(observed)
    action_wrong = predicted_action != observed_action
    object_wrong = predicted_object != observed_object
    if action_wrong and object_wrong:
        return "action_and_object_mismatch"
    if action_wrong:
        return "action_mismatch"
    if object_wrong:
        return "object_mismatch"
    return "matched"


class ExperienceRevisionEngine:
    def __init__(self, transitions: dict[str, dict[str, int]]):
        self.transitions = transitions

    def run(self) -> dict:
        train, test = [], []
        for prior, outcomes in self.transitions.items():
            for outcome, count in outcomes.items():
                (test if holdout(prior, outcome) else train).append((prior, outcome, count))

        rules: dict[str, Counter[str]] = defaultdict(Counter)
        subjects: dict[tuple[str, str], set[str]] = defaultdict(set)
        global_outcomes = Counter()
        for prior, outcome, count in train:
            context, result = structural(prior), structural(outcome)
            rules[context][result] += count
            subjects[(context, result)].add(parts(prior)[0])
            global_outcomes[result] += count
        fallback = global_outcomes.most_common(1)[0][0] if global_outcomes else None

        feedback: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        failures = Counter()
        trials = []
        correct = baseline_correct = covered = total = 0
        for prior, outcome, count in test:
            context, observed = structural(prior), structural(outcome)
            predicted = rules[context].most_common(1)[0][0] if context in rules else fallback
            kind = mismatch_kind(predicted, observed)
            success = kind == "matched"
            if predicted is not None:
                feedback[(context, predicted)]["success" if success else "failure"] += count
            failures[kind] += count
            correct += count * success
            baseline_correct += count * (fallback == observed)
            covered += count * (context in rules)
            total += count
            trials.append({"prior": prior, "abstract_context": context,
                           "predicted": predicted, "observed": observed,
                           "failure_cause": kind, "correct": success, "count": count})

        learned_rules = []
        for context, outcomes in rules.items():
            predicted, support = outcomes.most_common(1)[0]
            result = feedback[(context, predicted)]
            successes, failed = result["success"], result["failure"]
            tested = successes + failed
            reliability = (successes + 1) / (tested + 2)
            status = ("reusable" if tested >= 2 and reliability >= 0.6 else
                      "weakened" if failed >= 2 and reliability < 0.5 else "tentative")
            alternatives = [{"outcome": outcome, "support": count}
                            for outcome, count in outcomes.most_common(4) if outcome != predicted]
            learned_rules.append({"context": context, "prediction": predicted,
                                  "training_support": support,
                                  "independent_entities": len(subjects[(context, predicted)]),
                                  "holdout_successes": successes, "holdout_failures": failed,
                                  "reliability": round(reliability, 4), "status": status,
                                  "observed_alternatives": alternatives})
        learned_rules.sort(key=lambda item: (-item["independent_entities"],
                                              -item["training_support"], item["context"]))
        evaluation = {"correct": correct, "baseline_correct": baseline_correct, "total": total,
                      "accuracy": round(correct / total, 4) if total else 0.0,
                      "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
                      "coverage": round(covered / total, 4) if total else 0.0}
        summary = {"rules_formed": len(learned_rules),
                   "reusable_rules": sum(item["status"] == "reusable" for item in learned_rules),
                   "weakened_rules": sum(item["status"] == "weakened" for item in learned_rules),
                   "prediction_trials": len(trials), "prediction_errors": sum(failures.values()),
                   "failure_causes": dict(sorted(failures.items())), "evaluation": evaluation}
        return {"version": 37,
                "cycle": ["structure_event", "form_rule", "predict_holdout",
                          "diagnose_failure", "revise_rule"],
                "rules": learned_rules[:5000], "trials": trials[:2000], "summary": summary,
                "causal_credit": False,
                "warning": "predictive structural rules are revisable; they are not causal proof"}
