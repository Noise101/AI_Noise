#!/usr/bin/env python3
"""Turn audited event sequences into revisable structural predictions.

This mechanism does not supply meanings.  It proposes reusable rules from training
observations, tests them on deterministic holdout transitions, diagnoses failures,
and keeps failed rules as counterexamples rather than rewriting history.
"""

from __future__ import annotations

import hashlib
import math
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
    def __init__(self, transitions: dict[str, dict[str, int]],
                 contextual_transitions: dict[str, dict[str, int]] | None = None):
        self.transitions = transitions
        self.contextual_transitions = contextual_transitions or {}

    def contextual_evaluation(self) -> dict:
        """Test two-event context without replacing the simpler model unless it generalizes."""
        train, test = [], []
        for context, outcomes in self.contextual_transitions.items():
            for outcome, count in outcomes.items():
                key = hashlib.sha256(f"contextual:{context}->{outcome}".encode()).digest()[0]
                (test if key % 5 == 0 else train).append((context, outcome, count))
        rules: dict[str, Counter[str]] = defaultdict(Counter)
        fallback = Counter()
        for context, outcome, count in train:
            abstract_context = ">>".join(structural(event, False)
                                          for event in context.split(">>"))
            result = structural(outcome)
            rules[abstract_context][result] += count
            fallback[result] += count
        common = fallback.most_common(1)[0][0] if fallback else None
        correct = baseline = covered = total = 0
        feedback: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for context, outcome, count in test:
            abstract_context = ">>".join(structural(event, False)
                                          for event in context.split(">>"))
            observed = structural(outcome)
            predicted = rules[abstract_context].most_common(1)[0][0] if abstract_context in rules else common
            success = predicted == observed
            correct += count * success
            baseline += count * (common == observed)
            covered += count * (abstract_context in rules)
            total += count
            if predicted is not None and abstract_context in rules:
                feedback[(abstract_context, predicted)]["success" if success else "failure"] += count
        reusable = 0
        for context, outcomes in rules.items():
            predicted = outcomes.most_common(1)[0][0]
            result = feedback[(context, predicted)]
            tested = result["success"] + result["failure"]
            reusable += tested >= 2 and result["success"] / tested >= 0.6
        return {"correct": correct, "baseline_correct": baseline, "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0,
                "baseline_accuracy": round(baseline / total, 4) if total else 0.0,
                "coverage": round(covered / total, 4) if total else 0.0,
                "reusable_rules": reusable, "context_events": 2}

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
        failure_patterns = Counter()
        failure_examples: dict[str, dict] = {}
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
            if not success:
                _, prior_action, prior_object = parts(structural(prior))
                _, predicted_action, _ = parts(predicted or "||")
                _, observed_action, observed_object = parts(observed)
                pattern = (f"{kind}|after:{prior_action}|predicted:{predicted_action or 'none'}|"
                           f"observed:{observed_action}|object:{observed_object or prior_object}")
                failure_patterns[pattern] += count
                failure_examples.setdefault(pattern, {"prior": prior, "predicted": predicted,
                                                       "observed": observed})
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
        grouped_failures = [{"pattern": pattern, "count": count,
                             "example": failure_examples[pattern],
                             "query_terms": [term.split(":", 1)[-1] for term in pattern.split("|")[1:]
                                             if term.split(":", 1)[-1] not in {"none", "object"}]}
                            for pattern, count in failure_patterns.most_common(30)]
        summary = {"rules_formed": len(learned_rules),
                   "reusable_rules": sum(item["status"] == "reusable" for item in learned_rules),
                   "weakened_rules": sum(item["status"] == "weakened" for item in learned_rules),
                   "prediction_trials": len(trials), "prediction_errors": sum(failures.values()),
                   "failure_causes": dict(sorted(failures.items())),
                   "failure_patterns": grouped_failures, "evaluation": evaluation}
        contextual = self.contextual_evaluation()
        required = max(5, math.ceil(contextual.get("total", 0) * 0.01))
        if (contextual["correct"] - contextual["baseline_correct"] >= required
                and contextual["correct"] - contextual["baseline_correct"]
                > evaluation["correct"] - evaluation["baseline_correct"]):
            summary["evaluation"] = contextual
            summary["reusable_rules"] = contextual["reusable_rules"]
            summary["selected_context"] = "two_event"
        else:
            summary["selected_context"] = "one_event"
        summary["contextual_evaluation"] = contextual
        return {"version": 37,
                "cycle": ["structure_event", "form_rule", "predict_holdout",
                          "diagnose_failure", "revise_rule"],
                "rules": learned_rules[:5000], "trials": trials[:2000], "summary": summary,
                "causal_credit": False,
                "warning": "predictive structural rules are revisable; they are not causal proof"}
