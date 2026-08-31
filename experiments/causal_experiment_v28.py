#!/usr/bin/env python3
"""Pre-registered causal-candidate evaluation over cross-story event transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


AUXILIARIES = {"is", "was", "were", "are", "be", "been", "being", "has", "have", "had",
               "do", "did", "does", "will", "would", "could", "should"}
STOP_TOKENS = {"the", "a", "an", "to", "of", "and", "in", "on", "at", "for", "with",
               "he", "she", "it", "they", "his", "her", "their", "that", "this"}


def event_parts(key: str) -> tuple[str, str, list[str]]:
    parts = key.split("|", 2)
    subject = parts[0] if parts else ""
    action = parts[1] if len(parts) > 1 else ""
    obj = parts[2].split("_") if len(parts) > 2 else []
    return subject, action, obj


def normalized_action(key: str) -> str:
    _, action, obj = event_parts(key)
    if action not in AUXILIARIES:
        return action
    return next((token for token in obj if token not in STOP_TOKENS and len(token) > 2), action)


def observable_features(prior: str) -> set[str]:
    subject, _, obj = event_parts(prior)
    features = {f"prior_action={normalized_action(prior)}"}
    if subject:
        features.add(f"prior_subject={subject}")
    for token in obj:
        if token not in STOP_TOKENS and len(token) >= 4:
            features.add(f"prior_detail={token}")
            if sum(item.startswith("prior_detail=") for item in features) >= 3:
                break
    return features


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return (centre - spread) / denominator


@dataclass
class RevisableBelief:
    condition: str
    outcome: str
    support: int = 0
    counterexamples: int = 0
    status: str = "untested"
    revisions: list[dict] = field(default_factory=list)

    def update(self, outcome_observed: bool, amount: int = 1) -> None:
        before = self.status
        if outcome_observed:
            self.support += amount
        else:
            self.counterexamples += amount
        total = self.support + self.counterexamples
        precision = self.support / total
        self.status = ("supported" if self.support >= 3 and precision >= 0.65 else
                       "rejected" if self.counterexamples >= 3 and precision <= 0.35 else
                       "uncertain")
        if before not in {"untested", self.status}:
            self.revisions.append({"before": before, "after": self.status,
                                   "support": self.support, "counterexamples": self.counterexamples})


class CausalExperimentEngine:
    def __init__(self, transitions: dict[str, dict[str, int]]):
        self.transitions = transitions

    @staticmethod
    def _held_out(prior: str, outcome: str) -> bool:
        digest = hashlib.sha256(f"{prior}->{outcome}".encode()).digest()
        return digest[0] % 5 == 0

    def run(self) -> dict:
        train, test = [], []
        for prior, outcomes in self.transitions.items():
            for outcome, count in outcomes.items():
                item = (prior, outcome, count)
                (test if self._held_out(prior, outcome) else train).append(item)
        baseline = Counter()
        feature_totals = Counter()
        feature_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
        feature_contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
        feature_all_contexts: dict[str, set[str]] = defaultdict(set)
        for prior, outcome, count in train:
            outcome_action = normalized_action(outcome)
            baseline[outcome_action] += count
            for feature in observable_features(prior):
                feature_totals[feature] += count
                feature_outcomes[feature][outcome_action] += count
                feature_contexts[(feature, outcome_action)].add(prior)
                feature_all_contexts[feature].add(prior)
        train_total = sum(baseline.values())
        hypotheses = []
        for feature, outcomes in feature_outcomes.items():
            total = feature_totals[feature]
            for outcome, support in outcomes.items():
                if support < 3:
                    continue
                precision = support / total
                base_rate = baseline[outcome] / train_total if train_total else 0
                lift = precision - base_rate
                independent_support = len(feature_contexts[(feature, outcome)])
                independent_total = len(feature_all_contexts[feature])
                lower_bound = wilson_lower(independent_support, independent_total)
                predictive_feature = not feature.startswith("prior_subject=")
                score = (lower_bound - base_rate) * math.log2(1 + independent_support)
                status = ("supported_observational_candidate"
                          if predictive_feature and independent_support >= 5
                          and lower_bound >= base_rate + 0.05 else "weak")
                hypotheses.append({"condition": feature, "predicted_outcome": outcome,
                                   "support": support, "counterexamples": total - support,
                                   "independent_support": independent_support,
                                   "independent_contexts": independent_total,
                                   "precision": round(precision, 4), "baseline_rate": round(base_rate, 4),
                                   "lift": round(lift, 4), "lower_confidence_bound": round(lower_bound, 4),
                                   "score": round(score, 4), "status": status,
                                   "warning": "observational contrast, not proof of causation"})
        hypotheses.sort(key=lambda item: (-item["score"], -item["support"], item["condition"],
                                          item["predicted_outcome"]))
        accepted = [item for item in hypotheses if item["status"] == "supported_observational_candidate"]
        by_feature: dict[str, list[dict]] = defaultdict(list)
        for hypothesis in accepted:
            by_feature[hypothesis["condition"]].append(hypothesis)
        baseline_prediction = baseline.most_common(1)[0][0] if baseline else None
        preregistered, correct, baseline_correct, total = [], 0, 0, 0
        for prior, outcome, count in test:
            matches = [hypothesis for feature in observable_features(prior)
                       for hypothesis in by_feature.get(feature, [])]
            chosen = max(matches, key=lambda item: (item["score"], item["support"]), default=None)
            prediction = chosen["predicted_outcome"] if chosen else baseline_prediction
            observed = normalized_action(outcome)
            # This record is constructed before the observed value is compared below.
            record = {"prior": prior, "prediction": prediction,
                      "confidence_basis": None if not chosen else chosen["condition"],
                      "count": count, "observed_after_registration": observed}
            record["correct"] = prediction == observed
            preregistered.append(record)
            correct += count * (prediction == observed)
            baseline_correct += count * (baseline_prediction == observed)
            total += count
        return {"method": "deterministic 80/20 holdout; predictions registered before comparison",
                "train_observations": sum(item[2] for item in train),
                "test_observations": total, "baseline_prediction": baseline_prediction,
                "hypotheses": hypotheses[:1000], "supported_hypotheses": len(accepted),
                "evaluation": {"accuracy": round(correct / total, 4) if total else 0.0,
                               "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
                               "correct": correct, "baseline_correct": baseline_correct, "total": total},
                "preregistered_predictions": preregistered,
                "limitations": ["event parser is still shallow", "holdout evidence is observational",
                                "a positive lift is a falsifiable cause candidate, not a causal proof"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/global-language-memory.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/causal-memory.json")
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    report = CausalExperimentEngine(memory.get("event_transitions", {})).run()
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps({"supported_hypotheses": report["supported_hypotheses"],
                      **report["evaluation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
