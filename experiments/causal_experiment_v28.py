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
from representation_learning_v31 import evaluate_representations, transform_transitions


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
        # Compare different actions in a shared observed object context. These are
        # counterfactual questions to seek evidence for, not synthetic answers.
        matched: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for prior, outcome, count in train:
            _, prior_action, prior_object = event_parts(prior)
            object_head = next((token for token in prior_object if token not in STOP_TOKENS), "none")
            matched[object_head][prior_action][normalized_action(outcome)] += count
        contrasts = []
        for object_head, actions in matched.items():
            supported = [(action, outcomes) for action, outcomes in actions.items()
                         if sum(outcomes.values()) >= 3]
            for index, (left_action, left_outcomes) in enumerate(supported):
                for right_action, right_outcomes in supported[index + 1:]:
                    outcomes = set(left_outcomes) | set(right_outcomes)
                    for outcome_action in outcomes:
                        left_rate = left_outcomes[outcome_action] / sum(left_outcomes.values())
                        right_rate = right_outcomes[outcome_action] / sum(right_outcomes.values())
                        difference = abs(left_rate - right_rate)
                        if difference >= 0.25:
                            contrasts.append({"shared_context": f"object={object_head}",
                                              "action_a": left_action, "action_b": right_action,
                                              "outcome": outcome_action,
                                              "rate_difference": round(difference, 4),
                                              "status": "observational_contrast"})
        contrasts.sort(key=lambda item: (-item["rate_difference"], item["shared_context"],
                                         item["action_a"], item["action_b"]))
        return {"method": "deterministic 80/20 holdout; predictions registered before comparison",
                "train_observations": sum(item[2] for item in train),
                "test_observations": total, "baseline_prediction": baseline_prediction,
                "hypotheses": hypotheses[:1000], "supported_hypotheses": len(accepted),
                "evaluation": {"accuracy": round(correct / total, 4) if total else 0.0,
                               "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
                               "correct": correct, "baseline_correct": baseline_correct, "total": total},
                "preregistered_predictions": preregistered,
                "matched_contrasts": contrasts[:100],
                "counterfactual_questions": [
                    {"question": (f"In {item['shared_context']}, would outcome={item['outcome']} "
                                  f"change if action={item['action_a']} were replaced by "
                                  f"action={item['action_b']}?"),
                     "status": "needs_comparative_evidence"}
                    for item in contrasts[:30]],
                "limitations": ["event parser is still shallow", "holdout evidence is observational",
                                "a positive lift is a falsifiable cause candidate, not a causal proof"]}


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


def evaluate_causal_views(transitions: dict[str, dict[str, int]], representation: dict,
                          previous: dict | None = None) -> dict:
    """Keep concrete evidence unless an abstract view materially improves unseen prediction."""
    previous = previous or {}
    concrete = CausalExperimentEngine(transitions).run()
    scheme = representation.get("selected_scheme", "surface")
    if scheme == "surface":
        concrete["selected_view"] = "concrete"
        concrete["view_evaluations"] = {"concrete": concrete["evaluation"]}
        selected = concrete
    else:
        abstract_transitions = transform_transitions(transitions, representation)
        abstract = CausalExperimentEngine(abstract_transitions).run()

        def improvement(report: dict) -> int:
            evaluation = report.get("evaluation", {})
            return evaluation.get("correct", 0) - evaluation.get("baseline_correct", 0)

        abstract_eval = abstract.get("evaluation", {})
        required = max(5, math.ceil(abstract_eval.get("total", 0) * 0.01))
        use_abstract = (improvement(abstract) >= required
                        and improvement(abstract) > improvement(concrete))
        selected = abstract if use_abstract else concrete
        selected["selected_view"] = "abstract" if use_abstract else "concrete"
        selected["view_evaluations"] = {
            "concrete": concrete.get("evaluation", {}),
            "abstract": abstract.get("evaluation", {}),
        }
        selected["view_hypotheses"] = {
            "concrete": concrete.get("supported_hypotheses", 0),
            "abstract": abstract.get("supported_hypotheses", 0),
        }
        selected["abstract_scheme"] = scheme

    # Track accuracy/lift/supported-hypothesis-count against training size: a
    # single "supported_hypotheses == 0" snapshot can't show whether more data is
    # approaching or receding from the independent_support significance bar.
    evaluation = selected.get("evaluation", {})
    training_size = selected.get("train_observations", 0)
    learning_curve = list(previous.get("learning_curve", []))
    curve_point = {"training_examples": training_size,
                   "accuracy": evaluation.get("accuracy", 0.0),
                   "baseline_accuracy": evaluation.get("baseline_accuracy", 0.0),
                   "lift": evaluation.get("correct", 0) - evaluation.get("baseline_correct", 0),
                   "supported_hypotheses": selected.get("supported_hypotheses", 0)}
    if not learning_curve or learning_curve[-1]["training_examples"] != training_size:
        learning_curve.append(curve_point)
    learning_curve = learning_curve[-200:]
    selected["learning_curve"] = learning_curve
    selected["learning_curve_trend"] = classify_trend(learning_curve, "lift", window=10, min_delta=1)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/global-language-memory.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/causal-memory.json")
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    # Only v29 quality-audited observations may be causal evidence. Older events are retained
    # for language history but cannot silently contaminate this evaluation.
    representation = evaluate_representations(memory.get("quality_event_transitions", {}))
    transitions = memory.get("quality_event_transitions", {})
    previous = (json.loads(args.output.read_text(encoding="utf-8"))
               if args.output.exists() else {})
    report = evaluate_causal_views(transitions, representation, previous)
    report["representation"] = {"selected_scheme": representation["selected_scheme"],
                                "selection_status": representation["selection_status"]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps({"supported_hypotheses": report["supported_hypotheses"],
                      **report["evaluation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
