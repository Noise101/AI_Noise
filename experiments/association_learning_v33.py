#!/usr/bin/env python3
"""Learn revisable associations from audited experience without causal credit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


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


def held_out(prior: str, outcome: str) -> bool:
    return hashlib.sha256(f"association:{prior}->{outcome}".encode()).digest()[0] % 5 == 0


class AssociationLearner:
    def __init__(self, transitions: dict[str, dict[str, int]],
                 event_counts: dict[str, int] | None = None):
        self.transitions = transitions
        self.event_counts = event_counts or {}

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
                (test if held_out(prior, outcome) else train).append((prior, outcome, count))
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
        return {"version": 33,
                "method": "associations learned on 80%; predictions corrected on unseen 20%",
                "structural_associations": self.structural_edges(),
                "predictive_associations": predictive[:5000],
                "evaluation": evaluation, "predictions": predictions[:1000],
                "reinforced": sum(item["status"] == "reinforced" for item in predictive),
                "weakened": sum(item["status"] == "weakened" for item in predictive),
                "warning": "association guides recall and prediction; it is not causal evidence"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/global-language-memory.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local/association-memory.json")
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    report = AssociationLearner(memory.get("quality_event_transitions", {}),
                                memory.get("quality_event_counts", {})).run()
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["evaluation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
