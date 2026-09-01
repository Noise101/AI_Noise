#!/usr/bin/env python3
"""Stage-one active causal learning in a local controllable 2D micro-world."""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from collections import Counter


FEATURES = ("size", "roughness", "wall_contact", "color", "shape")


def candidate_hypotheses() -> list[dict]:
    """Bounded rule parts, not a supplied answer; unseen intervention chooses among them."""
    return [{"weights": dict(zip(FEATURES, weights)), "bias": bias}
            for weights in itertools.product(range(3), repeat=len(FEATURES))
            for bias in range(3)]


def predict(hypothesis: dict, observation: dict, force: int) -> bool:
    resistance = hypothesis["bias"] + sum(
        hypothesis["weights"][name] * observation[name] for name in FEATURES)
    return force > resistance


class PushWorld:
    """Environment owns this law. It is never included in the learner observation."""

    @staticmethod
    def observe(experiment: dict) -> dict:
        return {name: experiment[name] for name in FEATURES} | {
            "position": [0, 0], "visible_object_id": "A"}

    @staticmethod
    def act(experiment: dict) -> dict:
        hidden_resistance = (experiment["size"] + experiment["roughness"]
                             + 2 * experiment["wall_contact"])
        moved = experiment["force"] > hidden_resistance
        return {"action": "push", "target": "A", "force": experiment["force"],
                "position_before": [0, 0], "position_after": [1, 0] if moved else [0, 0],
                "moved": moved}


def experiment_id(experiment: dict) -> str:
    return ":".join(str(experiment[name]) for name in (*FEATURES, "force"))


def held_out(experiment: dict) -> bool:
    return hashlib.sha256(f"micro-holdout:{experiment_id(experiment)}".encode()).digest()[0] % 5 == 0


def all_experiments() -> list[dict]:
    return [dict(zip((*FEATURES, "force"), values)) for values in itertools.product(
        range(1, 4), range(3), range(2), range(2), range(2), range(1, 6))]


def empty_world_memory() -> dict:
    return {"version": 41, "stage": 1, "environment": "controllable 2D push world",
            "observations": [], "revision_history": [], "summary": {},
            "world_rule_visible_to_learner": False, "remote_llm_calls": 0}


def consistent_hypotheses(observations: list[dict]) -> list[dict]:
    candidates = candidate_hypotheses()
    exact = [hypothesis for hypothesis in candidates if all(
        predict(hypothesis, item["observation"], item["action"]["force"])
        == item["result"]["moved"] for item in observations)]
    if exact:
        return exact
    errors = [(sum(predict(hypothesis, item["observation"], item["action"]["force"])
                   != item["result"]["moved"] for item in observations), hypothesis)
              for hypothesis in candidates]
    minimum = min((item[0] for item in errors), default=0)
    return [hypothesis for error, hypothesis in errors if error == minimum]


def majority_prediction(hypotheses: list[dict], observation: dict, force: int) -> tuple[bool, float]:
    votes = Counter(predict(hypothesis, observation, force) for hypothesis in hypotheses)
    predicted, count = votes.most_common(1)[0]
    return predicted, count / max(1, len(hypotheses))


def choose_experiment(memory: dict, hypotheses: list[dict]) -> dict | None:
    tried = {item["experiment_id"] for item in memory.get("observations", [])}
    choices = []
    for experiment in all_experiments():
        if held_out(experiment) or experiment_id(experiment) in tried:
            continue
        observation = PushWorld.observe(experiment)
        moved_votes = sum(predict(item, observation, experiment["force"]) for item in hypotheses)
        probability = moved_votes / max(1, len(hypotheses))
        disagreement = 1.0 - abs(probability - 0.5) * 2
        novelty = sum(not any(old["observation"][name] == observation[name]
                              for old in memory.get("observations", [])) for name in FEATURES) / len(FEATURES)
        choices.append((disagreement + 0.1 * novelty,
                        hashlib.sha256(experiment_id(experiment).encode()).hexdigest(), experiment))
    return max(choices, key=lambda item: (item[0], item[1]))[2] if choices else None


def evaluate_holdout(hypotheses: list[dict]) -> dict:
    correct = total = confident = 0
    for experiment in all_experiments():
        if not held_out(experiment):
            continue
        observation = PushWorld.observe(experiment)
        predicted, confidence = majority_prediction(hypotheses, observation, experiment["force"])
        actual = PushWorld.act(experiment)["moved"]
        correct += predicted == actual
        confident += confidence >= 0.8
        total += 1
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "high_confidence_rate": round(confident / total, 4) if total else 0.0}


def learn_steps(memory: dict, steps: int = 3) -> dict:
    if memory.get("summary", {}).get("status") == "stage_1_mastered":
        return memory["summary"]
    for _ in range(max(0, steps)):
        before = consistent_hypotheses(memory.get("observations", []))
        experiment = choose_experiment(memory, before)
        if experiment is None:
            break
        observation = PushWorld.observe(experiment)
        predicted, confidence = majority_prediction(before, observation, experiment["force"])
        result = PushWorld.act(experiment)
        action = {"kind": "push", "target": "A", "force": experiment["force"]}
        record = {"experiment_id": experiment_id(experiment), "selected_by": "hypothesis_disagreement",
                  "observation": observation, "prediction": {"moved": predicted,
                  "confidence": round(confidence, 4)}, "action": action, "result": result,
                  "prediction_error": predicted != result["moved"],
                  "world_rule_disclosed": False,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        memory.setdefault("observations", []).append(record)
        after = consistent_hypotheses(memory["observations"])
        if len(after) < len(before) or record["prediction_error"]:
            memory.setdefault("revision_history", []).append({
                "experiment_id": record["experiment_id"], "hypotheses_before": len(before),
                "hypotheses_after": len(after), "prediction_error": record["prediction_error"],
                "change": "discard_rules_inconsistent_with_intervention"})
    hypotheses = consistent_hypotheses(memory.get("observations", []))
    holdout = evaluate_holdout(hypotheses)
    errors = sum(item["prediction_error"] for item in memory.get("observations", []))
    mastered = (len(memory.get("observations", [])) >= 20 and holdout["accuracy"] >= 0.95
                and len(hypotheses) <= 5 and errors >= 1)
    memory["summary"] = {"stage": 1, "status": "stage_1_mastered" if mastered else "learning",
                         "interventions": len(memory.get("observations", [])),
                         "prediction_errors": errors, "corrective_revisions": len(
                             memory.get("revision_history", [])),
                         "surviving_hypotheses": len(hypotheses), "holdout": holdout,
                         "next_action": ("retain mastery and await authorized stage 2" if mastered else
                                         "choose the experiment with greatest hypothesis disagreement"),
                         "world_rule_visible_to_learner": False,
                         "remote_llm_calls": 0}
    return memory["summary"]
