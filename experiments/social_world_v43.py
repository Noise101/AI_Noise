#!/usr/bin/env python3
"""Stage-three learning about another agent's partial observations and beliefs."""

from __future__ import annotations

import hashlib
import itertools
import time
from collections import Counter


LOCATIONS = ("left", "right")
MODELS = ("reality", "initial", "witness_else_initial", "message_else_initial",
          "message_then_witness_else_initial", "witness_then_message_else_initial",
          "fixed_left", "fixed_right")


def scenarios() -> list[dict]:
    return [{"initial_location": initial, "current_location": current,
             "partner_witnessed_transfer": witnessed, "message": message}
            for initial, current, witnessed, message in itertools.product(
                LOCATIONS, LOCATIONS, (False, True), (None, *LOCATIONS))]


def scenario_id(scenario: dict) -> str:
    return (f"{scenario['initial_location']}:{scenario['current_location']}:"
            f"{int(scenario['partner_witnessed_transfer'])}:{scenario['message']}")


def held_out(scenario: dict) -> bool:
    return hashlib.sha256(f"social-holdout:{scenario_id(scenario)}".encode()).digest()[0] % 5 == 0


class SocialWorld:
    """The partner acts from private belief; that belief is never returned to Noise."""

    @staticmethod
    def observe(scenario: dict) -> dict:
        return {"object_initial_location": scenario["initial_location"],
                "object_current_location_seen_by_noise": scenario["current_location"],
                "partner_present_during_transfer": scenario["partner_witnessed_transfer"],
                "message_sent_by_noise": scenario["message"]}

    @staticmethod
    def partner_action(scenario: dict) -> dict:
        private_belief = scenario["initial_location"]
        if scenario["partner_witnessed_transfer"]:
            private_belief = scenario["current_location"]
        if scenario["message"] is not None:
            private_belief = scenario["message"]
        return {"partner_action": "search", "searched_location": private_belief,
                "private_belief_disclosed": False}


def model_prediction(model: str, observation: dict) -> str:
    initial = observation["object_initial_location"]
    current = observation["object_current_location_seen_by_noise"]
    witnessed = observation["partner_present_during_transfer"]
    message = observation["message_sent_by_noise"]
    if model == "reality":
        return current
    if model == "initial":
        return initial
    if model == "witness_else_initial":
        return current if witnessed else initial
    if model == "message_else_initial":
        return message if message is not None else initial
    if model == "message_then_witness_else_initial":
        return message if message is not None else (current if witnessed else initial)
    if model == "witness_then_message_else_initial":
        return current if witnessed else (message if message is not None else initial)
    return "left" if model == "fixed_left" else "right"


def empty_social_memory() -> dict:
    return {"version": 43, "stage": 3, "environment": "partial-observation partner world",
            "observations": [], "revision_history": [], "summary": {},
            "partner_private_belief_visible": False, "remote_llm_calls": 0}


def surviving_models(memory: dict) -> list[str]:
    exact = [model for model in MODELS if all(
        model_prediction(model, item["observation"]) == item["result"]["searched_location"]
        for item in memory.get("observations", []))]
    if exact:
        return exact
    errors = {model: sum(model_prediction(model, item["observation"])
                         != item["result"]["searched_location"]
                         for item in memory.get("observations", [])) for model in MODELS}
    best = min(errors.values(), default=0)
    return [model for model, error in errors.items() if error == best]


def predict_majority(models: list[str], observation: dict) -> tuple[str, float]:
    votes = Counter(model_prediction(model, observation) for model in models)
    prediction, count = votes.most_common(1)[0]
    return prediction, count / max(1, len(models))


def choose_scenario(memory: dict, models: list[str]) -> dict | None:
    tried = {item["scenario_id"] for item in memory.get("observations", [])}
    choices = []
    for scenario in scenarios():
        if held_out(scenario) or scenario_id(scenario) in tried:
            continue
        observation = SocialWorld.observe(scenario)
        votes = Counter(model_prediction(model, observation) for model in models)
        disagreement = 1 - max(votes.values()) / max(1, len(models))
        communication_novelty = 0.05 * (scenario["message"] is not None)
        choices.append((disagreement + communication_novelty,
                        hashlib.sha256(scenario_id(scenario).encode()).hexdigest(), scenario))
    return max(choices)[2] if choices else None


def evaluate_unseen(models: list[str]) -> dict:
    correct = false_belief_correct = false_belief_total = total = 0
    for scenario in scenarios():
        if not held_out(scenario):
            continue
        observation = SocialWorld.observe(scenario)
        prediction, _ = predict_majority(models, observation)
        actual = SocialWorld.partner_action(scenario)["searched_location"]
        correct += prediction == actual
        false_belief = (not scenario["partner_witnessed_transfer"]
                        and scenario["message"] is None
                        and scenario["initial_location"] != scenario["current_location"])
        false_belief_total += false_belief
        false_belief_correct += false_belief and prediction == actual
        total += 1
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "false_belief_correct": false_belief_correct,
            "false_belief_total": false_belief_total}


def learn_steps(memory: dict, steps: int = 3) -> dict:
    if memory.get("summary", {}).get("status") in {
            "stage_3_mastered", "stage_3_foundation_mastered"}:
        return memory["summary"]
    for _ in range(max(0, steps)):
        before = surviving_models(memory)
        scenario = choose_scenario(memory, before)
        if scenario is None:
            break
        observation = SocialWorld.observe(scenario)
        prediction, confidence = predict_majority(before, observation)
        result = SocialWorld.partner_action(scenario)
        record = {"scenario_id": scenario_id(scenario), "selected_by": "model_disagreement",
                  "observation": observation,
                  "noise_action": ({"kind": "tell", "location": scenario["message"]}
                                   if scenario["message"] else {"kind": "remain_silent"}),
                  "prediction": {"partner_search_location": prediction,
                                 "confidence": round(confidence, 4)},
                  "result": result, "prediction_error": prediction != result["searched_location"],
                  "partner_private_belief_disclosed": False,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        memory.setdefault("observations", []).append(record)
        after = surviving_models(memory)
        if len(after) < len(before) or record["prediction_error"]:
            memory.setdefault("revision_history", []).append({
                "scenario_id": record["scenario_id"], "models_before": before,
                "models_after": after, "prediction_error": record["prediction_error"],
                "change": "discard_other-mind_models_inconsistent_with_observed_action"})
    models = surviving_models(memory)
    evaluation = evaluate_unseen(models)
    errors = sum(item["prediction_error"] for item in memory.get("observations", []))
    mastered = (len(memory.get("observations", [])) >= 12 and len(models) == 1
                and evaluation["accuracy"] == 1.0 and errors >= 1
                and (evaluation["false_belief_total"] == 0
                     or evaluation["false_belief_correct"] == evaluation["false_belief_total"]))
    memory["summary"] = {"stage": 3,
                         "status": "stage_3_foundation_mastered" if mastered else "learning",
                         "social_experiments": len(memory.get("observations", [])),
                         "prediction_errors": errors,
                         "model_revisions": len(memory.get("revision_history", [])),
                         "surviving_other_models": len(models), "unseen_social_tasks": evaluation,
                         "partner_private_belief_visible": False, "remote_llm_calls": 0,
                         "next_action": ("continue with the rest of stage 3" if mastered else
                                         "test where candidate models predict different partner actions")}
    return memory["summary"]
