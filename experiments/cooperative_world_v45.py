#!/usr/bin/env python3
"""Stage-four bounded world for learning sustained cooperation from consequences."""

from __future__ import annotations

import hashlib
import itertools
import time
from collections import Counter


COMPETENCIES = ("joint_goal", "role_assignment", "joint_planning", "negotiation",
                "trust", "failure_attribution", "group_information", "norm_discovery",
                "fairness", "verified_dialogue")
MIN_OBSERVATIONS = {name: 3 for name in COMPETENCIES}
MIN_OBSERVATIONS.update({"failure_attribution": 7, "trust": 5,
                         "group_information": 5, "norm_discovery": 5})


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sid(track: str, scenario: dict) -> str:
    body = ":".join(f"{key}={scenario[key]}" for key in sorted(scenario))
    return f"{track}:{body}"


def _heldout(track: str, scenario: dict) -> bool:
    """Stable split with enough unseen cases; split is never exposed to the learner."""
    return hashlib.sha256(("stage4-holdout:" + _sid(track, scenario)).encode()).digest()[0] % 4 == 0


def _joint_goal_scenarios() -> list[dict]:
    return [{"noise_goal": a, "partner_goal": b, "shared_dependency": shared}
            for a, b, shared in itertools.product(("food", "shelter", "knowledge"),
                                                   ("food", "shelter", "knowledge"),
                                                   (False, True))]


def _joint_goal(model: str, s: dict) -> str:
    if model == "mutual_dependency" and s["shared_dependency"]:
        return "shared_project"
    if model in {"mutual_dependency", "balanced"}:
        return min(s["noise_goal"], s["partner_goal"])
    if model == "self_first":
        return s["noise_goal"]
    if model == "partner_first":
        return s["partner_goal"]
    return "shared_project"


SKILLS = ("lift", "read", "navigate")
AGENTS = ("ava", "ben", "noise")
TRUE_SKILL_OWNER = dict(zip(SKILLS, ("ben", "noise", "ava")))


def _role_scenarios() -> list[dict]:
    return [{"task": task, "fatigued": tired} for task, tired in itertools.product(SKILLS, (None, *AGENTS))]


def _role(model: str, s: dict) -> str:
    base_maps = {
        "learned_specialists": TRUE_SKILL_OWNER,
        "noise_does_all": {x: "noise" for x in SKILLS},
        "ava_does_all": {x: "ava" for x in SKILLS},
        "ben_does_all": {x: "ben" for x in SKILLS},
        "rotating": dict(zip(SKILLS, AGENTS)),
    }
    owner = base_maps[model][s["task"]]
    if model == "learned_specialists" and owner == s["fatigued"]:
        return "ask_for_recovery"
    return owner


def _plan_scenarios() -> list[dict]:
    return [{"door": door, "box": box, "key": key}
            for door, box, key in itertools.product((False, True), repeat=3)]


def _plan(model: str, s: dict) -> str:
    needed = []
    if s["door"]:
        needed.append("hold_door")
    if s["box"]:
        needed.append("move_box")
    if s["key"]:
        needed.append("take_key")
    if model == "dependencies":
        return ">".join(needed + ["exit"])
    if model == "reverse":
        return ">".join(["exit"] + list(reversed(needed)))
    if model == "solo":
        return "noise_attempts_all"
    return "wait_forever"


def _negotiation_scenarios() -> list[dict]:
    return [{"noise_priority": a, "partner_priority": b, "divisible": d}
            for a, b, d in itertools.product((1, 2, 3), (1, 2, 3), (False, True))]


def _negotiation(model: str, s: dict) -> str:
    a, b = s["noise_priority"], s["partner_priority"]
    if model == "priority_then_split":
        if s["divisible"]:
            return "proportional_split"
        return "noise_yields" if b > a else ("partner_yields" if a > b else "alternate")
    if model == "equal_always":
        return "equal_split"
    if model == "self_wins":
        return "partner_yields"
    if model == "other_wins":
        return "noise_yields"
    return "alternate"


def _trust_scenarios() -> list[dict]:
    return [{"kept": kept, "accidental_failures": accidental,
             "intentional_failures": intentional, "repairs": repair}
            for kept, accidental, intentional, repair in itertools.product(range(3), range(2),
                                                                            range(2), range(2))]


def _trust(model: str, s: dict) -> str:
    if model == "evidence_and_repair":
        score = s["kept"] + s["repairs"] - s["accidental_failures"] - 2 * s["intentional_failures"]
        return "cooperate" if score >= 2 else ("verify" if score >= 0 else "avoid")
    if model == "forgive_everything":
        return "cooperate"
    if model == "one_failure_bans":
        return "avoid" if s["accidental_failures"] + s["intentional_failures"] else "cooperate"
    if model == "repair_erases_all":
        return "cooperate" if s["repairs"] else "avoid"
    return "verify"


FAILURE_SIGNALS = ("missing_message", "wrong_worker", "clock_expired",
                   "promise_broken", "noise_bad_prediction")
TRUE_CAUSES = dict(zip(FAILURE_SIGNALS, ("information_gap", "role_mismatch", "time_limit",
                                         "commitment_failure", "noise_model_error")))


def _failure_scenarios() -> list[dict]:
    return [{"signal": signal, "repeated": repeated}
            for signal, repeated in itertools.product(FAILURE_SIGNALS, (False, True))]


def _failure(model: str, s: dict) -> str:
    if model == "diagnostic":
        return TRUE_CAUSES[s["signal"]]
    if model == "blame_partner":
        return "commitment_failure"
    if model == "blame_self":
        return "noise_model_error"
    if model == "blame_time":
        return "time_limit"
    return "unknown"


def _information_scenarios() -> list[dict]:
    return [{"witness_mask": w, "need_mask": n}
            for w, n in itertools.product(range(8), range(1, 8))]


def _information(model: str, s: dict) -> str:
    if model == "needed_and_uninformed":
        recipients = s["need_mask"] & (~s["witness_mask"] & 7)
    elif model == "tell_all":
        recipients = 7
    elif model == "tell_needy":
        recipients = s["need_mask"]
    elif model == "tell_uninformed":
        recipients = ~s["witness_mask"] & 7
    else:
        recipients = 0
    return format(recipients, "03b")


def _norm_scenarios() -> list[dict]:
    return [{"act": act, "emergency": emergency, "permission": permission}
            for act, emergency, permission in itertools.product(
                ("skip_queue", "keep_borrowed", "decline_help"), (False, True), (False, True))]


def _norm(model: str, s: dict) -> str:
    if model == "contextual_norm":
        acceptable = s["permission"] or (s["emergency"] and s["act"] != "keep_borrowed")
        return "acceptable_exception" if acceptable else "violation"
    if model == "majority_is_right":
        return "acceptable_exception"
    if model == "no_exceptions":
        return "violation"
    if model == "permission_only":
        return "acceptable_exception" if s["permission"] else "violation"
    return "unknown"


def _fairness_scenarios() -> list[dict]:
    return [{"need_a": a, "need_b": b, "contribution_a": ca, "contribution_b": cb,
             "emergency": emergency}
            for a, b, ca, cb, emergency in itertools.product((1, 3), repeat=5)]


def _fairness(model: str, s: dict) -> str:
    if model == "context_sensitive":
        if s["emergency"] == 3 and s["need_a"] != s["need_b"]:
            return "favor_a" if s["need_a"] > s["need_b"] else "favor_b"
        if s["contribution_a"] != s["contribution_b"]:
            return "favor_a" if s["contribution_a"] > s["contribution_b"] else "favor_b"
        return "equal"
    if model == "equal_always":
        return "equal"
    if model == "need_only":
        return "favor_a" if s["need_a"] > s["need_b"] else ("favor_b" if s["need_b"] > s["need_a"] else "equal")
    if model == "contribution_only":
        return "favor_a" if s["contribution_a"] > s["contribution_b"] else ("favor_b" if s["contribution_b"] > s["contribution_a"] else "equal")
    return "favor_a"


def _dialogue_scenarios() -> list[dict]:
    return [{"evidence": evidence, "conflict": conflict, "testable": testable}
            for evidence, conflict, testable in itertools.product(("none", "weak", "strong"),
                                                                   (False, True), (False, True))]


def _dialogue(model: str, s: dict) -> str:
    if model == "verify_then_decide":
        if s["conflict"] and s["testable"]:
            return "ask_and_test"
        if s["evidence"] == "strong" and not s["conflict"]:
            return "accept_with_reason"
        return "hold_or_reject"
    if model == "copy_partner":
        return "accept_with_reason"
    if model == "reject_partner":
        return "hold_or_reject"
    if model == "always_question":
        return "ask_and_test"
    return "hold_or_reject"


SPECS = {
    "joint_goal": (_joint_goal_scenarios, _joint_goal,
                   ("mutual_dependency", "balanced", "self_first", "partner_first", "shared_always"),
                   "mutual_dependency"),
    "role_assignment": (_role_scenarios, _role,
                        ("learned_specialists", "noise_does_all", "ava_does_all", "ben_does_all", "rotating"),
                        "learned_specialists"),
    "joint_planning": (_plan_scenarios, _plan, ("dependencies", "reverse", "solo", "wait"), "dependencies"),
    "negotiation": (_negotiation_scenarios, _negotiation,
                    ("priority_then_split", "equal_always", "self_wins", "other_wins", "alternate"),
                    "priority_then_split"),
    "trust": (_trust_scenarios, _trust,
              ("evidence_and_repair", "forgive_everything", "one_failure_bans", "repair_erases_all", "always_verify"),
              "evidence_and_repair"),
    "failure_attribution": (_failure_scenarios, _failure,
                            ("diagnostic", "blame_partner", "blame_self", "blame_time", "unknown"), "diagnostic"),
    "group_information": (_information_scenarios, _information,
                          ("needed_and_uninformed", "tell_all", "tell_needy", "tell_uninformed", "silent"),
                          "needed_and_uninformed"),
    "norm_discovery": (_norm_scenarios, _norm,
                       ("contextual_norm", "majority_is_right", "no_exceptions", "permission_only", "unknown"),
                       "contextual_norm"),
    "fairness": (_fairness_scenarios, _fairness,
                 ("context_sensitive", "equal_always", "need_only", "contribution_only", "favor_noise"),
                 "context_sensitive"),
    "verified_dialogue": (_dialogue_scenarios, _dialogue,
                          ("verify_then_decide", "copy_partner", "reject_partner", "always_question", "passive"),
                          "verify_then_decide"),
}


def empty_cooperative_memory() -> dict:
    return {"version": 45, "stage": 4,
            "tracks": {name: {"candidates": list(SPECS[name][2]), "observations": [],
                              "revisions": []} for name in COMPETENCIES},
            "error_memory": [], "summary": {}, "remote_llm_calls": 0,
            "local_partner_mode": "deterministic_non_llm_training_partner"}


def _majority_prediction(candidates: list[str], rule, scenario: dict) -> str:
    votes = Counter(rule(candidate, scenario) for candidate in candidates)
    return sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _choose(track: str, state: dict) -> dict | None:
    generator, rule, _, _ = SPECS[track]
    tried = {item["scenario_id"] for item in state["observations"]}
    choices = []
    for scenario in generator():
        sid = _sid(track, scenario)
        if sid in tried or _heldout(track, scenario):
            continue
        outputs = {rule(model, scenario) for model in state["candidates"]}
        disagreement = len(outputs)
        choices.append((disagreement, hashlib.sha256(sid.encode()).hexdigest(), scenario))
    return max(choices)[2] if choices else None


def _learn_one(memory: dict, track: str) -> bool:
    state = memory["tracks"][track]
    if (len(state["candidates"]) == 1
            and len(state["observations"]) >= MIN_OBSERVATIONS[track]):
        return False
    _, rule, _, truth = SPECS[track]
    scenario = _choose(track, state)
    if scenario is None:
        return False
    before = list(state["candidates"])
    prediction = _majority_prediction(before, rule, scenario)
    actual = rule(truth, scenario)
    record = {"scenario_id": _sid(track, scenario), "context": scenario,
              "prediction": prediction, "observed_consequence": actual,
              "prediction_error": prediction != actual, "at": _stamp()}
    state["observations"].append(record)
    state["candidates"] = [model for model in before if rule(model, scenario) == actual]
    state["revisions"].append({"before": before, "after": state["candidates"],
                               "evidence": record["scenario_id"]})
    if record["prediction_error"]:
        memory["error_memory"].append({"competency": track, **record})
    return True


def _evaluate(track: str, state: dict) -> dict:
    generator, rule, _, truth = SPECS[track]
    correct = total = 0
    model = state["candidates"][0] if len(state["candidates"]) == 1 else None
    for scenario in generator():
        if not _heldout(track, scenario):
            continue
        prediction = rule(model, scenario) if model else None
        correct += prediction == rule(truth, scenario)
        total += 1
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "hypotheses_remaining": len(state["candidates"])}


def learn_cooperation(memory: dict, steps: int = 10) -> dict:
    if memory.get("version") != 45:
        memory.clear()
        memory.update(empty_cooperative_memory())
    for _ in range(max(0, steps)):
        progressed = False
        for track in COMPETENCIES:
            if _learn_one(memory, track):
                progressed = True
                break
        if not progressed:
            break
    evaluations = {name: _evaluate(name, memory["tracks"][name]) for name in COMPETENCIES}
    passed = {name: result["accuracy"] == 1.0 and result["hypotheses_remaining"] == 1
              for name, result in evaluations.items()}
    complete = all(passed.values())
    observations = sum(len(state["observations"]) for state in memory["tracks"].values())
    memory["summary"] = {
        "stage": 4,
        "status": "stage_4_complete" if complete else
                  ("stage_4_integration_testing" if sum(passed.values()) >= 7 else
                   ("stage_4_cooperation_learning" if sum(passed.values()) >= 3 else
                    "stage_4_foundation_learning")),
        "competencies_passed": sum(passed.values()), "competencies_total": len(COMPETENCIES),
        "competencies": evaluations, "cooperative_experiments": observations,
        "prediction_errors": len(memory["error_memory"]),
        "self_errors_remembered": sum(x["competency"] == "failure_attribution"
                                      for x in memory["error_memory"]),
        "remote_llm_calls": 0, "continues_autonomous_learning_after_completion": True,
        "limitations": ["completion is mastery of a bounded cooperative micro-world",
                        "real people and societies require uncertain, revisable models"]}
    return memory["summary"]
