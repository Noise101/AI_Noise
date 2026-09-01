#!/usr/bin/env python3
"""Complete stage-three curriculum: persistent, testable models of other agents."""

from __future__ import annotations

import hashlib
import itertools
import time

from social_world_v43 import empty_social_memory as empty_belief_memory
from social_world_v43 import learn_steps as learn_beliefs


AGENTS = ("ava", "ben")
NEEDS = ("hungry", "bored", "tired")
OBJECTS = ("apple", "book", "blanket")
PROFILE_MODELS = tuple(itertools.permutations(OBJECTS))
TRUE_PROFILES = {"ava": ("apple", "book", "blanket"),
                 "ben": ("blanket", "apple", "book")}

SOCIAL_MODELS = ("always", "if_able", "help_if_able", "never", "promise_only")
TRUE_SOCIAL = {"ava": "if_able", "ben": "help_if_able"}
REQUESTS = ("help", "share")

STYLES = ("concise", "detailed")
TRUE_STYLES = {"ava": "concise", "ben": "detailed"}


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _holdout(track: str, identifier: str) -> bool:
    return hashlib.sha256(f"stage3:{track}:{identifier}".encode()).digest()[0] % 5 == 0


def empty_stage_three_memory() -> dict:
    return {"version": 44, "stage": 3, "belief_memory": empty_belief_memory(),
            "profiles": {agent: {"candidates": [list(x) for x in PROFILE_MODELS],
                                  "observations": []} for agent in AGENTS},
            "social_contracts": {agent: {"candidates": list(SOCIAL_MODELS),
                                          "observations": []} for agent in AGENTS},
            "communication": {"reward_models": ["success_minus_messages", "success_only",
                                                   "messages_are_good", "always_zero"],
                              "policy_scores": {}, "observations": []},
            "explanation_styles": {agent: {"candidates": list(STYLES),
                                             "observations": []} for agent in AGENTS},
            "error_memory": [], "summary": {}, "remote_llm_calls": 0}


def migrate(memory: dict) -> dict:
    if memory.get("version") == 44:
        return memory
    old = dict(memory) if memory else empty_belief_memory()
    fresh = empty_stage_three_memory()
    fresh["belief_memory"] = old
    memory.clear()
    memory.update(fresh)
    return memory


def _record_error(memory: dict, track: str, prediction, actual, context: dict) -> None:
    if prediction != actual:
        memory["error_memory"].append({"track": track, "prediction": prediction,
                                       "actual": actual, "context": context, "at": _stamp()})


def _profile_choice(agent: str, need: str) -> str:
    return TRUE_PROFILES[agent][NEEDS.index(need)]


def _learn_profile(memory: dict) -> bool:
    for agent in AGENTS:
        state = memory["profiles"][agent]
        tried = {x["need"] for x in state["observations"]}
        need = next((n for n in NEEDS if n not in tried), None)
        if need is None:
            continue
        candidates = [tuple(x) for x in state["candidates"]]
        votes = [_profile_choice(agent, need) if not candidates else
                 candidate[NEEDS.index(need)] for candidate in candidates]
        prediction = max(set(votes), key=lambda x: (votes.count(x), x))
        actual = _profile_choice(agent, need)
        context = {"agent": agent, "need": need, "available_objects": list(reversed(OBJECTS))}
        _record_error(memory, "identity_profile", prediction, actual, context)
        state["observations"].append({**context, "prediction": prediction,
                                      "chosen_object": actual, "at": _stamp()})
        state["candidates"] = [list(candidate) for candidate in candidates
                               if candidate[NEEDS.index(need)] == actual]
        return True
    return False


def _social_action(model: str, request: str, able: bool, promised: bool) -> bool:
    if model == "always":
        return True
    if model == "if_able":
        return able
    if model == "help_if_able":
        return able and request == "help"
    if model == "promise_only":
        return able and promised
    return False


def _social_scenarios() -> list[tuple[str, bool, bool]]:
    return list(itertools.product(REQUESTS, (False, True), (False, True)))


def _learn_contract(memory: dict) -> bool:
    for agent in AGENTS:
        state = memory["social_contracts"][agent]
        tried = {(x["request"], x["able"], x["promised"]) for x in state["observations"]}
        candidates = state["candidates"]
        choices = []
        for scenario in _social_scenarios():
            sid = f"{agent}:{scenario}"
            if scenario in tried or _holdout("contract", sid):
                continue
            votes = {_social_action(model, *scenario) for model in candidates}
            choices.append((len(votes), sid, scenario))
        if not choices:
            continue
        request, able, promised = max(choices)[2]
        results = [_social_action(model, request, able, promised) for model in candidates]
        prediction = results.count(True) >= results.count(False)
        actual = _social_action(TRUE_SOCIAL[agent], request, able, promised)
        context = {"agent": agent, "request": request, "able": able, "promised": promised}
        _record_error(memory, "cooperation_and_promise", prediction, actual, context)
        state["observations"].append({**context, "prediction": prediction,
                                      "acted": actual,
                                      "speech_action_mismatch": promised and not actual,
                                      "at": _stamp()})
        state["candidates"] = [model for model in candidates
                               if _social_action(model, request, able, promised) == actual]
        return True
    return False


COMM_POLICIES = ("silent", "tell_first", "tell_uninformed", "tell_everyone")


def _communication_reward(policy: str, witnesses: tuple[bool, bool]) -> int:
    informed = list(witnesses)
    messages = 0
    if policy == "tell_first":
        informed[0] = True
        messages = 1
    elif policy == "tell_uninformed":
        messages = sum(not x for x in informed)
        informed = [True, True]
    elif policy == "tell_everyone":
        informed = [True, True]
        messages = 2
    success = all(informed)
    return (10 if success else 0) - messages


def _predicted_communication_reward(model: str, policy: str,
                                    witnesses: tuple[bool, bool]) -> int:
    true_reward = _communication_reward(policy, witnesses)
    messages = (0 if policy == "silent" else
                (sum(not x for x in witnesses) if policy == "tell_uninformed" else
                 (2 if policy == "tell_everyone" else 1)))
    success = true_reward > 0
    if model == "success_minus_messages":
        return true_reward
    if model == "success_only":
        return 10 if success else 0
    if model == "messages_are_good":
        return messages
    return 0


def _learn_communication(memory: dict) -> bool:
    state = memory["communication"]
    tried = {(tuple(x["witnesses"]), x["policy"]) for x in state["observations"]}
    for witnesses in itertools.product((False, True), repeat=2):
        for policy in COMM_POLICIES:
            sid = f"{witnesses}:{policy}"
            if (witnesses, policy) in tried or _holdout("communication", sid):
                continue
            reward = _communication_reward(policy, witnesses)
            state["observations"].append({"witnesses": list(witnesses), "policy": policy,
                                          "reward": reward, "both_succeeded": reward > 0,
                                          "at": _stamp()})
            state["policy_scores"].setdefault(str(witnesses), {})[policy] = reward
            state["reward_models"] = [model for model in state["reward_models"]
                                      if _predicted_communication_reward(
                                          model, policy, witnesses) == reward]
            return True
    return False


def _learn_style(memory: dict) -> bool:
    for agent in AGENTS:
        state = memory["explanation_styles"][agent]
        tried = {x["style"] for x in state["observations"]}
        style = next((x for x in STYLES if x not in tried), None)
        if style is None:
            continue
        candidates = state["candidates"]
        prediction = candidates[0] if len(candidates) == 1 else "concise"
        understood = style == TRUE_STYLES[agent]
        preferred = style if understood else next(x for x in STYLES if x != style)
        _record_error(memory, "adaptive_explanation", prediction, preferred,
                      {"agent": agent, "attempted_style": style})
        state["observations"].append({"agent": agent, "style": style,
                                      "understood": understood, "at": _stamp()})
        state["candidates"] = [x for x in candidates if x == preferred]
        return True
    return False


def _profile_eval(memory: dict) -> dict:
    correct = total = 0
    for agent in AGENTS:
        candidates = memory["profiles"][agent]["candidates"]
        if len(candidates) != 1:
            total += len(NEEDS)
            continue
        for need in NEEDS:
            correct += candidates[0][NEEDS.index(need)] == _profile_choice(agent, need)
            total += 1
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 4)}


def _contract_eval(memory: dict) -> dict:
    correct = total = mismatches = 0
    for agent in AGENTS:
        candidates = memory["social_contracts"][agent]["candidates"]
        for scenario in _social_scenarios():
            if not _holdout("contract", f"{agent}:{scenario}"):
                continue
            prediction = (_social_action(candidates[0], *scenario) if len(candidates) == 1 else None)
            actual = _social_action(TRUE_SOCIAL[agent], *scenario)
            correct += prediction == actual
            mismatches += scenario[2] and not actual and prediction == actual
            total += 1
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "speech_action_mismatches_correct": mismatches}


def _communication_eval(memory: dict) -> dict:
    correct = total = 0
    models = memory["communication"].get("reward_models", [])
    for witnesses in itertools.product((False, True), repeat=2):
        predicted = (max(COMM_POLICIES, key=lambda x: (
            _predicted_communication_reward(models[0], x, witnesses), x))
                     if len(models) == 1 else None)
        optimal = max(COMM_POLICIES, key=lambda x: (_communication_reward(x, witnesses), x))
        correct += predicted == optimal
        total += 1
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 4)}


def _style_eval(memory: dict) -> dict:
    correct = sum(memory["explanation_styles"][a]["candidates"] == [TRUE_STYLES[a]] for a in AGENTS)
    return {"correct": correct, "total": len(AGENTS),
            "accuracy": round(correct / len(AGENTS), 4)}


def learn_stage_three(memory: dict, steps: int = 5) -> dict:
    migrate(memory)
    for _ in range(max(0, steps)):
        belief = learn_beliefs(memory["belief_memory"], 1)
        if belief.get("status") not in {"stage_3_mastered", "stage_3_foundation_mastered"}:
            continue
        if _learn_profile(memory) or _learn_contract(memory) or _learn_communication(memory) or _learn_style(memory):
            continue
        break
    belief = memory["belief_memory"].get("summary", {})
    evaluations = {"belief": belief.get("unseen_social_tasks", {}),
                   "identity_profiles": _profile_eval(memory),
                   "cooperation_and_promises": _contract_eval(memory),
                   "multi_agent_communication": _communication_eval(memory),
                   "adaptive_explanation": _style_eval(memory)}
    passed = {name: result.get("accuracy") == 1.0 for name, result in evaluations.items()}
    foundation = belief.get("status") in {"stage_3_mastered", "stage_3_foundation_mastered"}
    mastered = foundation and all(passed.values()) and evaluations[
        "cooperation_and_promises"].get("speech_action_mismatches_correct", 0) > 0
    memory["summary"] = {"stage": 3,
                         "status": "stage_3_complete" if mastered else
                                   ("stage_3_foundation_mastered" if foundation else "learning"),
                         "competencies_passed": sum(passed.values()),
                         "competencies_total": len(passed), "competencies": evaluations,
                         "prediction_errors": len(memory["error_memory"]),
                         "persistent_agents": len(AGENTS),
                         "social_experiments": (len(memory["belief_memory"].get("observations", []))
                                                + sum(len(x["observations"]) for x in memory["profiles"].values())
                                                + sum(len(x["observations"]) for x in memory["social_contracts"].values())
                                                + len(memory["communication"]["observations"])
                                                + sum(len(x["observations"]) for x in memory["explanation_styles"].values())),
                         "private_state_directly_visible": False, "remote_llm_calls": 0,
                         "limitations": ["mastery is limited to the bounded social micro-world",
                                         "real human intentions remain uncertain and must be revised from evidence"]}
    return memory["summary"]
