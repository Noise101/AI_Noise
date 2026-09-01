#!/usr/bin/env python3
"""Stage-two tool use and multi-step learning without a supplied solution plan."""

from __future__ import annotations

import hashlib
import math
import random
import time


LOCATIONS = ("start", "tool0", "tool1", "target")
ACTIONS = tuple([f"move:{place}" for place in LOCATIONS]
                + ["pick:0", "pick:1", "place", "climb", "take"])


def task_specs() -> list[dict]:
    tasks = []
    for height in (2, 3, 4):
        for cap0 in (1, 2, 3):
            for cap1 in (1, 2, 3):
                if max(cap0, cap1) < height - 1:
                    continue
                for swapped in (False, True):
                    tasks.append({"goal_height": height, "tool_capacities": [cap0, cap1],
                                  "tool_locations": (["tool1", "tool0"] if swapped else
                                                     ["tool0", "tool1"])})
    return tasks


def task_id(task: dict) -> str:
    return f"{task['goal_height']}:{task['tool_capacities']}:{task['tool_locations']}"


def held_out(task: dict) -> bool:
    return hashlib.sha256(f"tool-holdout:{task_id(task)}".encode()).digest()[0] % 5 == 0


def initial_state(task: dict) -> dict:
    return {"agent_location": "start", "holding": None, "tool_locations": list(
                task["tool_locations"]), "tool_capacities": list(task["tool_capacities"]),
            "placed_capacity": 0, "on_platform": False, "goal_height": task["goal_height"],
            "goal_taken": False}


class ToolWorld:
    @staticmethod
    def observe(state: dict) -> dict:
        return {key: (list(value) if isinstance(value, list) else value)
                for key, value in state.items()}

    @staticmethod
    def act(state: dict, action: str) -> tuple[dict, bool, float]:
        after = ToolWorld.observe(state)
        success = False
        if action.startswith("move:"):
            after["agent_location"] = action.split(":", 1)[1]
            success = True
        elif action.startswith("pick:"):
            index = int(action[-1])
            if (after["holding"] is None
                    and after["agent_location"] == after["tool_locations"][index]):
                after["holding"] = index
                success = True
        elif action == "place":
            if after["holding"] is not None and after["agent_location"] == "target":
                index = after["holding"]
                after["placed_capacity"] = after["tool_capacities"][index]
                after["holding"] = None
                success = True
        elif action == "climb":
            if after["agent_location"] == "target" and after["placed_capacity"] > 0:
                after["on_platform"] = True
                success = True
        elif action == "take":
            if (after["on_platform"]
                    and 1 + after["placed_capacity"] >= after["goal_height"]):
                after["goal_taken"] = True
                success = True
        progress = (0.8 * (state["holding"] is None and after["holding"] is not None)
                    + 1.2 * (state["placed_capacity"] == 0 and after["placed_capacity"] > 0)
                    + 1.5 * (not state["on_platform"] and after["on_platform"]))
        reward = (10.0 if after["goal_taken"] and not state["goal_taken"] else
                  progress if progress else (0.1 if success else -0.2))
        reward -= 0.05
        return after, success, reward


def action_family(action: str) -> str:
    return action.split(":", 1)[0]


def features(state: dict, action: str) -> dict[str, float]:
    result = {"bias": 1.0, "goal_height": state["goal_height"] / 4,
              "placed_capacity": state["placed_capacity"] / 3,
              "holding_capacity": (0 if state["holding"] is None else
                                   state["tool_capacities"][state["holding"]] / 3),
              "on_platform": float(state["on_platform"]),
              "at_target": float(state["agent_location"] == "target"),
              "holding": float(state["holding"] is not None)}
    if action.startswith("pick:"):
        index = int(action[-1])
        result["selected_capacity"] = state["tool_capacities"][index] / 3
        result["at_selected_tool"] = float(
            state["agent_location"] == state["tool_locations"][index])
        result["selected_capacity_minus_height"] = (
            state["tool_capacities"][index] - state["goal_height"]) / 4
        result["reachable_unheld_tool"] = result["at_selected_tool"] * (1 - result["holding"])
    elif action.startswith("move:"):
        destination = action.split(":", 1)[1]
        result["destination_target"] = float(destination == "target")
        result["destination_same"] = float(destination == state["agent_location"])
        result["holding_to_target"] = result["holding"] * result["destination_target"]
        for index in range(2):
            if destination == state["tool_locations"][index]:
                result["destination_tool_capacity"] = state["tool_capacities"][index] / 3
                result["empty_to_tool"] = 1 - result["holding"]
                result["destination_capacity_minus_height"] = (
                    state["tool_capacities"][index] - state["goal_height"]) / 4
    elif action == "place":
        result["ready_to_place"] = result["at_target"] * result["holding"]
    elif action == "climb":
        result["ready_to_climb"] = result["at_target"] * float(state["placed_capacity"] > 0)
    elif action == "take":
        result["placed_capacity_minus_height"] = (
            1 + state["placed_capacity"] - state["goal_height"]) / 4
    return result


def q_value(weights: dict, state: dict, action: str) -> float:
    family_weights = weights.get(action_family(action), {})
    return sum(family_weights.get(name, 0.0) * value for name, value in features(state, action).items())


def choose_action(weights: dict, state: dict, rng: random.Random, epsilon: float) -> str:
    if rng.random() < epsilon:
        return ACTIONS[rng.randrange(len(ACTIONS))]
    scored = [(q_value(weights, state, action),
               hashlib.sha256(f"{action}:{state}".encode()).hexdigest(), action) for action in ACTIONS]
    return max(scored)[2]


def empty_tool_memory() -> dict:
    return {"version": 42, "stage": 2, "environment": "tool transport and elevated target",
            "episodes": 0, "weights": {}, "successful_plans": [], "failure_memory": {},
            "summary": {}, "solution_plan_supplied": False, "remote_llm_calls": 0}


def run_episode(memory: dict, task: dict, training: bool = True, max_steps: int = 14) -> dict:
    episode = memory.get("episodes", 0)
    rng = random.Random(int(hashlib.sha256(f"tool-episode:{episode}:{task_id(task)}".encode()
                                           ).hexdigest()[:16], 16))
    epsilon = max(0.05, 0.55 * math.exp(-episode / 500)) if training else 0.0
    state = initial_state(task)
    trace = []
    weights = memory.setdefault("weights", {})
    for _ in range(max_steps):
        action = choose_action(weights, state, rng, epsilon)
        before = ToolWorld.observe(state)
        after, success, reward = ToolWorld.act(state, action)
        trace.append({"observation": before, "action": action, "action_succeeded": success,
                      "reward": round(reward, 3), "result": ToolWorld.observe(after)})
        if training:
            family = action_family(action)
            family_weights = weights.setdefault(family, {})
            current = q_value(weights, state, action)
            future = 0.0 if after["goal_taken"] else max(q_value(weights, after, item)
                                                         for item in ACTIONS)
            error = reward + 0.92 * future - current
            for name, value in features(state, action).items():
                family_weights[name] = family_weights.get(name, 0.0) + 0.08 * error * value
            if not success:
                key = f"{family}|at:{state['agent_location']}|holding:{state['holding'] is not None}"
                memory.setdefault("failure_memory", {})[key] = (
                    memory.setdefault("failure_memory", {}).get(key, 0) + 1)
        state = after
        if state["goal_taken"]:
            break
    return {"task_id": task_id(task), "success": state["goal_taken"],
            "steps": len(trace), "trace": trace,
            "plan": [item["action"] for item in trace], "solution_plan_supplied": False}


def evaluate_unseen(memory: dict) -> dict:
    results = [run_episode(memory, task, training=False) for task in task_specs() if held_out(task)]
    successes = sum(item["success"] for item in results)
    return {"successes": successes, "total": len(results),
            "success_rate": round(successes / len(results), 4) if results else 0.0,
            "mean_steps_on_success": round(sum(item["steps"] for item in results if item["success"])
                                           / max(1, successes), 2)}


def learn_episodes(memory: dict, episodes: int = 25) -> dict:
    if memory.get("summary", {}).get("status") == "stage_2_mastered":
        return memory["summary"]
    training_tasks = [task for task in task_specs() if not held_out(task)]
    for _ in range(max(0, episodes)):
        episode = memory.get("episodes", 0)
        task = training_tasks[episode % len(training_tasks)]
        result = run_episode(memory, task, training=True)
        memory["episodes"] = episode + 1
        if result["success"]:
            memory.setdefault("successful_plans", []).append(
                {"episode": episode, "task_id": result["task_id"], "steps": result["steps"],
                 "plan": result["plan"], "learned_from_actions": True})
            memory["successful_plans"] = memory["successful_plans"][-200:]
    evaluation = evaluate_unseen(memory)
    mastered = (memory["episodes"] >= 600 and evaluation["success_rate"] >= 0.9
                and len(memory.get("successful_plans", [])) >= 30
                and bool(memory.get("failure_memory")))
    memory["summary"] = {"stage": 2, "status": "stage_2_mastered" if mastered else "learning",
                         "episodes": memory["episodes"],
                         "successful_training_plans": len(memory.get("successful_plans", [])),
                         "remembered_action_failures": sum(memory.get("failure_memory", {}).values()),
                         "learned_action_families": len(memory.get("weights", {})),
                         "unseen_tasks": evaluation,
                         "solution_plan_supplied": False, "remote_llm_calls": 0,
                         "next_action": ("retain mastery and await authorized stage 3" if mastered else
                                         "continue acting, updating values, and retrying plans")}
    return memory["summary"]
