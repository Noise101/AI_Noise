#!/usr/bin/env python3
"""Discover which past state/action caused a delayed outcome, and at what lag."""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    sensors: tuple[int, ...]
    action: int


@dataclass(frozen=True)
class TemporalRule:
    lag: int
    sensor: int
    sensor_value: int
    action: int

    def matches(self, event: Event) -> bool:
        return event.action == self.action and event.sensors[self.sensor] == self.sensor_value

    def __str__(self) -> str:
        literal = ("" if self.sensor_value else "!") + f"s{self.sensor}"
        return f"{literal} & a{self.action} @-{self.lag}"


class DelayedWorld:
    def __init__(self, change_at: int = 80):
        self.change_at = change_at

    def true_rule(self, step: int) -> TemporalRule:
        return TemporalRule(3, 0, 1, 1) if step < self.change_at else TemporalRule(5, 2, 1, 0)

    def outcome(self, history: list[Event], step: int) -> int:
        rule = self.true_rule(step)
        return int(len(history) > rule.lag and rule.matches(history[-rule.lag - 1]))


class TemporalAgent:
    def __init__(self, sensor_count: int, max_lag: int, rng: random.Random, window: int = 48):
        self.sensor_count = sensor_count
        self.max_lag = max_lag
        self.rng = rng
        self.window = window
        self.events: list[Event] = []
        self.outcomes: list[int] = []
        self.best: TemporalRule | None = None
        self.confidence = 0.0

    def choose_action(self, sensors: tuple[int, ...], epsilon: float = 0.18) -> int:
        if self.best is None or self.rng.random() < epsilon:
            return self.rng.randrange(2)
        if sensors[self.best.sensor] == self.best.sensor_value:
            return self.best.action
        return 1 - self.best.action

    def observe(self, event: Event, outcome: int) -> None:
        self.events.append(event)
        self.outcomes.append(outcome)
        self._discover()

    @staticmethod
    def _log_likelihood(tp: int, fp: int, tn: int, fn: int) -> float:
        # Beta-smoothed likelihood rewards prediction and resists tiny-sample coincidences.
        p1 = (tp + 1) / (tp + fp + 2)
        p0 = (fn + 1) / (fn + tn + 2)
        return tp * math.log(p1) + fp * math.log(1 - p1) + fn * math.log(p0) + tn * math.log(1 - p0)

    def _score(self, rule: TemporalRule) -> float:
        tp = fp = tn = fn = 0
        start = max(rule.lag, len(self.outcomes) - self.window)
        for target_step in range(start, len(self.outcomes)):
            predicted = rule.matches(self.events[target_step - rule.lag])
            actual = bool(self.outcomes[target_step])
            if predicted and actual:
                tp += 1
            elif predicted:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1
        return self._log_likelihood(tp, fp, tn, fn)

    def _discover(self) -> None:
        if len(self.events) < self.max_lag + 8:
            return
        candidates = []
        for lag in range(1, self.max_lag + 1):
            for sensor in range(self.sensor_count):
                for sensor_value in (0, 1):
                    for action in (0, 1):
                        rule = TemporalRule(lag, sensor, sensor_value, action)
                        candidates.append((self._score(rule), rule))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, self.best = candidates[0]
        second_score = candidates[1][0]
        self.confidence = 1.0 - math.exp(-max(0.0, best_score - second_score))


def simulate(seed: int, steps: int = 170, change_at: int = 80, verbose: bool = False, max_lag: int = 7) -> dict[str, object]:
    rng = random.Random(seed)
    world = DelayedWorld(change_at)
    agent = TemporalAgent(sensor_count=4, max_lag=max_lag, rng=rng)
    history: list[Event] = []
    correct_before = correct_after = None

    for step in range(steps):
        sensors = tuple(rng.randrange(2) for _ in range(4))
        action = agent.choose_action(sensors)
        event = Event(sensors, action)
        history.append(event)
        outcome = world.outcome(history, step)
        agent.observe(event, outcome)
        if step == change_at - 1:
            correct_before = agent.best == world.true_rule(step)
        if verbose and (step % 10 == 0 or step in (change_at - 1, change_at, steps - 1)):
            print(f"{step:>3} outcome={outcome} learned={agent.best} conf={agent.confidence:.1%} true={world.true_rule(step)}")

    correct_after = agent.best == world.true_rule(steps - 1)
    return {"before": correct_before, "after": correct_after, "final": str(agent.best)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--steps", type=int, default=170)
    parser.add_argument("--change-at", type=int, default=80)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--max-lag", type=int, default=7)
    args = parser.parse_args()
    if args.trials:
        results = [simulate(seed, args.steps, args.change_at, max_lag=args.max_lag) for seed in range(args.trials)]
        print(f"before={sum(bool(r['before']) for r in results)/args.trials:.1%}")
        print(f"after={sum(bool(r['after']) for r in results)/args.trials:.1%}")
    else:
        print(simulate(args.seed, args.steps, args.change_at, verbose=True, max_lag=args.max_lag))
