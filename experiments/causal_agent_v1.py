#!/usr/bin/env python3
"""Online causal rule induction: hypotheses are synthesized from experience."""

from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    sensors: tuple[int, ...]

    def __str__(self) -> str:
        return "".join(map(str, self.sensors))


class ChangingWorld:
    """Reward rules are hidden Boolean formulas over unnamed sensor channels."""

    def __init__(self, sensor_count: int = 6, change_at: int = 35):
        self.sensor_count = sensor_count
        self.change_at = change_at

    def candidates(self) -> tuple[Observation, ...]:
        return tuple(Observation(bits) for bits in itertools.product((0, 1), repeat=self.sensor_count))

    def reward(self, observation: Observation, step: int) -> int:
        x = observation.sensors
        if step < self.change_at:
            return int(x[0] == 1 and x[3] == 0)
        return int(x[2] == 1 or x[5] == 1)

    def rule_name(self, step: int) -> str:
        return "s0 & !s3" if step < self.change_at else "s2 | s5"


@dataclass(frozen=True)
class Literal:
    sensor: int
    value: int

    def matches(self, observation: Observation) -> bool:
        return observation.sensors[self.sensor] == self.value

    def __str__(self) -> str:
        return ("" if self.value else "!") + f"s{self.sensor}"


@dataclass(frozen=True)
class Rule:
    """A conjunction or disjunction assembled from sensor literals."""

    operation: str
    literals: tuple[Literal, ...]

    def predicts(self, observation: Observation) -> int:
        values = [literal.matches(observation) for literal in self.literals]
        return int(all(values) if self.operation == "and" else any(values))

    @property
    def complexity(self) -> int:
        return len(self.literals)

    def __str__(self) -> str:
        joiner = " & " if self.operation == "and" else " | "
        return joiner.join(map(str, self.literals))


class RuleInducer:
    def __init__(self, sensor_count: int, rng: random.Random, memory_size: int = 18):
        self.sensor_count = sensor_count
        self.rng = rng
        self.memory: deque[tuple[Observation, int]] = deque(maxlen=memory_size)
        self.rules: list[Rule] = []
        self.weights: list[float] = []
        self.last_surprise = 0.0
        self._synthesize_rules()

    def _synthesize_rules(self) -> None:
        literals = [Literal(sensor, value) for sensor in range(self.sensor_count) for value in (0, 1)]
        rules: list[Rule] = []
        # The grammar is known; concrete rules are generated, scored and discarded online.
        for width in (1, 2):
            for combo in itertools.combinations(literals, width):
                if len({literal.sensor for literal in combo}) != width:
                    continue
                rules.append(Rule("and", combo))
                if width > 1:
                    rules.append(Rule("or", combo))
        self.rules = rules
        self._rescore()

    def _rescore(self) -> None:
        raw = []
        for rule in self.rules:
            mistakes = sum(rule.predicts(obs) != outcome for obs, outcome in self.memory)
            # Prefer simpler explanations, while allowing contradictions after a change.
            raw.append(math.exp(-2.2 * mistakes - 0.18 * rule.complexity))
        total = sum(raw)
        self.weights = [value / total for value in raw]

    def predict_probability(self, observation: Observation) -> float:
        return sum(weight * rule.predicts(observation) for rule, weight in zip(self.rules, self.weights))

    @staticmethod
    def binary_entropy(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def choose(self, observations: tuple[Observation, ...]) -> tuple[Observation, str]:
        scored = []
        for observation in observations:
            p = self.predict_probability(observation)
            uncertainty = self.binary_entropy(p)
            score = p + 0.75 * uncertainty
            scored.append((score, self.rng.random(), uncertainty, observation))
        _, _, uncertainty, selected = max(scored)
        return selected, "experiment" if uncertainty > 0.35 else "exploit"

    def learn(self, observation: Observation, outcome: int) -> float:
        p = self.predict_probability(observation)
        self.last_surprise = -math.log2(max(1e-9, p if outcome else 1 - p))
        # Large contradictions indicate that stale episodes should be forgotten faster.
        if self.last_surprise > 3.0:
            while len(self.memory) > 5:
                self.memory.popleft()
        self.memory.append((observation, outcome))
        self._rescore()
        return self.last_surprise

    def best_rule(self) -> tuple[Rule, float]:
        index = max(range(len(self.rules)), key=self.weights.__getitem__)
        return self.rules[index], self.weights[index]


def run(steps: int, change_at: int, seed: int) -> None:
    rng = random.Random(seed)
    world = ChangingWorld(change_at=change_at)
    agent = RuleInducer(world.sensor_count, rng)
    total = 0
    learned_before = None

    print("step sensors reward choice      learned-rule       confidence surprise world")
    for step in range(steps):
        observation, reason = agent.choose(world.candidates())
        reward = world.reward(observation, step)
        total += reward
        surprise = agent.learn(observation, reward)
        rule, confidence = agent.best_rule()
        if step == change_at - 1:
            learned_before = str(rule)
        marker = " <-- CHANGED" if step == change_at else ""
        print(
            f"{step:>4} {observation}    {reward}    {reason:<10} "
            f"{str(rule):<18} {confidence:>7.1%} {surprise:>8.2f} "
            f"{world.rule_name(step)}{marker}"
        )
    final_rule, confidence = agent.best_rule()
    print(f"\nbefore={learned_before}; after={final_rule}; reward={total}/{steps}; confidence={confidence:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--change-at", type=int, default=35)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run(args.steps, args.change_at, args.seed)
