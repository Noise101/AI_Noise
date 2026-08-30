#!/usr/bin/env python3
"""A tiny non-LLM agent that actively discovers a changing causal rule."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


FEATURES = ("red", "blue", "circle", "square")


@dataclass(frozen=True)
class Object:
    color: str
    shape: str

    @property
    def features(self) -> frozenset[str]:
        return frozenset((self.color, self.shape))

    def __str__(self) -> str:
        return f"{self.color}-{self.shape}"


OBJECTS = tuple(
    Object(color, shape)
    for color in ("red", "blue")
    for shape in ("circle", "square")
)


class ChangingWorld:
    def __init__(self, first_rule: str, second_rule: str, change_at: int):
        self.first_rule = first_rule
        self.second_rule = second_rule
        self.change_at = change_at

    def rule_at(self, step: int) -> str:
        return self.first_rule if step < self.change_at else self.second_rule

    def act(self, obj: Object, step: int) -> int:
        return int(self.rule_at(step) in obj.features)


class CausalAgent:
    """Bayesian hypothesis learner with surprise-triggered adaptation."""

    def __init__(self, rng: random.Random, hazard: float = 0.08, explore: float = 1.0):
        self.rng = rng
        self.hazard = hazard
        self.explore = explore
        self.belief = {feature: 1.0 / len(FEATURES) for feature in FEATURES}

    @staticmethod
    def entropy(distribution: dict[str, float]) -> float:
        return -sum(p * math.log2(p) for p in distribution.values() if p > 0)

    def prediction(self, obj: Object) -> float:
        return sum(p for feature, p in self.belief.items() if feature in obj.features)

    def posterior_if(self, obj: Object, outcome: int) -> dict[str, float]:
        likelihood = {}
        for feature, prior in self.belief.items():
            predicted = int(feature in obj.features)
            likelihood[feature] = prior * (0.99 if predicted == outcome else 0.01)
        total = sum(likelihood.values())
        return {feature: value / total for feature, value in likelihood.items()}

    def information_gain(self, obj: Object) -> float:
        p_reward = self.prediction(obj)
        expected_entropy = 0.0
        for outcome, probability in ((1, p_reward), (0, 1.0 - p_reward)):
            if probability > 0:
                expected_entropy += probability * self.entropy(self.posterior_if(obj, outcome))
        return self.entropy(self.belief) - expected_entropy

    def choose(self) -> tuple[Object, str]:
        scored = []
        for obj in OBJECTS:
            reward_value = self.prediction(obj)
            experiment_value = self.information_gain(obj)
            score = reward_value + self.explore * experiment_value
            scored.append((score, self.rng.random(), obj))
        _, _, chosen = max(scored)
        reason = "experiment" if self.information_gain(chosen) > 0.15 else "exploit"
        return chosen, reason

    def learn(self, obj: Object, outcome: int) -> float:
        predicted_probability = self.prediction(obj)
        surprise = -math.log2(max(1e-9, predicted_probability if outcome else 1 - predicted_probability))

        # A small change probability prevents certainty from making relearning impossible.
        uniform = 1.0 / len(FEATURES)
        self.belief = {
            feature: (1.0 - self.hazard) * probability + self.hazard * uniform
            for feature, probability in self.belief.items()
        }
        self.belief = self.posterior_if(obj, outcome)
        return surprise

    def best_hypothesis(self) -> tuple[str, float]:
        feature = max(self.belief, key=self.belief.get)
        return feature, self.belief[feature]


def run(steps: int, change_at: int, seed: int) -> None:
    rng = random.Random(seed)
    world = ChangingWorld("red", "square", change_at)
    agent = CausalAgent(rng)
    total_reward = 0

    print("step  object         result  choice      hypothesis(conf)  surprise  world")
    for step in range(steps):
        obj, reason = agent.choose()
        outcome = world.act(obj, step)
        total_reward += outcome
        surprise = agent.learn(obj, outcome)
        hypothesis, confidence = agent.best_hypothesis()
        marker = "  <-- RULE CHANGED" if step == change_at else ""
        print(
            f"{step:>4}  {str(obj):<13}  {outcome:^6}  {reason:<10}  "
            f"{hypothesis:<8}({confidence:>4.0%})  {surprise:>7.2f}  "
            f"{world.rule_at(step)}{marker}"
        )

    hypothesis, confidence = agent.best_hypothesis()
    print(f"\nreward={total_reward}/{steps}; final hypothesis={hypothesis} ({confidence:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--change-at", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.steps, args.change_at, args.seed)
