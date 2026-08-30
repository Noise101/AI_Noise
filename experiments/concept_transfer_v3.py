#!/usr/bin/env python3
"""Learn a reusable conjunctive concept and measure few-shot transfer."""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Situation:
    sensors: tuple[int, ...]
    action: int


class Predicate(Protocol):
    def matches(self, situation: Situation) -> bool: ...
    @property
    def key(self) -> str: ...


@dataclass(frozen=True)
class Literal:
    source: str
    index: int
    value: int

    def matches(self, situation: Situation) -> bool:
        actual = situation.action if self.source == "action" else situation.sensors[self.index]
        return actual == self.value

    @property
    def key(self) -> str:
        prefix = "a" if self.source == "action" else f"s{self.index}"
        return prefix if self.value else f"!{prefix}"


@dataclass(frozen=True)
class Concept:
    name: str
    parts: tuple[Literal, ...]

    def matches(self, situation: Situation) -> bool:
        return all(part.matches(situation) for part in self.parts)

    @property
    def key(self) -> str:
        return self.name


@dataclass(frozen=True)
class ConjunctiveRule:
    predicates: tuple[Predicate, ...]

    def predicts(self, situation: Situation) -> int:
        return int(all(predicate.matches(situation) for predicate in self.predicates))

    def __str__(self) -> str:
        return " & ".join(predicate.key for predicate in self.predicates)


class ConceptLibrary:
    def __init__(self):
        self.concepts: list[Concept] = []

    def add_from_rule(self, rule: ConjunctiveRule) -> Concept | None:
        # Action-independent co-causes are reusable across goals.
        parts = tuple(
            predicate for predicate in rule.predicates
            if isinstance(predicate, Literal) and predicate.source == "sensor"
        )
        if len(parts) < 2:
            return None
        concept = Concept(f"C{len(self.concepts)}", parts)
        if all(existing.parts != concept.parts for existing in self.concepts):
            self.concepts.append(concept)
            return concept
        return None


class RuleLearner:
    def __init__(self, sensor_count: int, library: ConceptLibrary | None = None, max_width: int = 4):
        literals: list[Predicate] = [
            Literal("sensor", sensor, value)
            for sensor in range(sensor_count)
            for value in (0, 1)
        ] + [Literal("action", 0, value) for value in (0, 1)]
        self.predicates = literals + ([] if library is None else list(library.concepts))
        self.max_width = max_width
        self.rule: ConjunctiveRule | None = None

    @staticmethod
    def _valid(combo: tuple[Predicate, ...]) -> bool:
        keys = [predicate.key.lstrip("!") for predicate in combo]
        return len(keys) == len(set(keys))

    @staticmethod
    def _score(rule: ConjunctiveRule, examples: list[tuple[Situation, int]]) -> float:
        errors = sum(rule.predicts(situation) != label for situation, label in examples)
        # MDL-like prior: reusable concepts cost one symbol in a new task.
        return -3.0 * errors - 0.72 * len(rule.predicates)

    def fit(self, examples: list[tuple[Situation, int]]) -> ConjunctiveRule:
        candidates = []
        for width in range(1, self.max_width + 1):
            for combo in itertools.combinations(self.predicates, width):
                if not self._valid(combo):
                    continue
                rule = ConjunctiveRule(combo)
                candidates.append((self._score(rule, examples), str(rule), rule))
        _, _, self.rule = max(candidates, key=lambda item: (item[0], item[1]))
        return self.rule


def situation_space(sensor_count: int = 7) -> list[Situation]:
    return [
        Situation(bits, action)
        for bits in itertools.product((0, 1), repeat=sensor_count)
        for action in (0, 1)
    ]


def shared_concept(situation: Situation) -> bool:
    return situation.sensors[0] == 1 and situation.sensors[3] == 0


def task_one(situation: Situation) -> int:
    return int(shared_concept(situation) and situation.action == 1)


def task_two(situation: Situation) -> int:
    return int(shared_concept(situation) and situation.sensors[5] == 1 and situation.action == 0)


def unrelated_task(situation: Situation) -> int:
    return int(situation.sensors[1] == 1 and situation.sensors[6] == 0 and situation.action == 1)


def accuracy(rule: ConjunctiveRule, labeler, space: list[Situation]) -> float:
    return sum(rule.predicts(item) == labeler(item) for item in space) / len(space)


def trial(seed: int, task1_samples: int = 48, task2_samples: int = 18) -> dict[str, object]:
    rng = random.Random(seed)
    space = situation_space()
    train1 = [(item, task_one(item)) for item in rng.sample(space, task1_samples)]
    first = RuleLearner(7, max_width=3)
    first_rule = first.fit(train1)

    library = ConceptLibrary()
    concept = library.add_from_rule(first_rule)

    train2 = [(item, task_two(item)) for item in rng.sample(space, task2_samples)]
    transfer = RuleLearner(7, library=library, max_width=4).fit(train2)
    fresh = RuleLearner(7, max_width=4).fit(train2)

    # Negative-transfer control: this target shares nothing with C0.
    train3 = [(item, unrelated_task(item)) for item in rng.sample(space, task2_samples)]
    unrelated_transfer = RuleLearner(7, library=library, max_width=4).fit(train3)
    unrelated_fresh = RuleLearner(7, max_width=4).fit(train3)
    return {
        "concept": None if concept is None else concept.key + "=" + " & ".join(p.key for p in concept.parts),
        "first_rule": str(first_rule),
        "transfer_rule": str(transfer),
        "fresh_rule": str(fresh),
        "transfer_accuracy": accuracy(transfer, task_two, space),
        "fresh_accuracy": accuracy(fresh, task_two, space),
        "unrelated_transfer_accuracy": accuracy(unrelated_transfer, unrelated_task, space),
        "unrelated_fresh_accuracy": accuracy(unrelated_fresh, unrelated_task, space),
    }


def evaluate(trials: int) -> None:
    results = [trial(seed) for seed in range(trials)]
    transfer = [float(result["transfer_accuracy"]) for result in results]
    fresh = [float(result["fresh_accuracy"]) for result in results]
    unrelated_transfer = [float(result["unrelated_transfer_accuracy"]) for result in results]
    unrelated_fresh = [float(result["unrelated_fresh_accuracy"]) for result in results]
    print(f"trials={trials}")
    print(f"mean transfer accuracy: {sum(transfer) / trials:.1%}")
    print(f"mean fresh accuracy:    {sum(fresh) / trials:.1%}")
    print(f"perfect transfer:       {sum(x == 1 for x in transfer) / trials:.1%}")
    print(f"perfect fresh:          {sum(x == 1 for x in fresh) / trials:.1%}")
    print(f"transfer wins:          {sum(a > b for a, b in zip(transfer, fresh)) / trials:.1%}")
    print(f"unrelated w/ concept:   {sum(unrelated_transfer) / trials:.1%}")
    print(f"unrelated fresh:        {sum(unrelated_fresh) / trials:.1%}")
    print(f"example: {results[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.trials)
