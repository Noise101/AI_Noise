#!/usr/bin/env python3
"""Grow concepts from residual error instead of enumerating complete hypotheses."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    sensors: tuple[int, ...]


@dataclass(frozen=True)
class EmpiricalConcept:
    inputs: tuple[int, ...]
    table: tuple[int, ...]

    def predict(self, observation: Observation) -> int:
        address = 0
        for index in self.inputs:
            address = (address << 1) | observation.sensors[index]
        return self.table[address]

    def describe(self) -> str:
        return f"C({','.join(f's{i}' for i in self.inputs)})={''.join(map(str, self.table))}"


class ConstructiveAgent:
    """Tests only one-feature extensions of its current representation."""

    def __init__(self, sensor_count: int = 7, window: int = 60):
        self.sensor_count = sensor_count
        self.window = window
        self.history: list[tuple[Observation, int]] = []
        self.concept: EmpiricalConcept | None = None
        self.expansions: list[dict[str, object]] = []
        self.candidates_evaluated = 0

    @staticmethod
    def fit_table(inputs: tuple[int, ...], examples: list[tuple[Observation, int]]) -> EmpiricalConcept:
        bins = [[0, 0] for _ in range(2 ** len(inputs))]
        global_counts = [0, 0]
        for observation, outcome in examples:
            address = 0
            for index in inputs:
                address = (address << 1) | observation.sensors[index]
            bins[address][outcome] += 1
            global_counts[outcome] += 1
        default = int(global_counts[1] > global_counts[0])
        table = tuple(default if zeros == ones == 0 else int(ones > zeros) for zeros, ones in bins)
        return EmpiricalConcept(inputs, table)

    @staticmethod
    def description_length(concept: EmpiricalConcept, validation: list[tuple[Observation, int]]) -> float:
        errors = sum(concept.predict(observation) != outcome for observation, outcome in validation)
        # Encode exceptions plus the truth table itself. Growth must compress evidence overall.
        return errors * math.log2(len(validation) + 1) + len(concept.table) + len(concept.inputs)

    def predict(self, observation: Observation) -> int:
        return 0 if self.concept is None else self.concept.predict(observation)

    def observe(self, observation: Observation, outcome: int, step: int) -> None:
        self.history.append((observation, outcome))
        if len(self.history) < 18 or len(self.history) % 3:
            return
        recent = self.history[-self.window :]
        split = max(8, int(len(recent) * 0.65))
        train, validation = recent[:split], recent[split:]
        if not validation:
            return

        if self.concept is None:
            candidates = []
            for sensor in range(self.sensor_count):
                candidate = self.fit_table((sensor,), train)
                candidates.append((self.description_length(candidate, validation), sensor, candidate))
                self.candidates_evaluated += 1
            self.concept = min(candidates, key=lambda item: (item[0], item[1]))[2]
            return

        # Refit the current table, then ask which single new input best explains residuals.
        # Structural change detection uses a recent window, but established concept
        # semantics are reconstructed from episodic memory to avoid forgetting rare states.
        current = self.fit_table(self.concept.inputs, train)
        current_dl = self.description_length(current, validation)
        proposals = []
        for sensor in range(self.sensor_count):
            if sensor in current.inputs:
                continue
            inputs = tuple(sorted(current.inputs + (sensor,)))
            proposal = self.fit_table(inputs, train)
            proposals.append((self.description_length(proposal, validation), sensor, proposal))
            self.candidates_evaluated += 1
        if proposals:
            best_dl, sensor, best = min(proposals, key=lambda item: (item[0], item[1]))
            if best_dl < current_dl:  # MDL improvement, not comparison with a known rule
                old = self.concept.describe()
                consolidated = self.fit_table(best.inputs, self.history)
                self.concept = consolidated
                self.expansions.append({"step": step, "added": sensor, "old": old, "new": consolidated.describe(), "bits_saved": current_dl - best_dl})
                return
        self.concept = self.fit_table(current.inputs, self.history)


class GrowingParityWorld:
    """The observable support expands twice; the target function itself is always parity."""

    def __init__(self, reveal_second: int = 55, reveal_third: int = 115):
        self.reveal_second = reveal_second
        self.reveal_third = reveal_third

    def sample(self, rng: random.Random, step: int) -> Observation:
        sensors = [rng.randrange(2) for _ in range(7)]
        if step < self.reveal_second:
            sensors[3] = 0
        if step < self.reveal_third:
            sensors[4] = 0
        return Observation(tuple(sensors))

    @staticmethod
    def outcome(observation: Observation) -> int:
        return observation.sensors[0] ^ observation.sensors[3] ^ observation.sensors[4]


def exhaustive_accuracy(concept: EmpiricalConcept | None, world: GrowingParityWorld) -> float:
    if concept is None:
        return 0.0
    items = [Observation(bits) for bits in itertools.product((0, 1), repeat=7)]
    return sum(concept.predict(item) == world.outcome(item) for item in items) / len(items)


def simulate(seed: int, steps: int = 220) -> dict[str, object]:
    rng = random.Random(seed)
    world = GrowingParityWorld()
    agent = ConstructiveAgent()
    prequential_errors = 0
    for step in range(steps):
        observation = world.sample(rng, step)
        outcome = world.outcome(observation)
        prequential_errors += int(agent.predict(observation) != outcome)
        agent.observe(observation, outcome, step)
    return {
        "accuracy": exhaustive_accuracy(agent.concept, world),
        "concept": None if agent.concept is None else agent.concept.describe(),
        "inputs": () if agent.concept is None else agent.concept.inputs,
        "expansions": agent.expansions,
        "candidates_evaluated": agent.candidates_evaluated,
        "prequential_errors": prequential_errors,
    }


def evaluate(trials: int) -> None:
    results = [simulate(seed) for seed in range(trials)]
    accuracies = [float(result["accuracy"]) for result in results]
    candidate_counts = [int(result["candidates_evaluated"]) for result in results]
    print(f"trials={trials}")
    print(f"final exact concepts: {sum(a == 1 for a in accuracies)/trials:.1%}")
    print(f"mean accuracy:        {statistics.mean(accuracies):.1%}")
    print(f"median candidates:    {statistics.median(candidate_counts):.0f}")
    print(f"example concept:      {results[0]['concept']}")
    print(f"example expansions:   {results[0]['expansions']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.trials)
