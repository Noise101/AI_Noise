#!/usr/bin/env python3
"""Grammar-light concept learning with empirical truth tables, including XOR."""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Situation:
    sensors: tuple[int, ...]
    action: int


class Feature(Protocol):
    @property
    def name(self) -> str: ...
    def value(self, situation: Situation) -> int: ...


@dataclass(frozen=True)
class AtomicFeature:
    index: int  # -1 denotes action

    @property
    def name(self) -> str:
        return "action" if self.index == -1 else f"s{self.index}"

    def value(self, situation: Situation) -> int:
        return situation.action if self.index == -1 else situation.sensors[self.index]


@dataclass(frozen=True)
class TableConcept:
    concept_name: str
    inputs: tuple[Feature, ...]
    table: tuple[int, ...]

    @property
    def name(self) -> str:
        return self.concept_name

    def value(self, situation: Situation) -> int:
        address = 0
        for feature in self.inputs:
            address = (address << 1) | feature.value(situation)
        return self.table[address]

    def describe(self) -> str:
        return f"{self.name}({','.join(f.name for f in self.inputs)})={''.join(map(str, self.table))}"


@dataclass(frozen=True)
class TableModel:
    features: tuple[Feature, ...]
    table: tuple[int, ...]

    def predict(self, situation: Situation) -> int:
        address = 0
        for feature in self.features:
            address = (address << 1) | feature.value(situation)
        return self.table[address]

    def describe(self) -> str:
        return f"T({','.join(f.name for f in self.features)})={''.join(map(str, self.table))}"


class TableLearner:
    """Selects observed variables; the Boolean function itself is learned, not predefined."""

    def __init__(self, features: list[Feature], max_inputs: int):
        self.features = features
        self.max_inputs = max_inputs

    @staticmethod
    def fit_table(features: tuple[Feature, ...], examples: list[tuple[Situation, int]]) -> tuple[int, ...]:
        bins = [[0, 0] for _ in range(2 ** len(features))]
        for situation, label in examples:
            address = 0
            for feature in features:
                address = (address << 1) | feature.value(situation)
            bins[address][label] += 1
        # Conservative default for unseen states: no effect.
        return tuple(int(ones > zeros) for zeros, ones in bins)

    @staticmethod
    def score(model: TableModel, examples: list[tuple[Situation, int]]) -> float:
        errors = sum(model.predict(situation) != label for situation, label in examples)
        used_entries = len(set(tuple(feature.value(s) for feature in model.features) for s, _ in examples))
        return -3.2 * errors - 0.30 * len(model.features) - 0.045 * used_entries

    def fit(self, examples: list[tuple[Situation, int]]) -> TableModel:
        candidates = []
        for width in range(1, self.max_inputs + 1):
            for features in itertools.combinations(self.features, width):
                table = self.fit_table(features, examples)
                model = TableModel(features, table)
                candidates.append((self.score(model, examples), model.describe(), model))
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


def factor_action(model: TableModel) -> TableConcept | None:
    """Extract a reusable sensor function when action is merely an enable/disable gate."""
    action_positions = [i for i, feature in enumerate(model.features) if feature.name == "action"]
    if len(action_positions) != 1:
        return None
    action_pos = action_positions[0]
    sensor_features = tuple(feature for feature in model.features if feature.name != "action")
    if not sensor_features:
        return None
    derived_tables = []
    for action_value in (0, 1):
        values = []
        for sensor_bits in itertools.product((0, 1), repeat=len(sensor_features)):
            full = list(sensor_bits)
            full.insert(action_pos, action_value)
            address = 0
            for bit in full:
                address = (address << 1) | bit
            values.append(model.table[address])
        derived_tables.append(tuple(values))
    if derived_tables[0] == tuple(0 for _ in derived_tables[0]) and len(set(derived_tables[1])) > 1:
        return TableConcept("C0", sensor_features, derived_tables[1])
    if derived_tables[1] == tuple(0 for _ in derived_tables[1]) and len(set(derived_tables[0])) > 1:
        return TableConcept("C0", sensor_features, derived_tables[0])
    return None


def space(sensor_count: int = 7) -> list[Situation]:
    return [Situation(bits, action) for bits in itertools.product((0, 1), repeat=sensor_count) for action in (0, 1)]


def xor_concept(situation: Situation) -> int:
    return situation.sensors[0] ^ situation.sensors[3]


def task1(situation: Situation) -> int:
    return xor_concept(situation) & situation.action


def task2(situation: Situation) -> int:
    return xor_concept(situation) & situation.sensors[5] & (1 - situation.action)


def accuracy(model: TableModel, target, items: list[Situation]) -> float:
    return sum(model.predict(item) == target(item) for item in items) / len(items)


def trial(seed: int, first_samples: int = 64, transfer_samples: int = 24) -> dict[str, object]:
    rng = random.Random(seed)
    items = space()
    atoms: list[Feature] = [AtomicFeature(i) for i in range(7)] + [AtomicFeature(-1)]
    examples1 = [(item, task1(item)) for item in rng.sample(items, first_samples)]
    first = TableLearner(atoms, max_inputs=3).fit(examples1)
    concept = factor_action(first)

    examples2 = [(item, task2(item)) for item in rng.sample(items, transfer_samples)]
    transfer_features = atoms + ([] if concept is None else [concept])
    transferred = TableLearner(transfer_features, max_inputs=3).fit(examples2)
    fresh = TableLearner(atoms, max_inputs=4).fit(examples2)
    return {
        "first": first.describe(),
        "concept": None if concept is None else concept.describe(),
        "transferred": transferred.describe(),
        "fresh": fresh.describe(),
        "transfer_accuracy": accuracy(transferred, task2, items),
        "fresh_accuracy": accuracy(fresh, task2, items),
    }


def evaluate(trials: int) -> None:
    results = [trial(seed) for seed in range(trials)]
    transfer = [float(r["transfer_accuracy"]) for r in results]
    fresh = [float(r["fresh_accuracy"]) for r in results]
    concepts = sum(r["concept"] is not None for r in results)
    print(f"trials={trials}; concepts formed={concepts/trials:.1%}")
    print(f"truth-table transfer={sum(transfer)/trials:.1%}")
    print(f"truth-table fresh=   {sum(fresh)/trials:.1%}")
    print(f"transfer wins=       {sum(a>b for a,b in zip(transfer,fresh))/trials:.1%}")
    print(f"example={results[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.trials)
