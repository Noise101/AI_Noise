#!/usr/bin/env python3
"""Integrated continual agent: temporal discovery + learned truth-table concepts."""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass

from truth_table_concepts_v4 import (
    AtomicFeature,
    Feature,
    Situation,
    TableConcept,
    TableLearner,
    TableModel,
    factor_action,
)


@dataclass(frozen=True)
class TemporalModel:
    lag: int
    table_model: TableModel

    def predict(self, situations: list[Situation], target_step: int) -> int:
        if target_step < self.lag:
            return 0
        return self.table_model.predict(situations[target_step - self.lag])

    def describe(self) -> str:
        return f"{self.table_model.describe()} @-{self.lag}"


class TemporalTableLearner:
    def __init__(self, features: list[Feature], max_inputs: int, max_lag: int = 7, window: int = 64):
        self.features = features
        self.max_inputs = max_inputs
        self.max_lag = max_lag
        self.window = window
        self.model: TemporalModel | None = None

    def fit(self, situations: list[Situation], outcomes: list[int]) -> TemporalModel | None:
        if len(outcomes) < self.max_lag + 12:
            return None
        candidates = []
        for lag in range(1, self.max_lag + 1):
            start = max(lag, len(outcomes) - self.window)
            examples = [(situations[t - lag], outcomes[t]) for t in range(start, len(outcomes))]
            learner = TableLearner(self.features, self.max_inputs)
            model = learner.fit(examples)
            score = learner.score(model, examples)
            candidates.append((score, -lag, model.describe(), TemporalModel(lag, model)))
        self.model = max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
        return self.model


class IntegratedAgent:
    def __init__(self, sensor_count: int = 7, transfer: bool = True):
        self.atoms: list[Feature] = [AtomicFeature(i) for i in range(sensor_count)] + [AtomicFeature(-1)]
        self.features = list(self.atoms)
        self.transfer = transfer
        self.concepts: list[TableConcept] = []
        self.situations: list[Situation] = []
        self.outcomes: list[int] = []
        self.stable_description: str | None = None
        self.stable_steps = 0
        self.model: TemporalModel | None = None
        self.skill_archive: list[TemporalModel] = []
        self.update_every = 3

    def observe(self, situation: Situation, outcome: int) -> None:
        self.situations.append(situation)
        self.outcomes.append(outcome)
        if len(self.outcomes) % self.update_every != 0:
            return
        learner = TemporalTableLearner(self.features, max_inputs=3 if self.transfer else 4)
        self.model = learner.fit(self.situations, self.outcomes)
        if self.model is None:
            return
        description = self.model.describe()
        if description == self.stable_description:
            self.stable_steps += 1
        else:
            self.stable_description = description
            self.stable_steps = 1
        # Stability is counted across spaced re-fits, not adjacent near-identical windows.
        if self.transfer and not self.concepts and self.stable_steps >= 7:
            concept = factor_action(self.model.table_model)
            if concept is not None:
                self.concepts.append(concept)
                self.features.append(concept)
                self.skill_archive.append(self.model)


class ContinualWorld:
    def __init__(self, change_at: int = 95):
        self.change_at = change_at

    @staticmethod
    def concept(situation: Situation) -> int:
        return situation.sensors[0] ^ situation.sensors[3]

    def lag(self, step: int) -> int:
        return 3 if step < self.change_at else 5

    def cause(self, situation: Situation, step: int) -> int:
        if step < self.change_at:
            return self.concept(situation) & situation.action
        return self.concept(situation) & situation.sensors[5] & (1 - situation.action)

    def outcome(self, history: list[Situation], step: int) -> int:
        lag = self.lag(step)
        return 0 if step < lag else self.cause(history[step - lag], step)


def model_accuracy(model: TemporalModel | None, world: ContinualWorld, phase_step: int) -> float:
    if model is None or model.lag != world.lag(phase_step):
        return 0.0
    items = [Situation(bits, action) for bits in itertools.product((0, 1), repeat=7) for action in (0, 1)]
    return sum(model.table_model.predict(item) == world.cause(item, phase_step) for item in items) / len(items)


def simulate(seed: int, steps: int = 210, change_at: int = 95) -> dict[str, object]:
    rng = random.Random(seed)
    world = ContinualWorld(change_at)
    transfer = IntegratedAgent(transfer=True)
    fresh = IntegratedAgent(transfer=False)
    history: list[Situation] = []
    recovery = {"transfer": steps, "fresh": steps}
    streak = {"transfer": 0, "fresh": 0}
    before = {}

    for step in range(steps):
        situation = Situation(tuple(rng.randrange(2) for _ in range(7)), rng.randrange(2))
        history.append(situation)
        outcome = world.outcome(history, step)
        transfer.observe(situation, outcome)
        fresh.observe(situation, outcome)
        if step == change_at - 1:
            before = {
                "transfer": model_accuracy(transfer.model, world, step),
                "fresh": model_accuracy(fresh.model, world, step),
            }
        if step >= change_at:
            for name, agent in (("transfer", transfer), ("fresh", fresh)):
                if model_accuracy(agent.model, world, step) == 1.0:
                    streak[name] += 1
                    if streak[name] == 5:
                        recovery[name] = step - change_at - 4
                else:
                    streak[name] = 0

    concept_accuracy = 0.0
    if transfer.concepts:
        items = [Situation(bits, 0) for bits in itertools.product((0, 1), repeat=7)]
        concept_accuracy = sum(transfer.concepts[0].value(item) == world.concept(item) for item in items) / len(items)
    return {
        "before_transfer": before.get("transfer", 0.0),
        "before_fresh": before.get("fresh", 0.0),
        "after_transfer": model_accuracy(transfer.model, world, steps - 1),
        "after_fresh": model_accuracy(fresh.model, world, steps - 1),
        "recovery_transfer": recovery["transfer"],
        "recovery_fresh": recovery["fresh"],
        "concept_accuracy_after": concept_accuracy,
        "concept": None if not transfer.concepts else transfer.concepts[0].describe(),
    }


def evaluate(trials: int) -> None:
    results = []
    started = time.monotonic()
    report_every = max(1, trials // 20)
    for seed in range(trials):
        results.append(simulate(seed))
        completed = seed + 1
        if completed % report_every == 0 or completed == trials:
            elapsed = time.monotonic() - started
            rate = completed / elapsed
            remaining = (trials - completed) / rate if rate else 0.0
            print(
                f"progress {completed:>4}/{trials} ({completed/trials:>6.1%}) | "
                f"elapsed {elapsed:>6.1f}s | eta {remaining:>6.1f}s",
                flush=True,
            )
    mean = lambda key: sum(float(result[key]) for result in results) / trials
    recovered_t = [float(r["recovery_transfer"]) for r in results if r["recovery_transfer"] < 210]
    recovered_f = [float(r["recovery_fresh"]) for r in results if r["recovery_fresh"] < 210]
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("inf")
    print(f"trials={trials}")
    print(f"phase1 exact transfer/fresh: {mean('before_transfer'):.1%} / {mean('before_fresh'):.1%}")
    print(f"phase2 final transfer/fresh: {mean('after_transfer'):.1%} / {mean('after_fresh'):.1%}")
    print(f"phase2 recovery median:      {median(recovered_t):.0f} / {median(recovered_f):.0f} steps")
    print(f"concept retained after task: {mean('concept_accuracy_after'):.1%}")
    print(f"example={results[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.trials)
