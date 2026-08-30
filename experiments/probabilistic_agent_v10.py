#!/usr/bin/env python3
"""Probabilistic causal learning under outcome noise and structural change."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Intervention:
    sensors: tuple[int, ...]
    action: int


@dataclass(frozen=True)
class ProbabilisticModel:
    sensor_inputs: tuple[int, ...]
    lag: int
    alpha: tuple[float, ...]
    beta: tuple[float, ...]

    def address(self, intervention: Intervention) -> int:
        address = 0
        for index in self.sensor_inputs:
            address = (address << 1) | intervention.sensors[index]
        return (address << 1) | intervention.action

    def probability(self, intervention: Intervention) -> float:
        address = self.address(intervention)
        return self.alpha[address] / (self.alpha[address] + self.beta[address])

    def uncertainty(self, intervention: Intervention) -> float:
        address = self.address(intervention)
        a, b = self.alpha[address], self.beta[address]
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def predict_outcome(self, history: list[Intervention], step: int) -> float:
        return 0.5 if step < self.lag else self.probability(history[step - self.lag])

    def describe(self) -> str:
        inputs = ",".join(f"s{i}" for i in self.sensor_inputs)
        return f"Beta({inputs},a) @-{self.lag}"


class ProbabilisticAgent:
    def __init__(self, sensor_count: int = 8, max_lag: int = 6, window: int = 240, active: bool = True):
        self.sensor_count = sensor_count
        self.max_lag = max_lag
        self.window = window
        self.active = active
        self.history: list[Intervention] = []
        self.outcomes: list[int] = []
        self.model: ProbabilisticModel | None = None
        self.alternatives: list[ProbabilisticModel] = []
        self.intervention_counts: dict[Intervention, int] = {}
        self.prequential_brier: list[float] = []
        self.last_probability = 0.5
        self.revisions: list[dict[str, object]] = []

    @staticmethod
    def fit_model(inputs: tuple[int, ...], lag: int, examples: list[tuple[Intervention, int]]) -> ProbabilisticModel:
        cells = 2 ** (len(inputs) + 1)
        alpha = [1.0] * cells
        beta = [1.0] * cells
        for intervention, outcome in examples:
            address = 0
            for index in inputs:
                address = (address << 1) | intervention.sensors[index]
            address = (address << 1) | intervention.action
            alpha[address] += outcome
            beta[address] += 1 - outcome
        return ProbabilisticModel(inputs, lag, tuple(alpha), tuple(beta))

    @staticmethod
    def score(model: ProbabilisticModel, validation: list[tuple[Intervention, int]]) -> float:
        log_loss = 0.0
        for intervention, outcome in validation:
            probability = min(1 - 1e-6, max(1e-6, model.probability(intervention)))
            log_loss -= outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)
        # BIC/MDL penalty: each probability cell is a learned degree of freedom.
        parameters = len(model.alpha)
        return log_loss + 0.10 * parameters * math.log(len(validation) + 1) + math.log(model.lag + 1)

    def _structures(self) -> set[tuple[int, ...]]:
        if self.model is None:
            return {(sensor,) for sensor in range(self.sensor_count)}
        current = self.model.sensor_inputs
        structures = {current}
        for sensor in range(self.sensor_count):
            if sensor in current:
                continue
            structures.add(tuple(sorted(current + (sensor,))))
            for displaced in current:
                structures.add(tuple(sorted(tuple(index for index in current if index != displaced) + (sensor,))))
        if len(current) > 1:
            for sensor in current:
                structures.add(tuple(index for index in current if index != sensor))
        return structures

    def choose_intervention(self, rng: random.Random, available: tuple[int, ...]) -> Intervention:
        candidates = [
            Intervention(bits, action)
            for bits in itertools.product((0, 1), repeat=self.sensor_count)
            for action in (0, 1)
            if all(bits[index] == 0 for index in range(self.sensor_count) if index not in available)
        ]
        if not self.active or not self.alternatives or rng.random() < 0.45:
            chosen = rng.choice(candidates)
        else:
            scored = []
            for candidate in candidates:
                probabilities = [model.probability(candidate) for model in self.alternatives]
                mean = sum(probabilities) / len(probabilities)
                disagreement = sum((p - mean) ** 2 for p in probabilities) / len(probabilities)
                uncertainty = 0.0 if self.model is None else self.model.uncertainty(candidate)
                novelty = 1 / math.sqrt(self.intervention_counts.get(candidate, 0) + 1)
                scored.append((disagreement + uncertainty + 0.05 * novelty, rng.random(), candidate))
            chosen = max(scored)[2]
        self.intervention_counts[chosen] = self.intervention_counts.get(chosen, 0) + 1
        self.history.append(chosen)
        return chosen

    def predict_before_outcome(self, step: int) -> float:
        self.last_probability = 0.5 if self.model is None else self.model.predict_outcome(self.history, step)
        return self.last_probability

    def observe(self, outcome: int, step: int) -> None:
        self.outcomes.append(outcome)
        self.prequential_brier.append((self.last_probability - outcome) ** 2)
        if len(self.outcomes) < 90 or len(self.outcomes) % 5:
            return
        scored: list[tuple[float, str, ProbabilisticModel]] = []
        structures = self._structures()
        for lag in range(1, self.max_lag + 1):
            start = max(lag, len(self.outcomes) - self.window)
            aligned = [(self.history[t - lag], self.outcomes[t]) for t in range(start, len(self.outcomes))]
            split = max(40, int(len(aligned) * 0.65))
            train, validation = aligned[:split], aligned[split:]
            if len(validation) < 8:
                continue
            for inputs in structures:
                model = self.fit_model(inputs, lag, train)
                scored.append((self.score(model, validation), model.describe(), model))
        if not scored:
            return
        scored.sort(key=lambda item: (item[0], item[1]))
        selected_score, _, selected = scored[0]
        if self.model is not None:
            current = [
                item for item in scored
                if item[2].sensor_inputs == self.model.sensor_inputs and item[2].lag == self.model.lag
            ]
            if current:
                current_score = current[0][0]
                constructive = [
                    item for item in scored
                    if set(self.model.sensor_inputs).issubset(item[2].sensor_inputs)
                    and (
                        len(item[2].sensor_inputs) > len(self.model.sensor_inputs)
                        or item[2].lag != self.model.lag
                    )
                ]
                best_constructive = min(constructive, key=lambda item: item[0]) if constructive else None
                if best_constructive is not None and best_constructive[0] + 2.0 < current_score:
                    selected_score, _, selected = best_constructive
                elif current_score <= selected_score + 8.0:
                    selected_score, _, selected = current[0]
        start = max(selected.lag, len(self.outcomes) - self.window)
        all_recent = [(self.history[t - selected.lag], self.outcomes[t]) for t in range(start, len(self.outcomes))]
        consolidated = self.fit_model(selected.sensor_inputs, selected.lag, all_recent)
        old = None if self.model is None else self.model.describe()
        if old != consolidated.describe():
            self.revisions.append({"step": step, "old": old, "new": consolidated.describe(), "score": selected_score})
        self.model = consolidated
        best = scored[0][0]
        self.alternatives = [model for score, _, model in scored[:32] if score <= best + 3.0]


class NoisyWorld:
    def __init__(self, change_at: int = 300, high: float = 0.90, low: float = 0.10):
        self.change_at = change_at
        self.high = high
        self.low = low

    def available(self, step: int) -> tuple[int, ...]:
        if step < 140:
            return (0, 1, 2, 5, 6, 7)
        if step < self.change_at:
            return (0, 1, 2, 3, 5, 6, 7)
        return tuple(range(8))

    def lag(self, step: int) -> int:
        return 3 if step < self.change_at else 5

    def cause(self, intervention: Intervention, step: int) -> int:
        concept = intervention.sensors[0] ^ intervention.sensors[3]
        if step >= self.change_at:
            concept ^= intervention.sensors[4]
            return concept & (1 - intervention.action)
        return concept & intervention.action

    def probability(self, intervention: Intervention, step: int) -> float:
        return self.high if self.cause(intervention, step) else self.low

    def outcome(self, history: list[Intervention], step: int, rng: random.Random) -> int:
        lag = self.lag(step)
        probability = 0.5 if step < lag else self.probability(history[step - lag], step)
        return int(rng.random() < probability)


def calibration_brier(model: ProbabilisticModel | None, world: NoisyWorld, step: int) -> float:
    if model is None or model.lag != world.lag(step):
        return 0.25
    items = [Intervention(bits, action) for bits in itertools.product((0, 1), repeat=8) for action in (0, 1)]
    return statistics.mean((model.probability(item) - world.probability(item, step)) ** 2 for item in items)


def simulate(seed: int, active: bool, steps: int = 800) -> dict[str, object]:
    rng = random.Random(seed)
    world = NoisyWorld()
    agent = ProbabilisticAgent(active=active)
    for step in range(steps):
        agent.choose_intervention(rng, world.available(step))
        agent.predict_before_outcome(step)
        agent.observe(world.outcome(agent.history, step, rng), step)
    return {
        "brier": calibration_brier(agent.model, world, steps - 1),
        "inputs": () if agent.model is None else agent.model.sensor_inputs,
        "lag": None if agent.model is None else agent.model.lag,
        "prequential_brier": statistics.mean(agent.prequential_brier),
        "revisions": agent.revisions,
    }


def evaluate(trials: int) -> None:
    active_results, passive_results = [], []
    for seed in range(trials):
        active_results.append(simulate(seed, True))
        passive_results.append(simulate(seed, False))
        if (seed + 1) % max(1, trials // 10) == 0:
            print(f"progress {seed + 1}/{trials}", flush=True)
    exact = lambda results: sum(set(r["inputs"]) == {0, 3, 4} and r["lag"] == 5 for r in results) / trials
    mean = lambda results, key: statistics.mean(float(r[key]) for r in results)
    print(f"active structure recovery:  {exact(active_results):.1%}")
    print(f"passive structure recovery: {exact(passive_results):.1%}")
    print(f"active calibration Brier:   {mean(active_results, 'brier'):.4f}")
    print(f"passive calibration Brier:  {mean(passive_results, 'brier'):.4f}")
    print(f"example: {active_results[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.trials)
