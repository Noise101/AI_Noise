#!/usr/bin/env python3
"""Unified loop: active experiments, constructive concepts, temporal causes, self-revision."""

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
class CausalModel:
    sensor_inputs: tuple[int, ...]
    lag: int
    table: tuple[int, ...]  # address = selected sensor bits followed by action

    def cause_prediction(self, intervention: Intervention) -> int:
        address = 0
        for index in self.sensor_inputs:
            address = (address << 1) | intervention.sensors[index]
        address = (address << 1) | intervention.action
        return self.table[address]

    def outcome_prediction(self, history: list[Intervention], step: int) -> int:
        return 0 if step < self.lag else self.cause_prediction(history[step - self.lag])

    def describe(self) -> str:
        inputs = ",".join(f"s{i}" for i in self.sensor_inputs)
        return f"T({inputs},a)={''.join(map(str, self.table))} @-{self.lag}"


class UnifiedAgent:
    def __init__(self, sensor_count: int = 6, max_lag: int = 6, window: int = 96, active: bool = True):
        self.sensor_count = sensor_count
        self.max_lag = max_lag
        self.window = window
        self.active = active
        self.history: list[Intervention] = []
        self.outcomes: list[int] = []
        self.model: CausalModel | None = None
        self.local_alternatives: list[CausalModel] = []
        self.prequential_errors: list[int] = []
        self.revisions: list[dict[str, object]] = []
        self.candidates_evaluated = 0
        self._prediction_before_outcome = 0
        self.intervention_counts: dict[Intervention, int] = {}

    @staticmethod
    def fit_table(sensor_inputs: tuple[int, ...], examples: list[tuple[Intervention, int]], lag: int) -> CausalModel:
        bins = [[0, 0] for _ in range(2 ** (len(sensor_inputs) + 1))]
        totals = [0, 0]
        for intervention, outcome in examples:
            address = 0
            for index in sensor_inputs:
                address = (address << 1) | intervention.sensors[index]
            address = (address << 1) | intervention.action
            bins[address][outcome] += 1
            totals[outcome] += 1
        default = int(totals[1] > totals[0])
        table = tuple(default if zeros == ones == 0 else int(ones > zeros) for zeros, ones in bins)
        return CausalModel(sensor_inputs, lag, table)

    @staticmethod
    def description_length(model: CausalModel, validation: list[tuple[Intervention, int]]) -> float:
        errors = sum(model.cause_prediction(intervention) != outcome for intervention, outcome in validation)
        return errors * math.log2(len(validation) + 1) + len(model.table) + len(model.sensor_inputs) + math.log2(model.lag + 1)

    def predict_current_outcome(self, step: int) -> int:
        self._prediction_before_outcome = 0 if self.model is None else self.model.outcome_prediction(self.history, step)
        return self._prediction_before_outcome

    def observe_outcome(self, outcome: int, step: int) -> None:
        self.outcomes.append(outcome)
        self.prequential_errors.append(int(self._prediction_before_outcome != outcome))
        if len(self.outcomes) < self.max_lag + 18 or len(self.outcomes) % 3:
            return
        self._revise_model(step)

    def _structures(self) -> set[tuple[int, ...]]:
        if self.model is None:
            return {(sensor,) for sensor in range(self.sensor_count)}
        current = self.model.sensor_inputs
        structures = {current}
        for sensor in range(self.sensor_count):
            if sensor not in current:
                structures.add(tuple(sorted(current + (sensor,))))
                # A wrong feature must be replaceable without first accepting a
                # larger, temporarily more expensive representation.
                for displaced in current:
                    structures.add(tuple(sorted(tuple(index for index in current if index != displaced) + (sensor,))))
        if len(current) > 1:
            for sensor in current:
                structures.add(tuple(index for index in current if index != sensor))
        return structures

    def _revise_model(self, step: int) -> None:
        scored: list[tuple[float, str, CausalModel]] = []
        structures = self._structures()
        for lag in range(1, self.max_lag + 1):
            start = max(lag, len(self.outcomes) - self.window)
            aligned = [(self.history[t - lag], self.outcomes[t]) for t in range(start, len(self.outcomes))]
            split = max(10, int(len(aligned) * 0.65))
            train, validation = aligned[:split], aligned[split:]
            if not validation:
                continue
            for structure in structures:
                model = self.fit_table(structure, train, lag)
                dl = self.description_length(model, validation)
                scored.append((dl, model.describe(), model))
                self.candidates_evaluated += 1
        if not scored:
            return
        scored.sort(key=lambda item: (item[0], item[1]))
        selected = scored[0][2]
        # Do not rewrite a stable worldview for a negligible finite-sample gain.
        if self.model is not None:
            matching_current = [
                (dl, model) for dl, _, model in scored
                if model.sensor_inputs == self.model.sensor_inputs and model.lag == self.model.lag
            ]
            if matching_current and matching_current[0][0] <= scored[0][0] + 2.0:
                selected = matching_current[0][1]
        # Structure/lag selection uses held-out evidence; semantics are then consolidated
        # over the whole recent episode so rare truth-table cells are not discarded.
        selected_start = max(selected.lag, len(self.outcomes) - self.window)
        selected_examples = [(self.history[t - selected.lag], self.outcomes[t]) for t in range(selected_start, len(self.outcomes))]
        selected = self.fit_table(selected.sensor_inputs, selected_examples, selected.lag)
        old = None if self.model is None else self.model.describe()
        if old != selected.describe():
            self.revisions.append({"step": step, "old": old, "new": selected.describe(), "evidence_dl": scored[0][0]})
        self.model = selected
        # Keep near-best local alternatives for experiment selection, without all-model enumeration.
        best_dl = scored[0][0]
        self.local_alternatives = [model for dl, _, model in scored[:24] if dl <= best_dl + 4.0]

    def choose_intervention(self, rng: random.Random, available: tuple[int, ...]) -> Intervention:
        candidates = [
            Intervention(bits, action)
            for bits in itertools.product((0, 1), repeat=self.sensor_count)
            for action in (0, 1)
            if all(bits[index] == 0 for index in range(self.sensor_count) if index not in available)
        ]
        if not self.active or len(self.local_alternatives) < 2 or rng.random() < 0.18:
            intervention = rng.choice(candidates)
        else:
            scored = []
            for candidate in candidates:
                votes = [model.cause_prediction(candidate) for model in self.local_alternatives]
                p = sum(votes) / len(votes)
                entropy = 0.0 if p in (0, 1) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)
                current_dissent = 0.0 if self.model is None else sum(v != self.model.cause_prediction(candidate) for v in votes) / len(votes)
                visits = self.intervention_counts.get(candidate, 0)
                novelty = 1.0 / math.sqrt(visits + 1)
                scored.append((entropy + 0.15 * current_dissent + 0.30 * novelty, rng.random(), candidate))
            intervention = max(scored)[2]
        self.intervention_counts[intervention] = self.intervention_counts.get(intervention, 0) + 1
        self.history.append(intervention)
        return intervention


class NonStationaryWorld:
    def __init__(self, change_at: int = 120, reveal_s3: int = 35):
        self.change_at = change_at
        self.reveal_s3 = reveal_s3

    def available_sensors(self, step: int) -> tuple[int, ...]:
        if step < self.reveal_s3:
            return (0, 1, 2, 5)
        if step < self.change_at:
            return (0, 1, 2, 3, 5)
        return tuple(range(6))

    def lag(self, step: int) -> int:
        return 2 if step < self.change_at else 4

    @staticmethod
    def shared_concept(intervention: Intervention, include_s4: bool) -> int:
        value = intervention.sensors[0] ^ intervention.sensors[3]
        return value ^ intervention.sensors[4] if include_s4 else value

    def cause(self, intervention: Intervention, step: int) -> int:
        if step < self.change_at:
            return self.shared_concept(intervention, False) & intervention.action
        return self.shared_concept(intervention, True) & (1 - intervention.action)

    def outcome(self, history: list[Intervention], step: int) -> int:
        lag = self.lag(step)
        return 0 if step < lag else self.cause(history[step - lag], step)


def model_accuracy(model: CausalModel | None, world: NonStationaryWorld, step: int) -> float:
    if model is None or model.lag != world.lag(step):
        return 0.0
    items = [Intervention(bits, action) for bits in itertools.product((0, 1), repeat=6) for action in (0, 1)]
    return sum(model.cause_prediction(item) == world.cause(item, step) for item in items) / len(items)


def simulate(seed: int, active: bool, steps: int = 270, change_at: int = 120) -> dict[str, object]:
    rng = random.Random(seed)
    world = NonStationaryWorld(change_at)
    agent = UnifiedAgent(active=active)
    recovery = steps
    streak = 0
    for step in range(steps):
        agent.choose_intervention(rng, world.available_sensors(step))
        agent.predict_current_outcome(step)
        outcome = world.outcome(agent.history, step)
        agent.observe_outcome(outcome, step)
        if step >= change_at:
            if model_accuracy(agent.model, world, step) == 1.0:
                streak += 1
                if streak == 3:
                    recovery = step - change_at - 2
            else:
                streak = 0
    return {
        "accuracy": model_accuracy(agent.model, world, steps - 1),
        "recovery": recovery,
        "inputs": () if agent.model is None else agent.model.sensor_inputs,
        "lag": None if agent.model is None else agent.model.lag,
        "prequential_error_rate": sum(agent.prequential_errors) / len(agent.prequential_errors),
        "revisions": agent.revisions,
        "candidates_evaluated": agent.candidates_evaluated,
    }


def evaluate(trials: int) -> None:
    active_results = []
    passive_results = []
    for seed in range(trials):
        active_results.append(simulate(seed, True))
        passive_results.append(simulate(seed, False))
        if (seed + 1) % max(1, trials // 10) == 0:
            print(f"progress {seed + 1}/{trials}", flush=True)
    mean = lambda results, key: statistics.mean(float(result[key]) for result in results)
    recovered_active = [int(r["recovery"]) for r in active_results if r["recovery"] < 270]
    recovered_passive = [int(r["recovery"]) for r in passive_results if r["recovery"] < 270]
    print(f"active final accuracy:  {mean(active_results, 'accuracy'):.1%}")
    print(f"passive final accuracy: {mean(passive_results, 'accuracy'):.1%}")
    print(f"active recovered:       {len(recovered_active)/trials:.1%}")
    print(f"passive recovered:      {len(recovered_passive)/trials:.1%}")
    print(f"median recovery active/passive: {statistics.median(recovered_active) if recovered_active else 'inf'} / {statistics.median(recovered_passive) if recovered_passive else 'inf'}")
    print(f"example final: {active_results[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.trials)
