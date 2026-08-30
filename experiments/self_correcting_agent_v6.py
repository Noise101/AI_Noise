#!/usr/bin/env python3
"""An agent that falsifies and revises its own learned concepts online."""

from __future__ import annotations

import argparse
import itertools
import random
from collections import deque

from truth_table_concepts_v4 import (
    AtomicFeature,
    Situation,
    TableConcept,
    TableLearner,
    TableModel,
    factor_action,
)


class SelfCorrectingAgent:
    """Competes a compressed belief against a less biased discovery model."""

    def __init__(self, sensor_count: int = 5, window: int = 48):
        self.atoms = [AtomicFeature(i) for i in range(sensor_count)] + [AtomicFeature(-1)]
        self.window = window
        self.examples: list[tuple[Situation, int]] = []
        self.discovery: TableModel | None = None
        self.compressed: TableModel | None = None
        self.concept: TableConcept | None = None
        self.discovery_errors: deque[int] = deque(maxlen=24)
        self.compressed_errors: deque[int] = deque(maxlen=24)
        self.revision_votes = 0
        self.revisions = 0
        self.last_prediction: dict[str, int] = {}
        self.stable_discovery: str | None = None
        self.stable_steps = 0
        self.event_log: list[dict[str, object]] = []

    def predict_before_observing(self, situation: Situation) -> int:
        predictions = {}
        if self.discovery is not None:
            predictions["discovery"] = self.discovery.predict(situation)
        if self.compressed is not None:
            predictions["compressed"] = self.compressed.predict(situation)
        self.last_prediction = predictions
        if not predictions:
            return 0
        # Prefer the compressed explanation while predictive evidence is tied.
        if "compressed" in predictions and self._error_rate(self.compressed_errors) <= self._error_rate(self.discovery_errors) + 0.02:
            return predictions["compressed"]
        return predictions.get("discovery", predictions.get("compressed", 0))

    @staticmethod
    def _error_rate(errors: deque[int]) -> float:
        return (sum(errors) + 1) / (len(errors) + 2)  # Beta(1,1) smoothing

    def observe(self, situation: Situation, outcome: int, step: int) -> None:
        for name, prediction in self.last_prediction.items():
            errors = self.discovery_errors if name == "discovery" else self.compressed_errors
            errors.append(int(prediction != outcome))
        self.examples.append((situation, outcome))
        if len(self.examples) < 12 or len(self.examples) % 3:
            return

        recent = self.examples[-self.window :]
        self.discovery = TableLearner(self.atoms, max_inputs=3).fit(recent)
        description = self.discovery.describe()
        if description == self.stable_discovery:
            self.stable_steps += 1
        else:
            self.stable_discovery = description
            self.stable_steps = 1

        candidate = factor_action(self.discovery)
        if self.concept is None and candidate is not None and self.stable_steps >= 4:
            self._install_concept(candidate, step, "formed")
        elif self.concept is not None:
            self.compressed = TableLearner([self.concept, AtomicFeature(-1)], max_inputs=2).fit(recent)
            discovery_error = self._error_rate(self.discovery_errors)
            compressed_error = self._error_rate(self.compressed_errors)
            # Revision is earned by repeated out-of-sample superiority, not a known answer.
            if candidate is not None and candidate.describe() != self.concept.describe() and discovery_error + 0.08 < compressed_error:
                self.revision_votes += 1
            else:
                self.revision_votes = max(0, self.revision_votes - 1)
            if self.revision_votes >= 3:
                old = self.concept.describe()
                self._install_concept(candidate, step, "revised", old=old)
                self.revisions += 1
                self.revision_votes = 0

    def _install_concept(self, concept: TableConcept, step: int, event: str, old: str | None = None) -> None:
        self.concept = TableConcept("C0", concept.inputs, concept.table)
        self.compressed = None
        self.compressed_errors.clear()
        self.event_log.append({"step": step, "event": event, "old": old, "new": self.concept.describe()})


class ExpandingWorld:
    """Early observations hide a variable; later evidence reveals the first concept was incomplete."""

    def __init__(self, reveal_at: int = 70):
        self.reveal_at = reveal_at

    def sample(self, rng: random.Random, step: int) -> Situation:
        sensors = [rng.randrange(2) for _ in range(5)]
        if step < self.reveal_at:
            sensors[3] = 0  # biased experience makes C=s0 look sufficient
        return Situation(tuple(sensors), rng.randrange(2))

    @staticmethod
    def outcome(situation: Situation) -> int:
        hidden_concept = situation.sensors[0] ^ situation.sensors[3]
        return hidden_concept & situation.action


def concept_accuracy(concept: TableConcept | None) -> float:
    if concept is None:
        return 0.0
    items = [Situation(bits, 0) for bits in itertools.product((0, 1), repeat=5)]
    return sum(concept.value(item) == (item.sensors[0] ^ item.sensors[3]) for item in items) / len(items)


def simulate(seed: int, steps: int = 160, reveal_at: int = 70) -> dict[str, object]:
    rng = random.Random(seed)
    world = ExpandingWorld(reveal_at)
    agent = SelfCorrectingAgent()
    mistakes_before = mistakes_after = 0
    for step in range(steps):
        situation = world.sample(rng, step)
        prediction = agent.predict_before_observing(situation)
        outcome = world.outcome(situation)
        if prediction != outcome:
            if step < reveal_at:
                mistakes_before += 1
            else:
                mistakes_after += 1
        agent.observe(situation, outcome, step)
    revision_steps = [int(event["step"]) for event in agent.event_log if event["event"] == "revised"]
    return {
        "concept_accuracy": concept_accuracy(agent.concept),
        "revisions": agent.revisions,
        "revision_step": revision_steps[0] if revision_steps else steps,
        "mistakes_before": mistakes_before,
        "mistakes_after": mistakes_after,
        "events": agent.event_log,
    }


def evaluate(trials: int) -> None:
    results = []
    for seed in range(trials):
        results.append(simulate(seed))
        if (seed + 1) % max(1, trials // 10) == 0:
            print(f"progress {seed + 1}/{trials}", flush=True)
    mean = lambda key: sum(float(result[key]) for result in results) / trials
    revised = [int(result["revision_step"]) for result in results if result["revision_step"] < 160]
    median = sorted(revised)[len(revised) // 2] if revised else 160
    print(f"final concept accuracy: {mean('concept_accuracy'):.1%}")
    print(f"self-revision rate:     {sum(bool(r['revisions']) for r in results)/trials:.1%}")
    print(f"median revision step:   {median} (evidence expands at 70)")
    print(f"example events:         {results[0]['events']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.trials)
