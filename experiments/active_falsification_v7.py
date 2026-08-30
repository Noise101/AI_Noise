#!/usr/bin/env python3
"""Actively choose observations that can falsify the current concept."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    sensors: tuple[int, ...]


@dataclass(frozen=True)
class TruthHypothesis:
    inputs: tuple[int, ...]
    table: tuple[int, ...]

    def predict(self, query: Query) -> int:
        address = 0
        for index in self.inputs:
            address = (address << 1) | query.sensors[index]
        return self.table[address]

    @property
    def complexity(self) -> int:
        return len(self.inputs) + sum(self.table[i] != self.table[i - 1] for i in range(1, len(self.table)))

    def describe(self) -> str:
        return f"C({','.join(f's{i}' for i in self.inputs)})={''.join(map(str, self.table))}"


class VersionSpaceAgent:
    """Keeps every small truth table not yet falsified by experience."""

    def __init__(self, sensor_count: int = 5):
        self.sensor_count = sensor_count
        self.hypotheses = self._all_hypotheses()
        self.evidence: list[tuple[Query, int]] = []
        self.falsifications = 0

    def _all_hypotheses(self) -> list[TruthHypothesis]:
        hypotheses = []
        for width in (1, 2):
            for inputs in itertools.combinations(range(self.sensor_count), width):
                for table in itertools.product((0, 1), repeat=2**width):
                    if len(set(table)) == 1:
                        continue
                    hypotheses.append(TruthHypothesis(inputs, table))
        return hypotheses

    def observe(self, query: Query, outcome: int) -> None:
        previous_belief = self.belief()
        self.evidence.append((query, outcome))
        self.hypotheses = [hypothesis for hypothesis in self.hypotheses if hypothesis.predict(query) == outcome]
        if previous_belief is not None and previous_belief.predict(query) != outcome:
            self.falsifications += 1

    def belief(self) -> TruthHypothesis | None:
        if not self.hypotheses:
            return None
        return min(self.hypotheses, key=lambda hypothesis: (hypothesis.complexity, len(hypothesis.inputs), hypothesis.describe()))

    @staticmethod
    def entropy(probability: float) -> float:
        if probability <= 0 or probability >= 1:
            return 0.0
        return -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)

    def choose_experiment(self, candidates: list[Query], rng: random.Random, active: bool) -> Query:
        unseen = [candidate for candidate in candidates if candidate not in {query for query, _ in self.evidence}]
        if not unseen:
            unseen = candidates
        if not active:
            return rng.choice(unseen)
        scored = []
        for candidate in unseen:
            p_one = sum(hypothesis.predict(candidate) for hypothesis in self.hypotheses) / len(self.hypotheses)
            information = self.entropy(p_one)
            # Prefer tests that can specifically refute the currently simplest explanation.
            belief = self.belief()
            dissent = sum(h.predict(candidate) != belief.predict(candidate) for h in self.hypotheses) / len(self.hypotheses)
            scored.append((information + 0.15 * dissent, rng.random(), candidate))
        return max(scored)[2]

    def predictive_accuracy(self, target, candidates: list[Query]) -> float:
        belief = self.belief()
        if belief is None:
            return 0.0
        return sum(belief.predict(query) == target(query) for query in candidates) / len(candidates)


class XorWorld:
    @staticmethod
    def outcome(query: Query) -> int:
        return query.sensors[0] ^ query.sensors[3]


def run_trial(seed: int, active: bool, max_queries: int = 32) -> dict[str, object]:
    rng = random.Random(seed)
    world = XorWorld()
    agent = VersionSpaceAgent()
    candidates = [Query(bits) for bits in itertools.product((0, 1), repeat=5)]

    # Biased prior experience hides s3 variation, making C=s0 the simplest belief.
    initial_pool = [query for query in candidates if query.sensors[3] == 0]
    for query in rng.sample(initial_pool, 6):
        agent.observe(query, world.outcome(query))
    initial_belief = agent.belief().describe()

    solved_at = max_queries
    trace = []
    for query_index in range(1, max_queries + 1):
        query = agent.choose_experiment(candidates, rng, active)
        prediction = agent.belief().predict(query)
        outcome = world.outcome(query)
        agent.observe(query, outcome)
        accuracy = agent.predictive_accuracy(world.outcome, candidates)
        trace.append((query_index, query.sensors, prediction, outcome, agent.belief().describe(), accuracy))
        if accuracy == 1.0:
            solved_at = query_index
            break
    return {
        "initial_belief": initial_belief,
        "final_belief": agent.belief().describe(),
        "solved_at": solved_at,
        "falsifications": agent.falsifications,
        "trace": trace,
    }


def evaluate(trials: int) -> None:
    active_results = []
    passive_results = []
    for seed in range(trials):
        active_results.append(run_trial(seed, True))
        passive_results.append(run_trial(seed, False))
        if (seed + 1) % max(1, trials // 10) == 0:
            print(f"progress {seed + 1}/{trials}", flush=True)
    active_steps = [int(result["solved_at"]) for result in active_results]
    passive_steps = [int(result["solved_at"]) for result in passive_results]
    print(f"active median experiments:  {statistics.median(active_steps):.1f}")
    print(f"passive median experiments: {statistics.median(passive_steps):.1f}")
    print(f"active solved <=8:          {sum(x <= 8 for x in active_steps)/trials:.1%}")
    print(f"passive solved <=8:         {sum(x <= 8 for x in passive_steps)/trials:.1%}")
    print(f"example initial: {active_results[0]['initial_belief']}")
    print(f"example final:   {active_results[0]['final_belief']}")
    print(f"example trace:   {active_results[0]['trace']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.trials)
