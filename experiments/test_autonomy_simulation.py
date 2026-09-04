"""Multi-thousand-round simulation of the autonomy/curriculum state machine.

Unit tests exercise update_autonomy_state and update_collection_progress with a
handful of hand-picked snapshots. That style of test missed the deadlock fixed
in this same session: two changes that were each correct on their own (only
world_model_lift counts as progress; that value is structurally frozen at 0
before the fixed benchmark unlocks) combined into a plateau the worker could
never climb out of on its own -- and no test ran either function across enough
consecutive rounds to observe it.

This module drives the same pure functions across thousands of synthetic
rounds with a controllable "world" standing in for the real, expensive
pipeline (network reads, parsing, the actual world-model training), so
long-horizon regressions like that one can be caught before a live worker
finds them.
"""

from __future__ import annotations

import random
import unittest

from local_worker_v21 import (COLLECTION_STALL_ROUNDS, update_autonomy_state,
                              update_collection_progress)


class SimulatedWorld:
    """Stands in for the measurable signals a real round would report.

    Independent eligible narrative collections accumulate at a fixed per-round
    admit probability -- roughly the ~17% observed live in strategy_performance
    -- until the v51 benchmark's 30-collection quota locks it. Before that,
    world_model_lift is pinned at 0, mirroring train_and_evaluate's
    benchmark_not_ready early return. After lock, a genuine predictive gain
    only becomes available once curricula reach true_lift_ready_at, standing in
    for the real work still needed post-unlock to find and confirm a model.
    """

    def __init__(self, seed: int, admit_rate: float = 0.17, curricula_per_round: int = 2,
                gain_delay_curricula: int = 1600):
        self.rng = random.Random(seed)
        self.curricula = 0
        self.admit_rate = admit_rate
        self.curricula_per_round = curricula_per_round
        self.eligible_collections = 0
        self.locked = False
        self.true_lift_ready_at: int | None = None
        self.gain_delay_curricula = gain_delay_curricula

    def step(self) -> dict:
        self.curricula += self.curricula_per_round
        if not self.locked and self.rng.random() < self.admit_rate:
            self.eligible_collections += 1
        if self.eligible_collections >= 30 and not self.locked:
            self.locked = True
            self.true_lift_ready_at = self.curricula + self.gain_delay_curricula
        lift = 3 if self.locked and self.curricula >= self.true_lift_ready_at else 0
        return {"global_memory": {"curricula": self.curricula},
                "world_model": {"selected_evaluation": {"lift": lift},
                                "benchmark": {"locked": self.locked,
                                              "eligible_collection_count":
                                                  self.eligible_collections}},
                "experience_revision": {"evaluation": {"correct": 0, "total": 0}}}


class AutonomySimulationTest(unittest.TestCase):
    def test_thousands_of_rounds_never_deadlock_before_the_benchmark_can_measure_progress(self):
        curriculum: dict = {}
        world = SimulatedWorld(seed=7)
        entered_plateau = recovered_from_plateau = False
        previous_mode = None
        for round_number in range(3000):
            report = world.step()
            state = update_autonomy_state(curriculum, report)
            if not report["world_model"]["benchmark"]["locked"]:
                self.assertNotEqual(state["mode"], "capability_plateau",
                    f"round {round_number}: plateaued while the benchmark was still "
                    "unmeasurable (world_model_lift is structurally frozen at 0 there)")
                self.assertFalse(state["human_intervention_required"],
                    f"round {round_number}: asked for human intervention before the "
                    "benchmark could measure any progress at all")
            if state["mode"] == "capability_plateau":
                entered_plateau = True
            elif state["mode"] == "normal_curriculum" and previous_mode == "capability_plateau":
                recovered_from_plateau = True
            previous_mode = state["mode"]
        # The simulation must actually exercise the interesting states, not just
        # coast through normal_curriculum the whole run.
        self.assertTrue(entered_plateau, "simulation never reached capability_plateau")
        self.assertTrue(recovered_from_plateau,
                        "capability_plateau was never exited even after a genuine "
                        "post-unlock capability gain became available: a plateau "
                        "before the benchmark unlocks must not be a permanent trap")

    def test_collection_stall_is_flagged_within_a_bounded_number_of_cycles(self):
        curriculum: dict = {}
        world_model = {"benchmark": {"locked": False, "eligible_collection_count": 12}}
        stalled_at = None
        for cycle in range(COLLECTION_STALL_ROUNDS * 3):
            if update_collection_progress(curriculum, world_model):
                stalled_at = cycle
                break
        self.assertIsNotNone(stalled_at, "stagnant collection growth was never flagged")
        self.assertLessEqual(stalled_at, COLLECTION_STALL_ROUNDS)

    def test_state_machine_never_raises_or_grants_human_intervention_pre_benchmark(self):
        """Fuzz update_autonomy_state / update_collection_progress with adversarial,
        internally-inconsistent snapshots -- the kind hand-written unit tests don't
        think to try -- and check the state machine stays well-behaved throughout."""
        rng = random.Random(99)
        curriculum: dict = {}
        for _ in range(3000):
            locked = rng.random() < 0.3
            report = {
                "global_memory": {"curricula": rng.randint(0, 5000)},
                "world_model": {"selected_evaluation": {"lift": rng.choice([0, 0, 0, 1, -1, 5])},
                                "reusable_rules": [{}] * rng.randint(0, 5),
                                "benchmark": {"locked": locked,
                                              "eligible_collection_count": rng.randint(0, 40)}},
                "association": {"selected_evaluation": {"correct": rng.randint(0, 10),
                                                         "baseline_correct": rng.randint(0, 10)}},
                "causal_evaluation": {"evaluation": {"correct": rng.randint(0, 10),
                                                     "baseline_correct": rng.randint(0, 10)}},
                "representation": {"selected_evaluation": {"correct": rng.randint(0, 10)}},
                "experience_revision": {"evaluation": {"correct": rng.randint(0, 10),
                                                       "total": rng.randint(0, 20)},
                                        "reusable_rules": rng.randint(0, 5),
                                        "failure_patterns": [{}] * rng.randint(0, 3)},
            }
            state = update_autonomy_state(curriculum, report)
            update_collection_progress(curriculum, report["world_model"])
            if not locked:
                self.assertFalse(state["human_intervention_required"])


if __name__ == "__main__":
    unittest.main()
