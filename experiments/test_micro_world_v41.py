import unittest

from micro_world_v41 import (PushWorld, choose_experiment, consistent_hypotheses,
                             empty_world_memory, learn_steps)


class MicroWorldTest(unittest.TestCase):
    def test_sensor_does_not_disclose_hidden_rule(self):
        experiment = {"size": 1, "roughness": 2, "wall_contact": 0,
                      "color": 1, "shape": 0, "force": 3}
        observation = PushWorld.observe(experiment)
        self.assertNotIn("hidden_resistance", observation)
        self.assertNotIn("moved", observation)

    def test_agent_selects_an_untried_intervention(self):
        memory = empty_world_memory()
        hypotheses = consistent_hypotheses([])
        experiment = choose_experiment(memory, hypotheses)
        self.assertIsNotNone(experiment)

    def test_active_experiments_reduce_hypotheses_and_pass_unseen_worlds(self):
        memory = empty_world_memory()
        before = len(consistent_hypotheses([]))
        summary = learn_steps(memory, 40)
        self.assertLess(summary["surviving_hypotheses"], before)
        self.assertGreaterEqual(summary["holdout"]["accuracy"], 0.95)
        self.assertGreater(summary["corrective_revisions"], 0)
        self.assertFalse(summary["world_rule_visible_to_learner"])
        self.assertEqual(summary["remote_llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
