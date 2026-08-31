import unittest

from causal_experiment_v28 import CausalExperimentEngine, RevisableBelief, normalized_action


class CausalExperimentTest(unittest.TestCase):
    def test_auxiliary_event_gets_observable_predicate(self):
        self.assertEqual(normalized_action("he|was|running_to_safety"), "running")

    def test_counterexamples_can_reject_an_old_supported_belief(self):
        belief = RevisableBelief("failure", "give_up")
        belief.update(True, 4)
        self.assertEqual(belief.status, "supported")
        belief.update(False, 10)
        self.assertEqual(belief.status, "rejected")
        self.assertEqual(belief.revisions[-1]["before"], "supported")

    def test_holdout_predictions_are_recorded_with_observed_result(self):
        transitions = {}
        for index in range(40):
            transitions[f"animal{index}|fails|reach_food"] = {
                f"animal{index}|leaves|food": 1}
            transitions[f"bird{index}|sees|food"] = {
                f"bird{index}|waits|nearby": 1}
        report = CausalExperimentEngine(transitions).run()
        self.assertGreater(report["test_observations"], 0)
        self.assertTrue(all("prediction" in item and "observed_after_registration" in item
                            for item in report["preregistered_predictions"]))
        self.assertGreater(report["supported_hypotheses"], 0)


if __name__ == "__main__":
    unittest.main()
