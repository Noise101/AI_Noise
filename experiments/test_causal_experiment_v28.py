import unittest

from causal_experiment_v28 import (CausalExperimentEngine, RevisableBelief, classify_trend,
                                  evaluate_causal_views, normalized_action)


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

    def test_abstract_view_cannot_erase_concrete_causal_evidence(self):
        transitions = {}
        for index in range(120):
            transitions[f"animal{index}|fails|task"] = {f"animal{index}|leaves|place": 1}
            transitions[f"bird{index}|sees|food"] = {f"bird{index}|waits|place": 1}
        representation = {
            "selected_scheme": "learned_frequency_bands",
            "learned_action_bands": {
                "fails": "rare", "leaves": "rare", "sees": "rare", "waits": "rare"}}
        concrete = CausalExperimentEngine(transitions).run()
        report = evaluate_causal_views(transitions, representation)
        self.assertEqual(report["selected_view"], "concrete")
        self.assertEqual(report["evaluation"], concrete["evaluation"])
        self.assertEqual(report["supported_hypotheses"], concrete["supported_hypotheses"])
        self.assertIn("abstract", report["view_evaluations"])

    def test_learning_curve_accumulates_across_calls_without_duplicating_a_point(self):
        transitions = {}
        for index in range(40):
            transitions[f"animal{index}|fails|reach_food"] = {
                f"animal{index}|leaves|food": 1}
            transitions[f"bird{index}|sees|food"] = {
                f"bird{index}|waits|nearby": 1}
        representation = {"selected_scheme": "surface"}
        first = evaluate_causal_views(transitions, representation)
        self.assertIn("learning_curve", first)
        self.assertEqual(len(first["learning_curve"]), 1)
        self.assertEqual(first["learning_curve"][0]["supported_hypotheses"],
                         first["supported_hypotheses"])
        second = evaluate_causal_views(transitions, representation, first)
        # Re-running on unchanged transitions must not duplicate the point.
        self.assertEqual(second["learning_curve"], first["learning_curve"])

        stuck_curve = [{"training_examples": index, "lift": 0} for index in range(10)]
        self.assertEqual(classify_trend(stuck_curve, "lift", window=10, min_delta=1), "flat")

    def test_shared_context_generates_questions_not_causal_answers(self):
        transitions = {}
        for index in range(80):
            action = "pushes" if index % 2 else "waits"
            outcome = "falls" if action == "pushes" else "stays"
            transitions[f"agent{index}|{action}|stone"] = {f"agent{index}|{outcome}|stone": 1}
        report = CausalExperimentEngine(transitions).run()
        self.assertTrue(report["matched_contrasts"])
        self.assertTrue(all(item["status"] == "needs_comparative_evidence"
                            for item in report["counterfactual_questions"]))


if __name__ == "__main__":
    unittest.main()
