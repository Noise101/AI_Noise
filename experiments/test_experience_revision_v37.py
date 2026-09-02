import unittest

from experience_revision_v37 import ExperienceRevisionEngine, mismatch_kind, structural


class ExperienceRevisionTest(unittest.TestCase):
    def test_removes_entity_but_retains_action_and_object_role(self):
        self.assertEqual(structural("fox|sees|red_food"), "agent|sees|food")

    def test_diagnoses_which_part_of_prediction_failed(self):
        self.assertEqual(mismatch_kind("agent|takes|food", "agent|drops|food"),
                         "action_mismatch")
        self.assertEqual(mismatch_kind("agent|takes|food", "agent|takes|stone"),
                         "object_mismatch")

    def test_forms_tests_and_revises_rules_without_causal_credit(self):
        transitions = {}
        for index in range(80):
            prior = f"animal{index}|sees|red_food"
            outcome = (f"animal{index}|takes|red_food" if index % 7 else
                       f"animal{index}|leaves|red_food")
            transitions[prior] = {outcome: 1}
        report = ExperienceRevisionEngine(transitions).run()
        self.assertGreater(report["summary"]["rules_formed"], 0)
        self.assertGreater(report["summary"]["prediction_trials"], 0)
        self.assertTrue(report["trials"])
        self.assertFalse(report["causal_credit"])
        self.assertIn("failure_causes", report["summary"])
        self.assertTrue(report["summary"]["failure_patterns"])

    def test_two_event_context_is_measured_separately(self):
        contextual = {}
        for index in range(80):
            context = f"animal{index}|sees|food>>animal{index}|tries|food"
            contextual[context] = {f"animal{index}|takes|food": 1}
        report = ExperienceRevisionEngine({}, contextual).run()
        evaluation = report["summary"]["contextual_evaluation"]
        self.assertGreater(evaluation["total"], 0)
        self.assertEqual(evaluation["context_events"], 2)


if __name__ == "__main__":
    unittest.main()
