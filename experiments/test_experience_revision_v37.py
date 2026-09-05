import hashlib
import unittest

from experience_revision_v37 import ExperienceRevisionEngine, holdout, mismatch_kind, structural


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

    def test_holdout_keeps_a_whole_source_together_regardless_of_the_pair(self):
        self.assertEqual(holdout("fox|saw|grapes", "fox|found|grapes", ["http://a"]),
                         holdout("wolf|ate|meat", "wolf|left|home", ["http://a"]))

    def test_holdout_falls_back_to_the_pair_hash_without_source_attribution(self):
        expected = hashlib.sha256(
            b"revision:fox|saw|grapes->fox|found|grapes").digest()[0] % 5 == 0
        self.assertEqual(holdout("fox|saw|grapes", "fox|found|grapes"), expected)

    def test_engine_splits_by_source_when_transition_sources_is_given(self):
        transitions = {}
        for index in range(80):
            prior = f"animal{index}|sees|red_food"
            outcome = (f"animal{index}|takes|red_food" if index % 7 else
                       f"animal{index}|leaves|red_food")
            transitions[prior] = {outcome: 1}
        transition_sources = {prior: {outcome: [f"http://story-{index}"]}
                              for index, (prior, outcomes) in enumerate(transitions.items())
                              for outcome in outcomes}
        report = ExperienceRevisionEngine(transitions, None, transition_sources).run()
        self.assertGreater(report["summary"]["prediction_trials"], 0)

    def test_contextual_evaluation_splits_by_source_when_given(self):
        contextual = {}
        for index in range(80):
            context = f"animal{index}|sees|food>>animal{index}|tries|food"
            contextual[context] = {f"animal{index}|takes|food": 1}
        contextual_transition_sources = {
            context: {outcome: [f"http://story-{index}"]}
            for index, (context, outcomes) in enumerate(contextual.items())
            for outcome in outcomes}
        report = ExperienceRevisionEngine(
            {}, contextual, None, contextual_transition_sources).run()
        evaluation = report["summary"]["contextual_evaluation"]
        self.assertGreater(evaluation["total"], 0)


if __name__ == "__main__":
    unittest.main()
