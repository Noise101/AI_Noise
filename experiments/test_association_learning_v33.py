import unittest
from collections import Counter

from association_learning_v33 import AssociationLearner, classify_trend, predict_with_backoff


class AssociationLearningTest(unittest.TestCase):
    def test_builds_typed_associations_from_audited_events(self):
        report = AssociationLearner({}, {"fox|sees|red_grapes": 3}).run()
        kinds = {item["kind"] for item in report["structural_associations"]}
        self.assertEqual(kinds, {"agent_action", "action_object", "scene_cooccurrence"})

    def test_prediction_feedback_reinforces_and_weakens_without_causal_credit(self):
        transitions = {}
        for index in range(80):
            prior = f"fox|waits|tree_{index}"
            outcome = f"fox|eats|fruit_{index}" if index % 5 else f"fox|leaves|fruit_{index}"
            transitions[prior] = {outcome: 1}
        report = AssociationLearner(transitions).run()
        self.assertGreater(report["evaluation"]["total"], 0)
        self.assertGreater(report["evaluation"]["coverage"], 0)
        self.assertTrue(report["predictive_associations"])
        self.assertIn("not causal evidence", report["warning"])

    def test_failed_predictions_reduce_strength(self):
        transitions = {}
        for index in range(100):
            prior = f"bird|flies|place_{index}"
            outcome = f"bird|lands|nest_{index}" if index < 50 else f"bird|sings|nest_{index}"
            transitions[prior] = {outcome: 1}
        report = AssociationLearner(transitions).run()
        tested = [item for item in report["predictive_associations"]
                  if item["prediction_successes"] + item["prediction_failures"]]
        self.assertTrue(tested)
        self.assertTrue(any(item["prediction_failures"] for item in tested))

    def test_learning_curve_accumulates_across_runs_and_flags_a_declining_trend(self):
        transitions = {}
        for index in range(80):
            prior = f"fox|waits|tree_{index}"
            outcome = f"fox|eats|fruit_{index}" if index % 5 else f"fox|leaves|fruit_{index}"
            transitions[prior] = {outcome: 1}
        first = AssociationLearner(transitions).run()
        self.assertIn("learning_curve", first)
        self.assertEqual(len(first["learning_curve"]), 1)
        self.assertGreater(first["learning_curve"][0]["training_examples"], 0)
        second = AssociationLearner(transitions, previous=first).run()
        # Re-running on the same transitions must not duplicate the training-size point.
        self.assertEqual(len(second["learning_curve"]), 1)
        self.assertEqual(second["learning_curve"][0], first["learning_curve"][0])

        # A synthetic curve that visibly regresses must be read as declining, not flat.
        declining_curve = [{"training_examples": index, "lift": value}
                           for index, value in enumerate((8, 7, 6, 5, 2, 1, 0, -1, -2, -3))]
        self.assertEqual(classify_trend(declining_curve, "lift", window=10, min_delta=1),
                         "declining")

    def test_predict_with_backoff_falls_back_to_the_action_cue_when_all_cues_are_thin(self):
        links = {"subject:fox": Counter({"ate": 1}),
                 "action:waits": Counter({"leaves": 1}),
                 "object:tree": Counter({"ate": 1})}
        prediction, level, cues = predict_with_backoff("fox|waits|tree", links, fallback="ate")
        self.assertEqual(level, "action_only_backoff")
        self.assertEqual(prediction, "leaves")
        self.assertEqual(cues, ["action:waits"])

    def test_predict_with_backoff_votes_across_cues_once_they_clear_the_support_floor(self):
        links = {"subject:fox": Counter({"leaves": 5}),
                 "action:waits": Counter({"leaves": 4}),
                 "object:tree": Counter({"leaves": 3})}
        prediction, level, cues = predict_with_backoff("fox|waits|tree", links, fallback="ate")
        self.assertEqual(level, "cue_vote")
        self.assertEqual(prediction, "leaves")
        self.assertEqual(set(cues), {"subject:fox", "action:waits", "object:tree"})

    def test_predict_with_backoff_falls_back_to_the_global_majority_when_nothing_was_seen(self):
        prediction, level, cues = predict_with_backoff("fox|waits|tree", {}, fallback="ate")
        self.assertEqual((prediction, level, cues), ("ate", "global_fallback", []))

    def test_backoff_candidate_is_evaluated_and_reported_alongside_the_others(self):
        transitions = {}
        for index in range(80):
            prior = f"fox|waits|tree_{index}"
            outcome = f"fox|eats|fruit_{index}" if index % 5 else f"fox|leaves|fruit_{index}"
            transitions[prior] = {outcome: 1}
        report = AssociationLearner(transitions).run()
        self.assertIn("backoff_evaluation", report)
        self.assertGreater(report["backoff_evaluation"]["total"], 0)
        self.assertIn(report["selected_mode"],
                      {"exact_action", "learned_structural_class", "learned_structural_bands",
                       "cue_hierarchy_backoff"})

    def test_structural_action_classes_are_compared_with_their_own_baseline(self):
        transitions = {}
        for index in range(200):
            prior_action = "common" if index % 3 else f"rare{index}"
            outcome_action = "continues" if prior_action == "common" else f"varies{index}"
            transitions[f"agent{index}|{prior_action}|item"] = {
                f"agent{index}|{outcome_action}|item": 1}
        report = AssociationLearner(transitions).run()
        structural = report["structural_evaluation"]
        self.assertGreater(structural["total"], 0)
        self.assertIn(report["selected_mode"],
                      {"exact_action", "learned_structural_class", "learned_structural_bands"})


if __name__ == "__main__":
    unittest.main()
