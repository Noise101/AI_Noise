import unittest

from representation_learning_v31 import (abstract_event, evaluate_representations,
                                         learn_form_families, transform_transitions)


class RepresentationLearningTest(unittest.TestCase):
    def test_form_families_come_from_observed_spelling_not_dictionary(self):
        families = learn_form_families({"jump", "jumped", "wait"})
        self.assertEqual(families["jump"], families["jumped"])
        self.assertNotEqual(families["jump"], families["wait"])

    def test_role_abstraction_must_beat_surface_on_unseen_entities(self):
        transitions = {}
        for index in range(120):
            transitions[f"animal{index}|sees|food{index}"] = {
                f"animal{index}|waits|food{index}": 1}
            transitions[f"animal{index}|fails|task{index}"] = {
                f"animal{index}|leaves|task{index}": 1}
        report = evaluate_representations(transitions)
        self.assertNotEqual(report["selected_scheme"], "surface")
        chosen = next(item for item in report["evaluations"]
                      if item["scheme"] == report["selected_scheme"])
        surface = next(item for item in report["evaluations"] if item["scheme"] == "surface")
        self.assertGreater(chosen["accuracy"], surface["accuracy"])

    def test_selected_representation_can_feed_causal_evaluation(self):
        transitions = {"fox|sees|food": {"fox|waits|food": 2}}
        report = {"selected_scheme": "role_action", "learned_form_families": {}}
        self.assertEqual(transform_transitions(transitions, report),
                         {"agent|sees|object": {"agent|waits|object": 2}})


if __name__ == "__main__":
    unittest.main()
