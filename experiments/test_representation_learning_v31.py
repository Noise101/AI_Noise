import unittest

from representation_learning_v31 import (abstract_event, evaluate_representations,
                                         learn_form_families, learn_frequency_bands,
                                         learn_relational_action_classes,
                                         transform_transitions)


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

    def test_frequency_bands_are_learned_without_fixed_corpus_boundaries(self):
        transitions = []
        for index in range(200):
            prior = "agent|common|object" if index % 2 else f"agent|rare{index}|object"
            outcome = "agent|continues|object" if index % 2 else f"agent|varies{index}|object"
            transitions.append((prior, outcome, 1))
        mapping, thresholds = learn_frequency_bands(transitions)
        self.assertLess(thresholds[0], thresholds[1])
        self.assertTrue(set(mapping.values()).issubset({"rare", "mid", "high"}))
        self.assertEqual(mapping["common"], "high")

    def test_relational_classes_group_actions_by_repeated_observed_effect(self):
        train = [("fox|tries|food", "fox|waits|food", 3),
                 ("bird|looks|food", "bird|waits|food", 4),
                 ("cat|jumps|wall", "cat|falls|ground", 1)]
        classes = learn_relational_action_classes(train)
        self.assertEqual(classes["tries"], "leads_to:waits")
        self.assertEqual(classes["looks"], "leads_to:waits")
        self.assertEqual(classes["jumps"], "action:jumps")


if __name__ == "__main__":
    unittest.main()
