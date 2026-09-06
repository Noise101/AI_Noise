import unittest

from active_curriculum_v1 import (active_learning_targets, deprioritise_syntactic_curiosity,
                                  is_closed_class_phrase)


class ActiveCurriculumTest(unittest.TestCase):
    def test_closed_class_phrase_detection(self):
        self.assertTrue(is_closed_class_phrase("in the"))
        self.assertTrue(is_closed_class_phrase("to the"))
        self.assertFalse(is_closed_class_phrase("the fox"))
        self.assertFalse(is_closed_class_phrase("hungry wolf"))

    def test_syntactic_phrase_gaps_are_retired_not_pursued(self):
        ledger = {
            "phrase:in the": {"layer": "phrase", "query": "in the",
                              "status": "wanting_to_know", "pressure": 245.0},
            "phrase:hungry fox": {"layer": "phrase", "query": "hungry fox",
                                  "status": "wanting_to_know", "pressure": 3.0},
            "word:vineyard": {"layer": "word", "query": "vineyard",
                              "status": "wanting_to_know", "pressure": 5.0},
        }
        report = deprioritise_syntactic_curiosity(ledger)
        self.assertEqual(ledger["phrase:in the"]["pressure"], 0.0)
        self.assertEqual(ledger["phrase:in the"]["status"],
                         "syntactic_not_a_learnable_concept")
        self.assertEqual(ledger["phrase:hungry fox"]["pressure"], 3.0)
        self.assertEqual(ledger["word:vineyard"]["pressure"], 5.0)
        self.assertEqual(report["retired_syntactic_gaps"], 1)

    def test_ledger_is_capped_keeping_the_highest_pressure_entries(self):
        ledger = {f"g{i}": {"layer": "word", "query": f"w{i}",
                            "status": "wanting_to_know", "pressure": float(i),
                            "last_seen_cycle": i}
                  for i in range(50)}
        report = deprioritise_syntactic_curiosity(ledger, max_entries=10)
        self.assertEqual(len(ledger), 10)
        self.assertIn("g49", ledger)
        self.assertNotIn("g0", ledger)
        self.assertEqual(report["evicted_low_pressure_gaps"], 40)

    def test_targets_come_from_recurring_model_misses(self):
        event_structure = {"counterexamples": [
            {"subject": "fox", "object": "grapes", "predicted": "said", "observed": "ate"},
            {"subject": "fox", "object": "grapes", "predicted": "said", "observed": "ate"},
            {"subject": "fox", "object": "grapes", "predicted": "said", "observed": "ate"},
            {"subject": "bird", "object": "nest", "predicted": "flew", "observed": "built"},
        ]}
        targets = active_learning_targets(event_structure, visited=set())
        self.assertTrue(targets)
        top = targets[0]
        self.assertEqual(top["reason"],
                         "seek more events with an argument frame the model predicts wrong")
        self.assertIn("ate", top["seed"])
        self.assertIn("fox", top["seed"])
        self.assertEqual(top["confusion"]["count"], 3)

    def test_a_single_miss_is_not_a_target(self):
        event_structure = {"counterexamples": [
            {"subject": "fox", "object": "grapes", "predicted": "said", "observed": "ate"}]}
        self.assertEqual(active_learning_targets(event_structure, set()), [])

    def test_no_counterexamples_yields_no_targets(self):
        self.assertEqual(active_learning_targets({}, set()), [])
        self.assertEqual(active_learning_targets({"counterexamples": []}, set()), [])

    def test_already_visited_seeds_are_skipped(self):
        event_structure = {"counterexamples": [
            {"subject": "fox", "object": "grapes", "predicted": "said", "observed": "ate"}] * 3}
        targets = active_learning_targets(event_structure, visited={"ate fox grapes"})
        self.assertEqual(targets, [])


if __name__ == "__main__":
    unittest.main()
