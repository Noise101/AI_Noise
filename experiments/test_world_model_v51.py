import unittest

from world_model_v51 import (build_sequences, narrative_sequence, narrative_source, parse_frame,
                             passes_gain_gate, train_and_evaluate)


def audit_for_sources(count=30):
    records = {}
    for source in range(count):
        url = f"https://story/{source}"
        sentences = ["The fox was hungry.", "The fox wanted to eat.",
                     "The fox tried to reach food.", "The fox ate food."]
        for position, sentence in enumerate(sentences):
            records[f"{source}:{position}"] = {"source_url": url, "seed": f"story {source}",
                "source_position": position, "sentence": sentence, "curriculum_admitted": True}
    return {"records": records}


class WorldModelV51Test(unittest.TestCase):
    def test_history_is_not_mistaken_for_story(self):
        self.assertFalse(narrative_source("https://example/History_of_Oregon"))
        self.assertTrue(narrative_source("https://example/Animal_Stories/Fox"))

    def test_benchmark_requires_recurrent_actor_and_action_change(self):
        self.assertFalse(narrative_sequence({"frames": [
            {"actor": "a", "action": "said"}, {"actor": "b", "action": "said"},
            {"actor": "c", "action": "said"}, {"actor": "d", "action": "said"}]}))
        self.assertTrue(narrative_sequence({"frames": [
            {"actor": "fox", "action": "saw"}, {"actor": "fox", "action": "wanted"},
            {"actor": "fox", "action": "tried"}, {"actor": "fox", "action": "ate"}]}))

    def test_frame_keeps_state_goal_action_and_source_sentence(self):
        frame = parse_frame("The fox was hungry and wanted to eat food.")
        self.assertIsNotNone(frame)
        self.assertEqual(frame["actor"], "fox")
        self.assertIn("hungry", frame["states"])
        self.assertIn("eat", frame["goals"])
        self.assertEqual(frame["sentence"], "The fox was hungry and wanted to eat food.")

    def test_benchmark_is_locked_and_never_used_for_training(self):
        first = train_and_evaluate(audit_for_sources())
        expanded = audit_for_sources(35)
        second = train_and_evaluate(expanded, first)
        self.assertEqual(first["benchmark"]["source_urls"], second["benchmark"]["source_urls"])
        self.assertTrue(set(second["benchmark"]["source_urls"]).isdisjoint(
            second["training"]["source_urls"]))
        self.assertEqual(second["benchmark"]["fingerprint"], first["benchmark"]["fingerprint"])

    def test_no_candidate_is_adopted_without_fixed_benchmark_gain(self):
        result = train_and_evaluate(audit_for_sources())
        if result["selected_evaluation"]["lift"] <= 0:
            self.assertEqual(result["selected_mode"], "frequency_baseline")
        self.assertIn("benchmark_sources_never_train", result["invariants"])

    def test_small_or_unpaired_gain_cannot_clear_gate(self):
        self.assertFalse(passes_gain_gate({"total": 30, "lift": 3, "coverage": 1.0,
                                          "one_sided_sign_p": .2}))
        self.assertFalse(passes_gain_gate({"total": 10, "lift": 10, "coverage": 1.0,
                                          "one_sided_sign_p": .01}))
        self.assertTrue(passes_gain_gate({"total": 30, "lift": 4, "coverage": .5,
                                         "one_sided_sign_p": .05}))

    def test_state_persists_into_later_actor_experience(self):
        sequences = build_sequences(audit_for_sources(1))
        frames = next(iter(sequences.values()))["frames"]
        eating = next(frame for frame in frames if frame["action"] == "ate")
        self.assertIn("hungry", eating["known_states"])

    def test_frozen_benchmark_examples_do_not_change_when_source_grows(self):
        audit = audit_for_sources()
        first = train_and_evaluate(audit)
        benchmark_url = first["benchmark"]["source_urls"][0]
        audit["records"]["late:0"] = {"source_url": benchmark_url, "seed": "late",
            "source_position": 99, "sentence": "The fox ate berries.",
            "curriculum_admitted": True}
        second = train_and_evaluate(audit, first)
        self.assertEqual(first["benchmark_examples"], second["benchmark_examples"])
        self.assertEqual(first["benchmark"]["examples"], second["benchmark"]["examples"])


if __name__ == "__main__":
    unittest.main()
