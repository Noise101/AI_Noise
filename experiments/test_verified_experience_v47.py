import unittest

from verified_experience_v47 import rebuild_verified_experience, select_experience_profile


class VerifiedExperienceTest(unittest.TestCase):
    def test_rejects_complex_or_non_action_text_and_keeps_short_sequences(self):
        sentences = ["The fox saw grapes.", "He tried to reach them.",
                     "During the exceedingly elaborate ceremony; many names were listed."]
        records = {str(i): {"seed": "fox grapes", "source_url": "source", "source_position": i,
                            "sentence": sentence, "curriculum_admitted": True}
                   for i, sentence in enumerate(sentences)}
        report = rebuild_verified_experience({"records": records})
        self.assertGreaterEqual(report["summary"]["accepted_sentences"], 2)
        self.assertEqual(report["summary"]["transition_observations"], 1)
        self.assertGreater(report["summary"]["quarantined_sentences"], 0)

    def test_coordinate_clause_sentence_is_split_instead_of_quarantined_whole(self):
        # A single "X and Y" sentence too long to pass as one simple clause is not
        # one ambiguous compound event; each clause should stand on its own.
        sentence = ("The fox saw many ripe purple grapes hanging from the tall garden "
                    "vine and the fox jumped as high as he could to reach them.")
        records = {"0": {"seed": "fox grapes", "source_url": "source", "source_position": 0,
                         "sentence": sentence, "curriculum_admitted": True}}
        report = rebuild_verified_experience({"records": records},
                                             policy="developmental_grounded_18")
        self.assertEqual(report["summary"]["accepted_sentences"], 2)
        self.assertEqual(report["summary"]["quarantined_sentences"], 0)
        self.assertIn("fox|saw|many", report["event_counts"])
        self.assertIn("fox|jumped|high", report["event_counts"])
        self.assertIn("fox|saw|many", report["coherent_transitions"])

    def test_transition_sources_attributes_each_edge_to_its_source_url(self):
        records = {"0": {"seed": "fox grapes", "source_url": "http://a", "source_position": 0,
                         "sentence": "The fox saw grapes.", "curriculum_admitted": True},
                   "1": {"seed": "fox grapes", "source_url": "http://a", "source_position": 1,
                         "sentence": "The fox found grapes.", "curriculum_admitted": True}}
        report = rebuild_verified_experience({"records": records})
        self.assertIn("fox|saw|grapes", report["transition_sources"])
        self.assertEqual(report["transition_sources"]["fox|saw|grapes"]["fox|found|grapes"],
                         ["http://a"])

    def test_transition_sources_lists_every_distinct_source_once_without_duplicates(self):
        # Two independent sources producing the identical (prior, outcome) pair
        # must both be recorded, deduplicated, not counted per occurrence.
        records = {}
        for index, source in enumerate(("http://a", "http://b")):
            records[f"{index}-0"] = {"seed": f"seed{index}", "source_url": source,
                "source_position": 0, "sentence": "The fox saw grapes.",
                "curriculum_admitted": True}
            records[f"{index}-1"] = {"seed": f"seed{index}", "source_url": source,
                "source_position": 1, "sentence": "The fox found grapes.",
                "curriculum_admitted": True}
        report = rebuild_verified_experience({"records": records})
        self.assertEqual(report["transition_sources"]["fox|saw|grapes"]["fox|found|grapes"],
                         ["http://a", "http://b"])
        # Non-destructive: the existing counted transitions must be unaffected.
        self.assertEqual(report["transitions"]["fox|saw|grapes"]["fox|found|grapes"], 2)

    def test_unadmitted_curriculum_never_enters_verified_experience(self):
        memory = {"records": {"x": {"seed": "bad", "source_url": "source",
                                      "source_position": 0, "sentence": "The fox ran home.",
                                      "curriculum_admitted": False}}}
        report = rebuild_verified_experience(memory)
        self.assertEqual(report["summary"]["accepted_sentences"], 0)

    def test_sentence_limit_is_selected_on_unseen_whole_sources(self):
        records = {}
        for source in range(100):
            for position, sentence in enumerate(
                    ("The fox saw food.", "The fox tried food.", "The fox took food.")):
                records[f"{source}-{position}"] = {
                    "seed": f"story {source}", "source_url": f"source-{source}",
                    "source_position": position, "sentence": sentence,
                    "curriculum_admitted": True}
        experience, policy = select_experience_profile({"records": records})
        self.assertEqual(policy["selected_policy"], "developmental_grounded_18")
        self.assertEqual(policy["selection_status"], "selected_on_unseen_sources")
        self.assertTrue(policy["safety_invariants"])
        self.assertEqual(experience["policy"], policy["selected_policy"])


if __name__ == "__main__":
    unittest.main()
