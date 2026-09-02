import unittest

from verified_experience_v47 import rebuild_verified_experience


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

    def test_unadmitted_curriculum_never_enters_verified_experience(self):
        memory = {"records": {"x": {"seed": "bad", "source_url": "source",
                                      "source_position": 0, "sentence": "The fox ran home.",
                                      "curriculum_admitted": False}}}
        report = rebuild_verified_experience(memory)
        self.assertEqual(report["summary"]["accepted_sentences"], 0)


if __name__ == "__main__":
    unittest.main()
