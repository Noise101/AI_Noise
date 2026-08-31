import unittest

from mastery_drive_v24 import assess_language_mastery


class MasteryDriveTest(unittest.TestCase):
    def test_finds_a_language_ability_gap_instead_of_claiming_completion(self):
        report = {"knowledge": {"lexicon": {
            "characters": {"a": 3, "b": 1}, "word_forms": {"fox": 2, "unknown": 2},
            "grounded_meanings": [{"form": "fox"}], "researched_meanings": {},
            "phrase_candidates": [{"phrase": "at last"}], "researched_phrase_meanings": {},
            "conversation_cues": {"said": 2}, "researched_conversation_acts": {},
        }, "story": {"predictions_checked": 0, "mistakes_detected": 0, "why_questions": []},
            "concepts": {"beliefs": []}}}
        assessment = assess_language_mastery(report)
        self.assertEqual(assessment["status"], "learning_incomplete")
        self.assertIn(assessment["weakest_dimension"], assessment["dimensions"])
        self.assertLess(assessment["overall_score"], assessment["target"])

    def test_prediction_is_not_mastered_without_being_tested(self):
        assessment = assess_language_mastery({"knowledge": {}})
        self.assertEqual(assessment["dimensions"]["prediction"]["score"], 0)
        self.assertNotEqual(assessment["status"], "current_curriculum_mastered")

    def test_legacy_story_counts_do_not_inflate_audited_causality(self):
        report = {"knowledge": {"story": {"predictions_checked": 999,
                                             "mistakes_detected": 0,
                                             "why_questions": [{"status": "candidate_found"}]}}}
        assessment = assess_language_mastery(report, {"supported_hypotheses": 0,
            "evaluation": {"total": 0, "correct": 0, "baseline_correct": 0}})
        self.assertEqual(assessment["dimensions"]["prediction"]["score"], 0)
        self.assertEqual(assessment["dimensions"]["causality"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
