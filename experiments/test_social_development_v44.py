import unittest

from social_development_v44 import empty_stage_three_memory, learn_stage_three


class SocialDevelopmentTest(unittest.TestCase):
    def test_all_stage_three_competencies_are_required_and_learned(self):
        memory = empty_stage_three_memory()
        summary = learn_stage_three(memory, 200)
        self.assertEqual(summary["status"], "stage_3_complete")
        self.assertEqual(summary["competencies_passed"], summary["competencies_total"])
        self.assertGreater(summary["prediction_errors"], 0)
        self.assertFalse(summary["private_state_directly_visible"])
        self.assertEqual(summary["remote_llm_calls"], 0)

    def test_people_have_separate_persistent_models(self):
        memory = empty_stage_three_memory()
        learn_stage_three(memory, 200)
        self.assertNotEqual(memory["profiles"]["ava"]["candidates"],
                            memory["profiles"]["ben"]["candidates"])
        self.assertNotEqual(memory["explanation_styles"]["ava"]["candidates"],
                            memory["explanation_styles"]["ben"]["candidates"])

    def test_promised_words_do_not_override_observed_action(self):
        memory = empty_stage_three_memory()
        summary = learn_stage_three(memory, 200)
        evaluation = summary["competencies"]["cooperation_and_promises"]
        self.assertGreater(evaluation["speech_action_mismatches_correct"], 0)


if __name__ == "__main__":
    unittest.main()
