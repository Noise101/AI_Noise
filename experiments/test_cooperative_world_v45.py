import unittest

from cooperative_world_v45 import COMPETENCIES, empty_cooperative_memory, learn_cooperation


class CooperativeWorldTest(unittest.TestCase):
    def test_all_stage_four_competencies_pass_unseen_cases(self):
        memory = empty_cooperative_memory()
        summary = learn_cooperation(memory, 200)
        self.assertEqual(summary["status"], "stage_4_complete")
        self.assertEqual(summary["competencies_passed"], len(COMPETENCIES))
        for result in summary["competencies"].values():
            self.assertEqual(result["accuracy"], 1.0)
            self.assertGreater(result["total"], 0)
            self.assertEqual(result["hypotheses_remaining"], 1)

    def test_learning_keeps_prediction_errors_and_self_attribution(self):
        memory = empty_cooperative_memory()
        summary = learn_cooperation(memory, 200)
        self.assertGreater(summary["prediction_errors"], 0)
        attribution = memory["tracks"]["failure_attribution"]
        self.assertTrue(any(x["observed_consequence"] == "noise_model_error"
                            for x in attribution["observations"]))

    def test_local_partner_is_not_an_llm_or_authority(self):
        memory = empty_cooperative_memory()
        summary = learn_cooperation(memory, 200)
        self.assertEqual(summary["remote_llm_calls"], 0)
        self.assertEqual(memory["local_partner_mode"],
                         "deterministic_non_llm_training_partner")
        self.assertTrue(summary["continues_autonomous_learning_after_completion"])


if __name__ == "__main__":
    unittest.main()
