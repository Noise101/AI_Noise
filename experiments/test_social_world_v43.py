import unittest

from social_world_v43 import SocialWorld, empty_social_memory, learn_steps


class SocialWorldTest(unittest.TestCase):
    def test_partner_searches_according_to_unseen_transfer_belief(self):
        scenario = {"initial_location": "left", "current_location": "right",
                    "partner_witnessed_transfer": False, "message": None}
        observation = SocialWorld.observe(scenario)
        result = SocialWorld.partner_action(scenario)
        self.assertEqual(result["searched_location"], "left")
        self.assertNotIn("private_belief", observation)
        self.assertFalse(result["private_belief_disclosed"])

    def test_message_can_change_partner_action_without_exposing_private_state(self):
        scenario = {"initial_location": "left", "current_location": "left",
                    "partner_witnessed_transfer": False, "message": "right"}
        self.assertEqual(SocialWorld.partner_action(scenario)["searched_location"], "right")

    def test_noise_identifies_other_observation_model_on_unseen_scenarios(self):
        memory = empty_social_memory()
        summary = learn_steps(memory, 30)
        self.assertEqual(summary["status"], "stage_3_foundation_mastered")
        self.assertEqual(summary["unseen_social_tasks"]["accuracy"], 1.0)
        self.assertEqual(summary["surviving_other_models"], 1)
        self.assertGreater(summary["model_revisions"], 0)
        self.assertFalse(summary["partner_private_belief_visible"])
        self.assertEqual(summary["remote_llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
