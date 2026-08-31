import unittest

from local_conversation_v25 import make_noise_utterance, practice_once


class FakePartner:
    model = "local-test"
    def reply(self, utterance):
        return {"reply": "A vine is a plant stem.", "question": "Where did you see it?"}


class LocalConversationTest(unittest.TestCase):
    def test_noise_builds_its_utterance_from_own_gap(self):
        text = make_noise_utterance("fox grapes", {"next_mastery_goal": {"dimension": "words"}},
                                    {"word:vine": {"pressure": 5, "status": "wanting_to_know"}})
        self.assertIn("vine", text)
        self.assertIn("words", text)

    def test_local_reply_is_practice_and_never_evidence(self):
        turn = practice_once("fox grapes", {}, {}, FakePartner())
        self.assertEqual(turn["status"], "practiced")
        self.assertEqual(turn["evidence_score"], 0.0)
        self.assertFalse(turn["verified"])
        self.assertIn("vine", turn["observed_forms"])
        self.assertTrue(turn["practice_metrics"]["formed_followup"])
        self.assertFalse(turn["practice_metrics"]["independent_evidence_added"])


if __name__ == "__main__":
    unittest.main()
