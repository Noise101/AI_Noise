import unittest

from local_conversation_v25 import make_noise_utterance, practice_once, select_dialogue_unknown


class FakePartner:
    model = "local-test"
    def __init__(self):
        self.calls = []
    def reply(self, utterance):
        self.calls.append(utterance)
        return {"reply": "A vine is a plant stem.", "question": "Where did you see it?"}


class LocalConversationTest(unittest.TestCase):
    def test_verified_repetition_loses_priority_to_an_untried_question(self):
        curiosity = {"phrase:of the": {"pressure": 100, "status": "wanting_to_know"},
                     "word:vine": {"pressure": 20, "status": "wanting_to_know"}}
        memory = {"expressions": {"of the": {"attempts": 5}}}
        self.assertEqual(select_dialogue_unknown(curiosity, memory), "vine")

    def test_independently_observed_expression_is_skipped(self):
        curiosity = {"phrase:of the": {"pressure": 10000, "status": "wanting_to_know"},
                     "word:vine": {"pressure": 1, "status": "wanting_to_know"}}
        memory = {"expressions": {"of the": {"attempts": 1, "independent_sources": 2}}}
        self.assertEqual(select_dialogue_unknown(curiosity, memory), "vine")

    def test_noise_builds_its_utterance_from_own_gap(self):
        text = make_noise_utterance("fox grapes", {"next_mastery_goal": {"dimension": "words"}},
                                    {"word:vine": {"pressure": 5, "status": "wanting_to_know"}})
        self.assertIn("vine", text)
        self.assertIn("words", text)

    def test_local_reply_is_practice_and_never_evidence(self):
        partner = FakePartner()
        turn = practice_once("fox grapes", {}, {}, partner)
        self.assertEqual(turn["status"], "practiced")
        self.assertEqual(turn["evidence_score"], 0.0)
        self.assertFalse(turn["verified"])
        self.assertIn("vine", turn["observed_forms"])
        self.assertTrue(turn["practice_metrics"]["formed_followup"])
        self.assertTrue(turn["practice_metrics"]["answered_partner_question"])
        self.assertTrue(turn["practice_metrics"]["requested_contrast"])
        self.assertTrue(turn["practice_metrics"]["formed_revision"])
        self.assertIn("unverified", turn["noise_revision"])
        self.assertEqual(len(partner.calls), 2)
        self.assertFalse(turn["practice_metrics"]["independent_evidence_added"])


if __name__ == "__main__":
    unittest.main()
