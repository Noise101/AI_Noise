import unittest

from dialogue_web_verification_v48 import (Observation, empty_verification_memory,
                                            verify_dialogue_hypothesis)


class FakeSource:
    def __init__(self, name, passage):
        self.name, self.passage = name, passage

    def search(self, expression):
        return [Observation(self.name, f"https://{self.name}/one", self.passage)]


class DialogueWebVerificationTest(unittest.TestCase):
    def test_two_sources_can_support_a_structural_role_without_proving_meaning(self):
        turn = {"unknown_expression": "with the", "hypothesis_focus": "",
                "structural_hypothesis": {"status": "testable_candidate",
                    "predicted_role": "relation_between_neighboring_entities"}}
        result = verify_dialogue_hypothesis(turn, empty_verification_memory(), [
            FakeSource("source-a", "A child walked with the dog."),
            FakeSource("source-b", "A cup came with the meal."),
        ])
        self.assertEqual(result["hypothesis_status"],
                         "supported_structural_candidate_not_meaning_proof")
        self.assertEqual(result["structural_supporting_sources"], 2)
        self.assertFalse(result["meaning_committed"])

    def test_noise_rejects_partner_focus_when_independent_contexts_disagree(self):
        turn = {"unknown_expression": "in the", "hypothesis_focus": "cat",
                "partner_reply": "The local model says it connects to cat."}
        memory = empty_verification_memory()
        result = verify_dialogue_hypothesis(turn, memory, [
            FakeSource("source-a", "A bird slept in the tree during rain."),
            FakeSource("source-b", "The cup remained in the box all day."),
        ])
        self.assertEqual(result["status"], "usage_corroborated")
        self.assertEqual(result["hypothesis_status"], "rejected_as_overspecific")
        self.assertFalse(result["partner_claim_used_as_evidence"])
        self.assertFalse(result["meaning_committed"])
        self.assertEqual(result["final_judgment_made_by"], "noise_evidence_rule_v1")
        self.assertEqual(memory["summary"]["local_llm_claims_accepted_as_fact"], 0)

    def test_two_independent_contexts_can_support_but_not_prove_focus(self):
        turn = {"unknown_expression": "in the", "hypothesis_focus": "box"}
        result = verify_dialogue_hypothesis(turn, empty_verification_memory(), [
            FakeSource("source-a", "A toy was in the red box."),
            FakeSource("source-b", "We found it in the box yesterday."),
        ])
        self.assertEqual(result["hypothesis_status"], "supported_candidate_not_meaning_proof")
        self.assertFalse(result["meaning_committed"])

    def test_prior_corroboration_prevents_repeated_search(self):
        memory = empty_verification_memory()
        memory["expressions"]["of the"] = {
            "independent_sources": 2, "hypothesis_status": "unresolved", "attempts": 1}
        result = verify_dialogue_hypothesis(
            {"unknown_expression": "of the"}, memory,
            [FakeSource("unused", "of the")])
        self.assertEqual(result["status"], "already_independently_observed")
        self.assertEqual(len(memory["observations"]), 0)


if __name__ == "__main__":
    unittest.main()
