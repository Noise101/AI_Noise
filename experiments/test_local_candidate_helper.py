import unittest

from local_candidate_helper import LocalProposal, NullCandidateHelper, OllamaCandidateHelper


class LocalCandidateHelperTest(unittest.TestCase):
    def test_null_backend_keeps_core_system_independent(self):
        helper = NullCandidateHelper()
        self.assertFalse(helper.available())
        self.assertIsNone(helper.propose_japanese_senses("つる", "つるがいました"))

    def test_proposal_is_explicitly_not_evidence(self):
        proposal = LocalProposal("local", "candidate", [{"label": "鶴"}])
        self.assertFalse(proposal.verified)
        self.assertEqual(proposal.evidence_score, 0.0)


if __name__ == "__main__":
    unittest.main()
