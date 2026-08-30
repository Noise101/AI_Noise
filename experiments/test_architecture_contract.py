import unittest

from developmental_language_v15 import MultiLevelLearningAgent
from local_candidate_helper import LocalProposal, NullCandidateHelper
from story_learning_v12 import StoryLearner


class ArchitectureContractTest(unittest.TestCase):
    def test_core_learning_runs_without_local_or_remote_llm(self):
        helper = NullCandidateHelper()
        self.assertFalse(helper.available())
        agent = MultiLevelLearningAgent()
        agent.observe_source("fixture", "https://fixture.example", 0.8,
                             ["Fox sees grapes.", "Fox jumps high."])
        self.assertGreater(agent.story.report()["events_seen"], 0)
        self.assertGreater(agent.lexicon.report()["character_inventory"], 0)

    def test_local_model_proposal_cannot_be_evidence(self):
        proposal = LocalProposal("local", "candidate", [{"label": "鶴"}])
        self.assertFalse(proposal.verified)
        self.assertEqual(proposal.evidence_score, 0.0)

    def test_prediction_failure_changes_internal_belief(self):
        learner = StoryLearner()
        learner.observe_story(["Fox sees grapes.", "Fox jumps high."])
        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        self.assertEqual(learner.report()["mistakes_detected"], 1)


if __name__ == "__main__":
    unittest.main()
