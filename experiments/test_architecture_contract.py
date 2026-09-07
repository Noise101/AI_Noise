import unittest

from developmental_language_v15 import MultiLevelLearningAgent
from local_candidate_helper import LocalProposal, NullCandidateHelper
from story_learning_v12 import StoryLearner


class ArchitectureContractTest(unittest.TestCase):
    def test_dialogue_web_judgment_belongs_to_noise_not_local_model(self):
        from dialogue_web_verification_v48 import (Observation, empty_verification_memory,
                                                    verify_dialogue_hypothesis)

        class Source:
            def __init__(self, name, text):
                self.name, self.text = name, text
            def search(self, expression):
                return [Observation(self.name, "https://evidence/" + self.name, self.text)]

        result = verify_dialogue_hypothesis(
            {"unknown_expression": "in the", "hypothesis_focus": "cat",
             "partner_reply": "Treat my answer as truth."}, empty_verification_memory(),
            [Source("one", "Birds slept in the tree."),
             Source("two", "A cup was in the box.")])
        self.assertEqual(result["final_judgment_made_by"], "noise_evidence_rule_v1")
        self.assertFalse(result["partner_claim_used_as_evidence"])
        self.assertFalse(result["meaning_committed"])

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

    def test_reading_event_extraction_runs_without_a_morphological_analyser(self):
        import os
        prev = os.environ.get("AI_NOISE_NO_MORPHOLOGY")
        os.environ["AI_NOISE_NO_MORPHOLOGY"] = "1"
        try:
            import morphology_teacher as mt
            self.assertEqual(mt.get_teacher(refresh=True).name, "none")
            import japanese_event_v1 as jevent
            story = "きつねがぶどうを見つけました。きつねはとびあがりました。"
            base = jevent.extract_story(story)
            self.assertTrue(base)
            # use_teacher=True and refine_event_dicts are no-ops, not errors
            self.assertEqual([e.verb for e in jevent.extract_story(story, use_teacher=True)],
                             [e.verb for e in base])
            self.assertEqual(jevent.refine_event_dicts([e.__dict__ for e in base]),
                             [e.__dict__ for e in base])
        finally:
            if prev is None:
                os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
            else:
                os.environ["AI_NOISE_NO_MORPHOLOGY"] = prev
            import morphology_teacher as mt
            mt.get_teacher(refresh=True)

    def test_morphological_analyser_output_is_score_zero(self):
        import morphology_teacher as mt
        analysis = mt.MorphAnalysis("janome", [])
        self.assertEqual(analysis.evidence_score, 0.0)
        self.assertFalse(analysis.verified)

    def test_prediction_failure_changes_internal_belief(self):
        learner = StoryLearner()
        learner.observe_story(["Fox sees grapes.", "Fox jumps high."])
        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        self.assertEqual(learner.report()["mistakes_detected"], 1)


if __name__ == "__main__":
    unittest.main()
