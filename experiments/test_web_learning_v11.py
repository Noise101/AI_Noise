import unittest

from web_learning_v11 import AutonomousWebLearner, Claim, Evidence, KnowledgeLedger


class FakeProvider:
    def search(self, query, limit=5):
        return [{"id": "Q1", "label": query}]

    def entity(self, entity_id):
        entities = {
            "Q1": {
                "labels": {"en": {"value": "Root"}},
                "descriptions": {"en": {"value": "test root"}},
                "claims": {"P1": [{
                    "mainsnak": {"datavalue": {"value": {"id": "Q2"}}},
                    "rank": "normal", "references": [{"snaks": {}}],
                }]},
                "sitelinks": {"enwiki": {"title": "Root"}},
            },
            "Q2": {
                "labels": {"en": {"value": "Child"}},
                "descriptions": {"en": {"value": "test child"}},
                "claims": {}, "sitelinks": {},
            },
        }
        return entities[entity_id]

    def wikipedia_summary(self, title, language="en"):
        return {"content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Root"}}}


class WebLearningTest(unittest.TestCase):
    def test_generates_followup_goal_and_traverses_without_fixed_query(self):
        learner = AutonomousWebLearner(FakeProvider(), fanout=3)
        report = learner.learn("unknown topic", steps=2)
        self.assertEqual(report["entities_explored"], 2)
        self.assertTrue(any(goal["kind"] == "fill_entity_gap" for goal in report["goals"]))
        self.assertTrue(report["accepted_claims"][0]["source"].startswith("https://"))

    def test_stronger_contradictory_evidence_revises_belief(self):
        ledger = KnowledgeLedger()
        ledger.add_claim(Claim("Q1", "P1", "old", "normal", Evidence("https://a", "web", 0.4, 0)))
        ledger.add_claim(Claim("Q1", "P1", "new", "preferred", Evidence("https://b", "web", 0.8, 2)))
        accepted = [claim.value for claim in ledger.accepted_claims()]
        self.assertEqual(accepted, ["new"])
        self.assertEqual(ledger.conflicts[-1]["accepted"], "new")


if __name__ == "__main__":
    unittest.main()

