import unittest

from story_concepts_v14 import ConceptEvidence, ConceptExtractor, ConceptLedger, StoryConceptAgent


class StoryConceptTest(unittest.TestCase):
    def test_different_wording_maps_to_shared_concepts(self):
        agent = StoryConceptAgent()
        agent.ingest("A", "https://a.example", 0.8, [
            "A fox saw ripe grapes.", "She used all her tricks but could not reach the grapes.",
        ])
        agent.ingest("B", "https://b.example", 0.9, [
            "The fox came to a bunch of grapes.", "He jumped but missed the grapes.",
        ])
        beliefs = agent.ledger.report()["beliefs"]
        failed = next(item for item in beliefs if item["relation"] == "obtains")
        self.assertEqual(failed["status"], "corroborated")
        self.assertFalse(failed["accepted_polarity"])
        self.assertEqual(len(failed["citations"]), 2)

    def test_character_belief_is_not_confused_with_narrator_fact(self):
        extractor = ConceptExtractor()
        facts = extractor.extract("The fox saw ripe grapes.", "A", "https://a", 0.8)
        beliefs = extractor.extract('The fox said, "The grapes are sour."', "A", "https://a", 0.8)
        scopes = {(item.object, item.scope) for item in facts + beliefs if item.relation == "quality"}
        self.assertIn(("ripe", "narrator_fact"), scopes)
        self.assertIn(("sour", "fox_belief"), scopes)

    def test_counterevidence_changes_belief_and_records_revision(self):
        ledger = ConceptLedger()
        def item(source, polarity, score=0.8):
            return ConceptEvidence("fox", "obtains", "grapes", polarity, "narrator_fact",
                                   source, f"https://{source}", score, source)
        ledger.add(item("old", True))
        ledger.add(item("counter1", False))
        self.assertEqual(ledger.belief(("fox", "obtains", "grapes", "narrator_fact"))["status"], "disputed")
        ledger.add(item("counter2", False))
        belief = ledger.belief(("fox", "obtains", "grapes", "narrator_fact"))
        self.assertFalse(belief["accepted_polarity"])
        self.assertEqual(belief["status"], "provisional")
        self.assertGreaterEqual(len(ledger.revisions), 2)

    def test_resolves_child_story_pronouns_from_recent_entities(self):
        agent = StoryConceptAgent()
        agent.ingest("A", "https://a", 0.8, [
            "A fox saw ripe grapes.",
            "She used every trick but could not reach them.",
        ])
        beliefs = agent.ledger.report()["beliefs"]
        failed = next(item for item in beliefs if item["relation"] == "obtains")
        self.assertFalse(failed["accepted_polarity"])
        attempted = next(item for item in beliefs if item["relation"] == "attempts_to_obtain")
        self.assertTrue(attempted["accepted_polarity"])

    def test_cited_conclusion_exposes_uncertainty(self):
        belief = {
            "subject": "fox", "relation": "obtains", "object": "grapes",
            "scope": "narrator_fact", "status": "single_source", "accepted_polarity": False,
            "confidence": 1.0, "citations": ["https://source.example/story"],
        }
        conclusion = StoryConceptAgent.render_conclusion(belief)
        self.assertIn("does not obtain", conclusion["claim"])
        self.assertIn("only one", conclusion["uncertainty"])
        self.assertEqual(conclusion["citations"], ["https://source.example/story"])


if __name__ == "__main__":
    unittest.main()
