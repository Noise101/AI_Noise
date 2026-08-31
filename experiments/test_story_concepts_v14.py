import unittest

from story_concepts_v14 import ConceptEvidence, ConceptExtractor, ConceptLedger, StoryConceptAgent


class StoryConceptTest(unittest.TestCase):
    def test_different_wording_maps_to_shared_concepts(self):
        agent = StoryConceptAgent()
        agent.ingest("A", "https://a.example", 0.8, [
            "A fox saw ripe grapes.", "She missed the grapes.",
        ])
        agent.ingest("B", "https://b.example", 0.9, [
            "The fox saw a bunch of grapes.", "He missed the grapes.",
        ])
        beliefs = agent.ledger.report()["beliefs"]
        failed = next(item for item in beliefs if item["relation"] == "missed")
        self.assertEqual(failed["status"], "corroborated")
        self.assertTrue(failed["accepted_polarity"])
        self.assertEqual(len(failed["citations"]), 2)

    def test_character_belief_is_not_confused_with_narrator_fact(self):
        extractor = ConceptExtractor()
        facts = extractor.extract("The rabbit saw a carrot.", "A", "https://a", 0.8)
        self.assertEqual(facts[0].subject, "rabbit")
        self.assertEqual(facts[0].relation, "saw")
        self.assertEqual(facts[0].object, "carrot")

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
            "She jumped toward them.",
        ])
        beliefs = agent.ledger.report()["beliefs"]
        attempted = next(item for item in beliefs if item["relation"] == "jumped")
        self.assertEqual(attempted["subject"], "fox")

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

    def test_learns_relation_similarity_from_repeated_contexts(self):
        ledger = ConceptLedger()
        for relation in ("looked", "gazed"):
            for subject, obj in (("child", "moon"), ("owl", "moon")):
                ledger.add(ConceptEvidence(subject, relation, obj, True, "observed_event",
                                           relation, f"https://{relation}", 0.8,
                                           f"{relation}-{subject}"))
        groups = ledger.report()["learned_relation_groups"]
        self.assertEqual(groups[0]["relations"], ["gazed", "looked"])
        self.assertEqual(groups[0]["status"], "distributional_candidate")


if __name__ == "__main__":
    unittest.main()
