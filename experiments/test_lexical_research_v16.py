import unittest

from lexical_research_v16 import (
    DefinitionDocument, LexicalMeaningLedger, LexicalResearchAgent, SenseEvidence,
)


class FakeDefinitions:
    def __init__(self, name, definitions):
        self.name, self.definitions = name, definitions

    def lookup(self, word):
        return DefinitionDocument(self.name, f"https://{self.name}.example/{word}",
                                  self.definitions, 0.8)


class LexicalResearchTest(unittest.TestCase):
    def test_generated_gap_is_executed_against_two_sources_and_usage(self):
        agent = LexicalResearchAgent([
            FakeDefinitions("one", ["One more time; once more."]),
            FakeDefinitions("two", ["Another time; a repetition."]),
        ])
        gap = {"kind": "unknown_word_meaning", "form": "again",
               "query": '"again" meaning simple story example'}
        report = agent.investigate(gap, {("again", "and"): 2, ("and", "again"): 2})
        self.assertEqual(report["executed_query"], gap["query"])
        self.assertEqual(len(report["definition_sources"]), 2)
        self.assertEqual(report["meaning_belief"]["accepted_sense"], "repetition")
        self.assertEqual(report["meaning_belief"]["status"], "corroborated")

    def test_multiple_senses_are_alternatives_not_false_conflicts(self):
        agent = LexicalResearchAgent([FakeDefinitions("one", [
            "One more time.", "Back to a previous place or position.",
        ])])
        report = agent.investigate({"form": "again", "query": "again", "kind": "unknown_word_meaning"}, {})
        senses = {item["sense"] for item in report["meaning_belief"]["alternatives"]}
        self.assertEqual(senses, {"repetition", "return_to_prior_state"})

    def test_new_evidence_can_revise_leading_sense(self):
        ledger = LexicalMeaningLedger()
        ledger.add(SenseEvidence("return_to_prior_state", "old", "https://old", "definition", 0.9, "a"))
        ledger.add(SenseEvidence("repetition", "new1", "https://new1", "definition", 0.8, "b"))
        ledger.add(SenseEvidence("repetition", "new2", "https://new2", "usage", 0.8, "c"))
        self.assertEqual(ledger.belief()["accepted_sense"], "repetition")
        self.assertTrue(ledger.revisions)


if __name__ == "__main__":
    unittest.main()
