import unittest

from developmental_language_v15 import DevelopmentalLexicon
from lexical_research_v16 import DefinitionDocument
from phrase_learning_v17 import PhraseResearchAgent


class FakeDefinitions:
    def __init__(self, name, definition):
        self.name, self.definition = name, definition

    def lookup(self, phrase):
        return DefinitionDocument(self.name, f"https://{self.name}.example/{phrase}",
                                  [self.definition], 0.8)


class PhraseLearningTest(unittest.TestCase):
    def test_discovers_bigrams_and_trigrams_without_fixed_phrase_list(self):
        lexicon = DevelopmentalLexicon()
        lexicon.observe("He tried again and again.")
        lexicon.observe("She jumped again and again.")
        candidates = {item["phrase"] for item in lexicon.phrase_candidates()}
        self.assertIn("again and again", candidates)
        self.assertEqual(lexicon.phrase_gap()["kind"], "unknown_phrase_meaning")

    def test_two_definitions_corroborate_phrase_sense(self):
        agent = PhraseResearchAgent([
            FakeDefinitions("one", "After a long time; eventually."),
            FakeDefinitions("two", "Finally, after waiting."),
        ])
        gap = {"kind": "unknown_phrase_meaning", "form": "at last", "query": '"at last" phrase meaning'}
        result = agent.investigate(gap, {})
        self.assertEqual(result["meaning_belief"]["accepted_sense"], "eventually_after_delay")
        self.assertEqual(result["meaning_belief"]["status"], "corroborated")
        self.assertEqual(result["compositionality"], "unknown_until_component_words_are_grounded")

    def test_phrase_is_not_called_idiom_before_component_comparison(self):
        belief = {"accepted_sense": "eventually_after_delay"}
        self.assertEqual(
            PhraseResearchAgent.compositionality("at last", belief, {"last": {"accepted_sense": "final_item"}}),
            "unknown_until_component_words_are_grounded",
        )
        grounded = {"at": {"accepted_sense": "location"}, "last": {"accepted_sense": "final_item"}}
        self.assertEqual(PhraseResearchAgent.compositionality("at last", belief, grounded),
                         "noncompositional_candidate")


if __name__ == "__main__":
    unittest.main()
