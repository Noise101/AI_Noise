import unittest

from developmental_language_v15 import DevelopmentalLexicon, MultiLevelLearningAgent
from story_learning_v12 import Event


class DevelopmentalLanguageTest(unittest.TestCase):
    def test_learns_characters_words_phrase_and_meaning_in_parallel(self):
        lexicon = DevelopmentalLexicon()
        for _ in range(2):
            lexicon.observe("Fox sees ripe grapes.", Event("fox", "sees", "ripe_grapes"))
        report = lexicon.report()
        self.assertGreater(report["character_inventory"], 5)
        self.assertEqual(report["word_forms"]["fox"], 2)
        self.assertTrue(any(item["phrase"] == "ripe grapes" for item in report["phrase_candidates"]))
        fox = next(item for item in report["grounded_meanings"] if item["form"] == "fox")
        self.assertEqual(fox["dominant_role"], "agent")

    def test_unknown_word_generates_its_own_query(self):
        lexicon = DevelopmentalLexicon()
        lexicon.observe("Fox feels wistful.", Event("fox", "feels", ""))
        gap = lexicon.lexical_gap()
        self.assertEqual(gap["kind"], "unknown_word_meaning")
        self.assertIn(gap["form"], gap["query"])

    def test_japanese_repetition_proposes_boundaries_without_tokenizer(self):
        lexicon = DevelopmentalLexicon()
        lexicon.observe("きつねはぶどうをみた。きつねはぶどうをみた。")
        chunks = lexicon.phrase_candidates(minimum_count=2)
        self.assertTrue(any(item["kind"] == "unsegmented_chunk_candidate" for item in chunks))
        self.assertEqual(lexicon.lexical_gap()["kind"], "unknown_word_boundary")

    def test_same_source_updates_all_learning_levels(self):
        agent = MultiLevelLearningAgent()
        agent.observe_source("A", "https://a", 0.8,
                             ["A fox saw ripe grapes.", "She could not reach them."])
        self.assertGreater(agent.lexicon.sentences_seen, 0)
        self.assertGreater(agent.story.report()["events_seen"], 0)
        self.assertGreater(agent.concepts.ledger.report()["evidence_count"], 0)

    def test_seen_function_word_is_not_claimed_as_understood_content(self):
        lexicon = DevelopmentalLexicon()
        lexicon.observe("Fox jumps at the grapes.", Event("fox", "jumps", "at_the_grapes"))
        self.assertIn("the", lexicon.report()["word_forms"])
        grounded = {item["form"] for item in lexicon.report()["grounded_meanings"]}
        self.assertNotIn("the", grounded)
        self.assertNotIn("at", grounded)

    def test_sourced_meaning_is_written_back_and_can_be_revised(self):
        lexicon = DevelopmentalLexicon()
        lexicon.update_meaning_hypothesis("again", {
            "status": "single_source", "accepted_sense": "return_to_prior_state",
            "leading_sense": "return_to_prior_state", "alternatives": [],
        })
        lexicon.update_meaning_hypothesis("again", {
            "status": "corroborated", "accepted_sense": "repetition",
            "leading_sense": "repetition", "alternatives": [{"citations": ["https://source"]}],
        })
        report = lexicon.report()
        self.assertEqual(report["researched_meanings"]["again"]["accepted_sense"], "repetition")
        self.assertTrue(report["meaning_revisions"])

    def test_repeated_dialogue_cue_becomes_its_own_unknown(self):
        lexicon = DevelopmentalLexicon()
        lexicon.observe('Fox said, "I am hungry."')
        lexicon.observe('Crow said, "I have food."')
        gap = lexicon.conversation_gap()
        self.assertEqual(gap["kind"], "unknown_conversation_act")
        self.assertEqual(gap["form"], "said")
        self.assertEqual(gap["observations"], 2)
        self.assertIn("dialogue", gap["query"])


if __name__ == "__main__":
    unittest.main()
