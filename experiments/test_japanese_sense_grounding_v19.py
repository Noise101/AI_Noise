import unittest

from japanese_sense_grounding_v19 import ContextSenseLedger, JapaneseSense, _clean_wikitext


class JapaneseSenseGroundingTest(unittest.TestCase):
    def senses(self):
        return [
            JapaneseSense("鶴", "大型の水鳥。くちばし、首、足が長い。",
                          {"水鳥", "くちばし", "首", "足", "長い"}, ["https://bird"]),
            JapaneseSense("蔓", "植物の細長い茎。他の物に巻きつく。",
                          {"植物", "細長い", "茎", "巻きつく"}, ["https://vine"]),
        ]

    def test_story_features_ground_ambiguous_surface(self):
        ledger = ContextSenseLedger(self.senses())
        ledger.observe("つるは長いくちばしをびんに入れました。水鳥のような姿です。", "https://story")
        belief = ledger.belief()
        self.assertEqual(belief["accepted_sense"], "鶴")
        self.assertEqual(belief["status"], "context_grounded_provisional")

    def test_weak_context_retains_both_senses(self):
        ledger = ContextSenseLedger(self.senses())
        ledger.observe("そこにつるがありました。", "https://story")
        self.assertIsNone(ledger.belief()["accepted_sense"])
        self.assertEqual(len(ledger.belief()["alternatives"]), 2)

    def test_counter_context_can_revise_winner(self):
        ledger = ContextSenseLedger(self.senses())
        ledger.observe("長いくちばしを持つ水鳥です。", "https://first")
        self.assertEqual(ledger.belief()["accepted_sense"], "鶴")
        ledger.observe("植物の細長い茎で、木に巻きつく蔓です。巻きつく茎です。", "https://counter")
        self.assertEqual(ledger.belief()["accepted_sense"], "蔓")
        self.assertTrue(ledger.revisions)

    def test_wikitext_cleaning_does_not_require_language_model(self):
        self.assertEqual(_clean_wikitext("[[くちばし]]、[[くび|首]]が'''長い'''。"), "くちばし、首が長い。")


if __name__ == "__main__":
    unittest.main()
