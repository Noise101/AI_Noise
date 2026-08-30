import unittest
from types import SimpleNamespace

from curiosity_drive_v23 import observe_unknown, resolve_unknown, result_is_grounded


class CuriosityDriveTest(unittest.TestCase):
    def gap(self, observations=1, layer="word"):
        return SimpleNamespace(gap_id=f"{layer}:unknown", layer=layer, query="why?",
                               uncertainty=1.0, observations=observations)

    def test_repeated_unresolved_encounters_increase_wanting_pressure(self):
        ledger = {}
        first = observe_unknown(ledger, self.gap(1), cycle=0)["pressure"]
        repeated = observe_unknown(ledger, self.gap(5), cycle=3)["pressure"]
        self.assertGreater(repeated, first)
        self.assertEqual(ledger["word:unknown"]["status"], "wanting_to_know")

    def test_conversation_has_more_intrinsic_weight_than_a_word(self):
        ledger = {}
        word = observe_unknown(ledger, self.gap(2, "word"), 0)["pressure"]
        conversation = observe_unknown(ledger, self.gap(2, "conversation"), 0)["pressure"]
        self.assertGreater(conversation, word)

    def test_only_grounded_result_satisfies_curiosity(self):
        self.assertEqual(result_is_grounded("word", {"meaning_belief": {"accepted_sense": None}}),
                         (False, None))
        grounded, resolution = result_is_grounded(
            "word", {"meaning_belief": {"accepted_sense": "a bird"}})
        self.assertTrue(grounded)
        ledger = {}
        observe_unknown(ledger, self.gap(), 0)
        resolve_unknown(ledger, "word:unknown", resolution, 1)
        self.assertEqual(ledger["word:unknown"]["status"], "satisfied_for_now")
        self.assertEqual(ledger["word:unknown"]["pressure"], 0)


if __name__ == "__main__":
    unittest.main()
