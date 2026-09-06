import random
import unittest

from sequence_model_v1 import TinyRNN, build_vocab, collection_key, normalise, source_texts, train_and_evaluate


def audit_from(texts_by_collection: dict[str, list[str]]) -> dict:
    records = {}
    i = 0
    for collection, sentences in texts_by_collection.items():
        for j, sentence in enumerate(sentences):
            records[str(i)] = {"curriculum_admitted": True,
                               "source_url": f"http://x/wiki/{collection}/P{j}",
                               "sentence": sentence, "source_position": j}
            i += 1
    return {"records": records}


def learnable_audit(collections: int = 60) -> dict:
    line = "the fox saw the grapes and the fox wanted the sweet grapes. "
    return audit_from({f"Book{c}": [line] * 12 for c in range(collections)})


def random_audit(collections: int = 60) -> dict:
    rng = random.Random(0)
    alphabet = "abcdefghijklmnop "
    return audit_from({f"Book{c}": ["".join(rng.choice(alphabet) for _ in range(50))
                                    for _ in range(12)] for c in range(collections)})


class SequenceModelTest(unittest.TestCase):
    def test_normalise_keeps_only_lowercase_text_punctuation(self):
        self.assertEqual(normalise("The Fox! (jumped)\n"), "the fox! jumped ")

    def test_source_texts_groups_admitted_sentences_by_url_in_order(self):
        audit = {"records": {
            "a": {"curriculum_admitted": True, "source_url": "u",
                  "sentence": "the second sentence appears here in order.", "source_position": 2},
            "b": {"curriculum_admitted": True, "source_url": "u",
                  "sentence": "the first sentence should lead.", "source_position": 1},
            "c": {"curriculum_admitted": False, "source_url": "u",
                  "sentence": "the dropped one is not admitted.", "source_position": 3}}}
        self.assertEqual(
            source_texts(audit),
            {"u": "the first sentence should lead. the second sentence appears here in order."})

    def test_positive_control_clears_the_paired_significance_test(self):
        audit = learnable_audit()
        first = train_and_evaluate(audit, {}, train_seconds=30, max_steps=400)
        # a real, huge improvement -> very significant on the per-source test
        self.assertGreater(first["improvement_z"], 3.0)
        self.assertLess(first["improvement_p_one_sided"], 0.01)
        self.assertTrue(first["beats_char_baseline_significant"])
        # but "beats_char_baseline" needs it to hold for two consecutive cycles
        self.assertEqual(first["status"], "improvement_not_yet_significant")
        self.assertFalse(first["beats_char_baseline"])
        second = train_and_evaluate(audit, first, train_seconds=30, max_steps=400)
        self.assertEqual(second["status"], "beats_char_baseline")
        self.assertTrue(second["beats_char_baseline"])
        self.assertEqual(second["significant_streak"], 2)
        self.assertLess(second["mean_train_loss"], first["mean_train_loss"] + 0.01)

    def test_negative_control_random_text_is_not_significant(self):
        report = train_and_evaluate(random_audit(), {}, train_seconds=30, max_steps=400)
        self.assertLess(report["improvement_z"], 3.0)
        self.assertFalse(report["beats_char_baseline"])
        self.assertFalse(report["beats_char_baseline_significant"])

    def test_a_tiny_positive_point_gap_that_is_not_significant_earns_no_credit(self):
        # one held-out source with a hair of improvement: z stays tiny, no credit
        audit = audit_from({f"Book{c}": ["the fox saw the food and the fox ran home."] * 6
                            for c in range(30)})
        report = train_and_evaluate(audit, {}, train_seconds=30, max_steps=15)
        if report["improvement_bits"] and report["improvement_bits"] > 0:
            self.assertFalse(report["beats_char_baseline"] and report["improvement_z"] < 3.0)

    def test_benchmark_is_frozen_and_never_trains_on_held_out_sources(self):
        audit = learnable_audit()
        first = train_and_evaluate(audit, {}, train_seconds=30, max_steps=100)
        held = set(first["benchmark"]["held_out_collections"])
        self.assertTrue(held)
        grown = learnable_audit(80)
        second = train_and_evaluate(grown, first, train_seconds=30, max_steps=100)
        self.assertEqual(set(second["benchmark"]["held_out_collections"]), held)
        # a held-out collection_key must never be a training source
        texts = source_texts(grown)
        train_keys = {collection_key(u) for u in texts
                      if collection_key(u) not in held}
        self.assertFalse(train_keys & held)

    def test_insufficient_text_postpones_cleanly(self):
        report = train_and_evaluate(audit_from({"Book0": ["a short line here."]}), {})
        self.assertEqual(report["status"], "insufficient_text")
        self.assertFalse(report["benchmark"]["locked"])
        self.assertIsNone(report["held_out_bits_per_char"])
        self.assertFalse(report["can_sample"])

    def test_sample_only_emits_vocabulary_characters(self):
        report = train_and_evaluate(learnable_audit(), {}, train_seconds=30, max_steps=100)
        model = TinyRNN(report["state"]["vocab"], report["state"])
        text = model.sample("the ", length=80, rng=random.Random(1))
        self.assertEqual(len(text), 80)
        self.assertTrue(set(text).issubset(set(report["state"]["vocab"])))

    def test_bits_per_char_is_non_negative_and_zero_for_trivial_input(self):
        model = TinyRNN(["a", "b", "c", " "])
        value, n = model.bits_per_char("a")
        self.assertEqual((value, n), (0.0, 0))
        value, n = model.bits_per_char("abc abc")
        self.assertGreaterEqual(value, 0.0)

    def test_learning_curve_tracks_improvement_bits_over_cycles(self):
        audit = learnable_audit()
        state = {}
        for _ in range(3):
            state = train_and_evaluate(audit, state, train_seconds=30, max_steps=90)
        self.assertGreaterEqual(len(state["learning_curve"]), 2)
        self.assertIn("improvement_bits", state["learning_curve"][-1])


if __name__ == "__main__":
    unittest.main()
