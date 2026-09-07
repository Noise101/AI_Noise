import random
import unittest

import japanese_sequence_v1 as js


def learnable_texts(n=40):
    line = "むかしむかしおじいさんとおばあさんがいました。おじいさんは山へ行きました。"
    return {f"https://ja.wikisource.org/wiki/本{c}": line * 10 for c in range(n)}


def random_texts(n=40):
    rng = random.Random(0)
    kana = "あいうえおかきくけこさしすせそたちつてと。"
    return {f"https://ja.wikisource.org/wiki/乱{c}":
            "".join(rng.choice(kana) for _ in range(700)) for c in range(n)}


class JapaneseSequenceTest(unittest.TestCase):
    def test_normalise_keeps_kana_kanji_and_core_punctuation_only(self):
        self.assertEqual(js.normalise("Fox「桃」を食べた。\n 123"), "「桃」を食べた。")

    def test_collection_key_is_per_work(self):
        a = js.collection_key("https://www.aozora.gr.jp/cards/000121/files/627_13466.html")
        b = js.collection_key("https://www.aozora.gr.jp/cards/000121/files/4253_10259.html")
        self.assertNotEqual(a, b)
        self.assertEqual(
            js.collection_key("https://ja.wikisource.org/wiki/桃太郎"),
            js.collection_key("https://ja.wikisource.org/wiki/桃太郎"))

    def test_positive_control_beats_the_char_baseline_after_two_cycles(self):
        texts = learnable_texts()
        first = js.train_and_evaluate(texts, {}, train_seconds=30, max_steps=500)
        self.assertGreater(first["improvement_z"], js.SIGNIFICANCE_Z)
        self.assertTrue(first["beats_char_baseline_significant"])
        self.assertFalse(first["beats_char_baseline"])
        second = js.train_and_evaluate(texts, first, train_seconds=30, max_steps=500)
        self.assertEqual(second["status"], "beats_char_baseline")
        self.assertTrue(second["beats_char_baseline"])
        self.assertEqual(second["significant_streak"], 2)

    def test_negative_control_random_kana_is_not_significant(self):
        report = js.train_and_evaluate(random_texts(), {}, train_seconds=30, max_steps=500)
        self.assertLess(report["improvement_z"], js.SIGNIFICANCE_Z)
        self.assertFalse(report["beats_char_baseline"])

    def test_insufficient_text_reports_cleanly(self):
        report = js.train_and_evaluate({"https://ja.wikisource.org/wiki/短": "みじかい。"}, {})
        self.assertEqual(report["status"], "insufficient_text")
        self.assertIsNone(report["held_out_bits_per_char"])
        self.assertFalse(report["can_sample"])

    def test_benchmark_split_is_frozen_across_growth(self):
        first = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=100)
        held = set(first["benchmark"]["held_out_collections"])
        self.assertTrue(held)
        second = js.train_and_evaluate(learnable_texts(60), first, train_seconds=30, max_steps=100)
        self.assertEqual(set(second["benchmark"]["held_out_collections"]), held)

    def test_generate_returns_japanese_characters(self):
        model = js.TinyRNN(sorted("むかしあおじいん。やまへ行きました"))
        out = js.generate(model, "むかし", length=20)
        self.assertEqual(len(out), 20)
        self.assertTrue(all("぀" <= ch or ch in "。" for ch in out))

    def test_model_fingerprint_moves_with_weights_and_steps(self):
        r1 = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertTrue(r1["model_fingerprint"])
        self.assertEqual(r1["model_fingerprint"], r1["state"]["model_fingerprint"])
        same = js.model_fingerprint(r1["state"], r1["state"]["steps_trained"])
        self.assertEqual(same, r1["model_fingerprint"])
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertNotEqual(r2["model_fingerprint"], r1["model_fingerprint"])
        self.assertGreater(r2["state"]["steps_trained"], r1["state"]["steps_trained"])

    # --- re-audit #5 item 1: contamination reset ------------------------
    def _ctx(self, forbidden=()):
        return {"parser_version": 5, "provenance_policy": js.PROVENANCE_POLICY,
                "read_only": True, "forbidden_collections": list(forbidden)}

    def _clean_state(self):
        r = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=80,
                                  training_context=self._ctx())
        return r

    def test_legacy_state_without_training_regime_is_not_continued(self):
        legacy = {"state": {"vocab": list("あいうえお"), "Wxh": [[0.1] * 5] * 24,
                            "Whh": [[0.1] * 24] * 24, "Why": [[0.1] * 24] * 5,
                            "bh": [0.0] * 24, "by": [0.0] * 5},
                  "steps_trained": 2_380_000, "model_fingerprint": "legacyFP00000000"}
        r = js.train_and_evaluate(learnable_texts(40), legacy, train_seconds=30, max_steps=60,
                                  training_context=self._ctx())
        self.assertEqual(r["reset_reason"], "legacy_state_without_training_regime")
        self.assertEqual(r["contamination_status"], "retired_replaced")
        self.assertLess(r["steps_trained"], 1000)                 # 0-based, not +2.38M
        self.assertEqual(r["parent_model_fingerprint"], "legacyFP00000000")
        self.assertEqual(r["retired_model"]["retired_steps_trained"], 2_380_000)
        self.assertTrue(r["retired_model"]["retired_state"])

    def test_a_clean_state_continues_on_the_same_boundary(self):
        r1 = self._clean_state()
        self.assertIsNone(r1["reset_reason"])
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertIsNone(r2["reset_reason"])
        self.assertEqual(r2["contamination_status"], "clean_continued")
        self.assertGreater(r2["steps_trained"], r1["steps_trained"])

    def test_new_read_books_are_growth_not_a_reset(self):
        r1 = self._clean_state()
        r2 = js.train_and_evaluate(learnable_texts(70), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())      # more training data
        self.assertIsNone(r2["reset_reason"])
        self.assertGreater(r2["steps_trained"], r1["steps_trained"])

    def test_a_boundary_change_retires_the_model(self):
        r1 = self._clean_state()
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx(forbidden=["https://x/wiki/Y"]))
        self.assertEqual(r2["reset_reason"], "training_boundary_changed")
        self.assertLess(r2["steps_trained"], r1["steps_trained"])
        self.assertEqual(r2["parent_model_fingerprint"], r1["model_fingerprint"])

    def test_a_provenance_policy_change_retires_the_model(self):
        r1 = self._clean_state()
        ctx2 = self._ctx(); ctx2["provenance_policy"] = "some_other_policy"
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=ctx2)
        self.assertEqual(r2["reset_reason"], "training_boundary_changed")

    def test_a_forbidden_collection_in_training_retires_the_model(self):
        r1 = self._clean_state()
        # a selection/final collection sneaks into the training texts
        texts = dict(learnable_texts(40))
        bad_col = js._collection_for_forbidden(next(iter(texts)))
        r2 = js.train_and_evaluate(texts, r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx(forbidden=[bad_col]))
        self.assertIn(r2["reset_reason"],
                      ("training_boundary_changed", "forbidden_collection_present_in_training_set"))
        self.assertLess(r2["steps_trained"], r1["steps_trained"])

    def test_the_retired_state_is_returned_for_archiving_and_streaks_reset(self):
        legacy = {"state": {"vocab": list("あいうえお"), "Wxh": [[0.1] * 5] * 24,
                            "Whh": [[0.1] * 24] * 24, "Why": [[0.1] * 24] * 5,
                            "bh": [0.0] * 24, "by": [0.0] * 5},
                  "steps_trained": 999_999, "model_fingerprint": "x",
                  "significant_streak": 2, "beats_char_baseline_significant": True}
        r = js.train_and_evaluate(learnable_texts(40), legacy, train_seconds=30, max_steps=40,
                                  training_context=self._ctx())
        self.assertTrue(r["retired_model"]["retired_state"])
        self.assertLessEqual(r["significant_streak"], 1)        # not inherited (was 2)
        self.assertFalse(r.get("beats_char_baseline"))          # needs a fresh 2-in-a-row
        self.assertEqual(len(r["retirement_log"]), 1)

    def test_training_data_fingerprint_is_auditable(self):
        r = self._clean_state()
        tdf = r["training_data_fingerprint"]
        for k in ("boundary_fingerprint", "training_set_fingerprint", "training_regime",
                  "parser_version", "provenance_policy", "normalisation_version",
                  "training_source_count", "training_sources", "forbidden_collections"):
            self.assertIn(k, tdf)
        self.assertEqual(len(tdf["training_sources"]), tdf["training_source_count"])


if __name__ == "__main__":
    unittest.main()
