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


if __name__ == "__main__":
    unittest.main()
