import unittest

import japanese_corpus_v1 as jc
from japanese_corpus_v1 import (_clean, _aozora_title, _modernise, KERNEL_SEED, kernel_titles,
                                _sentence_level, tatoeba_readers)


_FAKE_TATOEBA = [
    f"{i}\tjpn\t{s}" for i, s in enumerate([
        "私は山にいました。", "ねこがねむっています。", "とりがそらをとびました。",
        "こどもがわらいました。", "いぬがはしります。", "はながさきました。",
        "あめがふっています。", "つきがでました。", "かぜがふきました。",
        "むしがなきました。", "ふねがきました。", "とりがなきました。",
        "きつねがはしりました。", "うさぎがはねました。", "さかながおよぎました。",
        "ほしがひかりました。", "くもがうごきました。", "ゆきがつもりました。",
        "かわがながれました。", "はっぱがおちました。", "とけいがとまりました。",
        "でんしゃがきました。", "パン。", "何？", "abc123 is here.",
    ], start=1000)
]


class JapaneseCorpusTest(unittest.TestCase):
    def test_clean_strips_headings_ruby_and_editorial_notes(self):
        raw = ("== 一 ==\n桃《もも》太郎《たろう》は 川《かわ》へ 行きました。\n"
               "[#ここから2字下げ]\n"
               "底本：「日本の伝説と童話」\nこの作品は、パブリックドメインです。")
        cleaned = _clean(raw)
        self.assertIn("桃太郎は川へ行きました。", cleaned)
        self.assertNotIn("もも", cleaned)
        self.assertNotIn("底本", cleaned)
        self.assertNotIn("パブリックドメイン", cleaned)
        self.assertNotIn("==", cleaned)

    def test_aozora_title_prefers_the_clean_work_title_not_the_filename(self):
        html = ('<head><meta name="DC.Title" content="赤い蝋燭" />'
                '<title>新美南吉 赤い蝋燭</title></head>'
                '<body><h1 class="title">赤い蝋燭</h1></body>')
        self.assertEqual(_aozora_title(html), "赤い蝋燭")
        # meta-only fallback (some older cards have no <h1 class="title">)
        self.assertEqual(
            _aozora_title('<meta  name="DC.Title"  content="ごん狐" />'), "ごん狐")
        self.assertEqual(_aozora_title("<html>no title here</html>"), "")

    def test_modernise_folds_all_katakana_orthography_to_hiragana(self):
        katakana = "アルトキ、イヌガニクヲクワエテ、ハシヲワタリマシタ。"
        out = _modernise(katakana)
        self.assertIn("いぬがにくを", out)
        self.assertNotIn("イヌ", out)

    def test_modernise_leaves_normal_mixed_text_and_loanwords_alone(self):
        mixed = "きつねはブドウをみつけました。コップの水をのみました。今日はいい天気だ。"
        self.assertEqual(_modernise(mixed), mixed)

    def test_modernise_folds_historical_kana_to_modern(self):
        old = ("あひるさんのお母さんは、真赤な洋服をかつてやりたいとおもひました。"
               "きれいだらうと、ゐましたが、あひるちやんはこまつてしまひました。"
               "「きません」といひました。")
        out = _modernise(old)
        self.assertIn("かって", out)
        self.assertIn("おもいました", out)
        self.assertIn("こまってしまいました", out)
        self.assertIn("といいました", out)
        self.assertIn("ちゃん", out)
        self.assertNotIn("ゐ", out)
        # あひる (家鴨) keeps its ひ -- it is not old orthography
        self.assertIn("あひる", out)

    def test_kernel_seed_is_a_small_verified_list(self):
        self.assertTrue(KERNEL_SEED)
        self.assertEqual(len(KERNEL_SEED), len(set(KERNEL_SEED)))

    def test_kernel_titles_offline_falls_back_to_the_seed(self):
        orig = jc.aesop_kernel
        jc.aesop_kernel = lambda *a, **k: ()
        try:
            self.assertEqual(kernel_titles(), KERNEL_SEED)
        finally:
            jc.aesop_kernel = orig

    def test_sentence_level_rises_with_kanji_and_length(self):
        self.assertLess(_sentence_level("ねこがねむる。"), _sentence_level("彼は複雑な問題を解決した。"))

    def test_tatoeba_readers_bundles_clean_short_sentences(self):
        jc._CLEAN_POOL.clear()
        readers = tatoeba_readers(2.0, n_readers=2, per_reader=5, band=3.0,
                                  _lines=_FAKE_TATOEBA)
        self.assertEqual(len(readers), 2)
        for r in readers:
            self.assertTrue(r.url.startswith("tatoeba://reader/2.0/"))
            lines = r.text.split("\n")
            self.assertEqual(len(lines), 5)
            self.assertNotIn("パン。", lines)          # too short
            self.assertNotIn("abc123 is here.", lines)  # latin/digits
        # skip advances to fresh material
        first = tatoeba_readers(2.0, n_readers=1, per_reader=5, band=3.0, _lines=_FAKE_TATOEBA)
        later = tatoeba_readers(2.0, n_readers=1, per_reader=5, band=3.0, skip=5,
                                _lines=_FAKE_TATOEBA)
        self.assertNotEqual(first[0].text, later[0].text)

    def test_tatoeba_offline_returns_empty(self):
        jc._CLEAN_POOL.clear()
        self.assertEqual(tatoeba_readers(2.0, _lines=[]), [])


if __name__ == "__main__":
    unittest.main()
