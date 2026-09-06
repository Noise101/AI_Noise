import unittest

from japanese_corpus_v1 import _clean, _aozora_title, _modernise, KERNEL_SEED, kernel_titles


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
        mixed = "きつねはブドウをみつけました。コップの水をのみました。"
        self.assertEqual(_modernise(mixed), mixed)

    def test_kernel_seed_is_a_small_verified_list(self):
        self.assertTrue(KERNEL_SEED)
        self.assertEqual(len(KERNEL_SEED), len(set(KERNEL_SEED)))

    def test_kernel_titles_offline_falls_back_to_the_seed(self):
        import japanese_corpus_v1 as jc
        orig = jc.aesop_kernel
        jc.aesop_kernel = lambda *a, **k: ()
        try:
            self.assertEqual(kernel_titles(), KERNEL_SEED)
        finally:
            jc.aesop_kernel = orig


if __name__ == "__main__":
    unittest.main()
