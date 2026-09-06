import unittest

from japanese_corpus_v1 import _clean, _aozora_title, KERNEL


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

    def test_kernel_is_a_non_empty_ordered_seed_list(self):
        self.assertGreaterEqual(len(KERNEL), 8)
        self.assertEqual(len(KERNEL), len(set(KERNEL)))


if __name__ == "__main__":
    unittest.main()
