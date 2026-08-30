import unittest

from kanjipedia_reference_v22 import exact_entry_from_html


class KanjipediaReferenceTest(unittest.TestCase):
    def test_accepts_only_an_exact_visible_entry(self):
        page = '<a href="/kanji/1">鶴</a><a href="/kotoba/2">鶴亀</a>'
        self.assertEqual(exact_entry_from_html("鶴", page), {"path": "/kanji/1", "label": "鶴"})
        self.assertIsNone(exact_entry_from_html("亀", page))

    def test_decodes_entities_but_does_not_extract_definition_prose(self):
        page = '<a href="/kotoba/3">一&amp;二</a><p>copyrighted definition</p>'
        self.assertEqual(exact_entry_from_html("一&二", page),
                         {"path": "/kotoba/3", "label": "一&二"})


if __name__ == "__main__":
    unittest.main()
