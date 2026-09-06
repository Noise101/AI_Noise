import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import japanese_reader_v1 as reader


@dataclass
class FakeText:
    title: str
    url: str
    text: str


_STORY = (
    "むかしむかし、おじいさんが山へ行きました。"
    "おじいさんは木を切りました。"
    "犬が川で魚を見つけました。"
    "犬は魚を食べました。"
    "おばあさんが家へ帰りました。"
)


def _fake_books(n):
    return [FakeText(f"物語{i}", f"http://fake/story/{i}", _STORY) for i in range(n)]


class JapaneseReaderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runtime = Path(self._tmp.name)
        self._books = _fake_books(8)
        self._fetch_calls = []

        def fake_fetch(title):
            self._fetch_calls.append(title)
            for b in self._books:
                if b.title == title:
                    return b
            return None

        self._orig = {
            "KERNEL": reader.corpus.KERNEL,
            "fetch": reader.corpus.fetch,
            "AOZORA_AUTHORS": reader.corpus.AOZORA_AUTHORS,
            "aozora_author_works": reader.corpus.aozora_author_works,
            "fetch_aozora": reader.corpus.fetch_aozora,
        }
        reader.corpus.KERNEL = tuple(b.title for b in self._books)
        reader.corpus.fetch = fake_fetch
        reader.corpus.AOZORA_AUTHORS = {}
        reader.corpus.aozora_author_works = lambda *a, **k: []
        reader.corpus.fetch_aozora = lambda *a, **k: None

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(reader.corpus, k, v)
        self._tmp.cleanup()

    def test_first_cycle_fetches_books_and_reads_one(self):
        status = reader.run_once(self.runtime)
        self.assertGreater(status["books_fetched"], 0)
        self.assertIn(status["reading"]["status"],
                      ("reread", "graduated", "shelved_no_progress"))
        self.assertTrue((self.runtime / reader.CURRICULUM_FILE).exists())
        self.assertTrue((self.runtime / reader.EVENTS_FILE).exists())

    def test_fetch_is_throttled_by_a_cooldown(self):
        reader.run_once(self.runtime)
        first = len(self._fetch_calls)
        self.assertGreater(first, 0)
        # next cycle is within the cooldown window -> no new network fetch
        reader.run_once(self.runtime)
        self.assertEqual(len(self._fetch_calls), first)

    def test_state_is_isolated_to_the_runtime_dir(self):
        reader.run_once(self.runtime)
        for name in (reader.CURRICULUM_FILE, reader.EVENTS_FILE,
                     reader.COMPREHENSION_FILE, reader.STATUS_FILE):
            self.assertTrue((self.runtime / name).exists())
        # a second independent runtime does not see the first's books
        with tempfile.TemporaryDirectory() as other:
            reader.corpus.KERNEL = ()
            status = reader.run_once(Path(other))
            self.assertEqual(status["reading"]["status"], "no_book")

    def test_status_renders_without_error(self):
        reader.run_once(self.runtime)
        text = reader.render_status(self.runtime)
        self.assertIn("Noise 日本語読書", text)


if __name__ == "__main__":
    unittest.main()
