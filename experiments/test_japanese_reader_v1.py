import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

# keep these tests analyser-independent; the teacher path has its own suite
os.environ.setdefault("AI_NOISE_NO_MORPHOLOGY", "1")

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
            "kernel_titles": reader.corpus.kernel_titles,
            "fetch": reader.corpus.fetch,
            "AOZORA_AUTHORS": reader.corpus.AOZORA_AUTHORS,
            "aozora_author_works": reader.corpus.aozora_author_works,
            "fetch_aozora": reader.corpus.fetch_aozora,
            "tatoeba_readers": reader.corpus.tatoeba_readers,
        }
        titles = tuple(b.title for b in self._books)
        reader.corpus.kernel_titles = lambda *a, **k: titles
        reader.corpus.fetch = fake_fetch
        reader.corpus.AOZORA_AUTHORS = {}
        reader.corpus.aozora_author_works = lambda *a, **k: []
        reader.corpus.fetch_aozora = lambda *a, **k: None
        reader.corpus.tatoeba_readers = lambda *a, **k: []       # offline

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
            reader.corpus.kernel_titles = lambda *a, **k: ()
            status = reader.run_once(Path(other))
            self.assertEqual(status["reading"]["status"], "no_book")

    def test_status_renders_without_error(self):
        reader.run_once(self.runtime)
        text = reader.render_status(self.runtime)
        self.assertIn("Noise 日本語読書", text)

    def test_a_parser_version_bump_re_reads_stuck_books(self):
        import japanese_event_v1 as jevent
        reader.run_once(self.runtime)
        cur = reader._read(self.runtime / reader.CURRICULUM_FILE)
        events = reader._read(self.runtime / reader.EVENTS_FILE)
        stuck = next(bid for bid in cur["shelf"])
        cur["shelf"][stuck].update(status="shelved_stuck", times_read=6,
                                   parser_version=jevent.PARSER_VERSION - 1)
        events[stuck] = [{"subject": "x", "verb": "stale", "obj": ""}]   # stale cache
        reader._write(self.runtime / reader.CURRICULUM_FILE, cur)
        reader._write(self.runtime / reader.EVENTS_FILE, events)
        reader.run_once(self.runtime)
        cur = reader._read(self.runtime / reader.CURRICULUM_FILE)
        self.assertNotEqual(cur["shelf"][stuck]["status"], "shelved_stuck")
        events = reader._read(self.runtime / reader.EVENTS_FILE)
        self.assertNotEqual(events.get(stuck), [{"subject": "x", "verb": "stale", "obj": ""}])

    def test_rnn_corpus_uses_raw_text_of_read_books_including_shelved_stuck(self):
        reader.run_once(self.runtime)
        cur = reader._read(self.runtime / reader.CURRICULUM_FILE)
        # a read book the parser could not break down: raw text, no events
        stuck = next(bid for bid in cur["shelf"])
        cur["shelf"][stuck].update(status="shelved_stuck", times_read=6,
                                   text="ぜんぶかなでかかれたはなし。" * 20)
        corpus = reader._rnn_corpus(cur, set())
        self.assertIn(cur["shelf"][stuck]["url"], corpus)
        self.assertIn("ぜんぶかなでかかれたはなし", corpus[cur["shelf"][stuck]["url"]])
        # a forbidden collection is still excluded
        forbidden = {reader.jb.collection(cur["shelf"][stuck]["url"])}
        self.assertNotIn(cur["shelf"][stuck]["url"], reader._rnn_corpus(cur, forbidden))
        # an unread (only fetched) book contributes nothing
        for bid, b in cur["shelf"].items():
            if not reader.curriculum.book_was_read(b):
                self.assertNotIn(b["url"], reader._rnn_corpus(cur, set()))
                break

    def test_fetched_but_unread_books_do_not_enter_the_events_store(self):
        # a book that was only fetched (registered on the shelf) must not appear
        # in reading-events.json until Noise actually reads it
        reader.run_once(self.runtime)
        cur = reader._read(self.runtime / reader.CURRICULUM_FILE)
        events = reader._read(self.runtime / reader.EVENTS_FILE)
        read_ids = {bid for bid, b in cur["shelf"].items()
                    if reader.curriculum.book_was_read(b)}
        self.assertTrue(read_ids)
        self.assertLess(len(events), len(cur["shelf"]))          # not every shelf book
        self.assertTrue(set(events) <= read_ids)                 # only read books

    def test_events_store_events_carry_heuristic_self_provenance(self):
        reader.run_once(self.runtime)
        events = reader._read(self.runtime / reader.EVENTS_FILE)
        flat = [e for evs in events.values() for e in evs]
        self.assertTrue(flat)
        self.assertTrue(all(e.get("provenance") == "heuristic_self" for e in flat))

    def test_write_events_is_fail_closed_on_provenance(self):
        reader.run_once(self.runtime)
        reader._events_path = self.runtime / reader.EVENTS_FILE
        good = [{"subject": "き", "verb": v, "obj": "", "provenance": "heuristic_self"}
                for v in ("みる", "とぶ", "なく")]
        reader._write_events({
            "keep": good,
            "strip_teacher": good + [{"subject": "x", "verb": "y", "provenance": "morphological_teacher"}],
            "drop_unstamped": [{"subject": "き", "verb": "みる", "obj": ""}] * 3,
        })
        back = reader._read(self.runtime / reader.EVENTS_FILE)
        self.assertIn("keep", back)
        self.assertNotIn("drop_unstamped", back)                 # no explicit stamp -> gone
        self.assertEqual(len(back["strip_teacher"]), 3)          # teacher event stripped out
        self.assertTrue(all(e["provenance"] == "heuristic_self" for e in back["strip_teacher"]))

    def test_heuristic_only_is_fail_closed(self):
        mixed = [{"verb": "a", "provenance": "heuristic_self"},
                 {"verb": "b", "provenance": "morphological_teacher"},
                 {"verb": "c"}, {"verb": "d", "provenance": "local_llm_scaffold"}]
        kept = reader._heuristic_only(mixed)
        self.assertEqual([e["verb"] for e in kept], ["a"])

    def test_caregiver_batch_does_not_crash_when_no_book_is_selectable(self):
        # regression: `model` is only bound inside `if book_id:`, but the
        # caregiver batch reads it -- no selectable book must not raise
        reader.run_once(self.runtime)                    # read + populate shelf
        cur = reader._read(self.runtime / reader.CURRICULUM_FILE)
        cur["cycle"] = 40                                # past the caregiver interval
        reader._write(self.runtime / reader.CURRICULUM_FILE, cur)
        care = self.runtime / reader.CAREGIVER_FILE
        state = reader._read(care) or {}
        state["pending"], state["last_batch_cycle"] = [], 0   # a batch is due
        reader._write(care, state)
        orig_select, orig_retention = reader.curriculum.select_next_book, reader.curriculum.retention_check_due
        reader.curriculum.select_next_book = lambda *a, **k: None
        reader.curriculum.retention_check_due = lambda *a, **k: None
        try:
            status = reader.run_once(self.runtime)       # must not raise
        finally:
            reader.curriculum.select_next_book = orig_select
            reader.curriculum.retention_check_due = orig_retention
        self.assertEqual(status["reading"]["status"], "no_book")


if __name__ == "__main__":
    unittest.main()
