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
        self._orig_wm = reader.word_meaning.learn_and_evaluate
        titles = tuple(b.title for b in self._books)
        reader.corpus.kernel_titles = lambda *a, **k: titles
        reader.corpus.fetch = fake_fetch
        reader.corpus.AOZORA_AUTHORS = {}
        reader.corpus.aozora_author_works = lambda *a, **k: []
        reader.corpus.fetch_aozora = lambda *a, **k: None
        reader.corpus.tatoeba_readers = lambda *a, **k: []       # offline
        reader.word_meaning.learn_and_evaluate = lambda *a, **k: {"status": "measured", "vocab": 0}

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(reader.corpus, k, v)
        reader.word_meaning.learn_and_evaluate = self._orig_wm
        self._tmp.cleanup()

    def test_first_cycle_fetches_books_and_reads_one(self):
        status = reader.run_once(self.runtime)
        self.assertGreater(status["books_fetched"], 0)
        self.assertIn(status["reading"]["status"],
                      ("reread", "graduated", "shelved_no_progress"))
        self.assertTrue((self.runtime / reader.CURRICULUM_FILE).exists())
        self.assertTrue((self.runtime / reader.EVENTS_FILE).exists())
        self.assertTrue((self.runtime / reader.PROPOSITIONS_FILE).exists())
        self.assertIn("propositions", status)
        self.assertEqual(status["propositions"]["read_books"], 1)

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

    def test_narrative_rnn_corpus_excludes_tatoeba_and_unrecognised_books(self):
        # P1-1: the dedicated narrative model trains ONLY on the raw text of books
        # whose url is in `narrative_urls` (= all_stories: read, >=3 events, not
        # tatoeba).  Tatoeba bundles and vocab-fuel never reach it -- that is what
        # let the general RNN's corpus diverge from the retell position baseline.
        reader.run_once(self.runtime)
        cur = reader._read(self.runtime / reader.CURRICULUM_FILE)
        ids = list(cur["shelf"])
        narr_id, tat_id, stuck_id = ids[0], ids[1], ids[2]
        cur["shelf"][narr_id].update(status="graduated", times_read=3,
                                     text="いぬがかわでさかなをみつけた。いぬはさかなをたべた。ねこがないた。" * 4,
                                     source="ja")
        cur["shelf"][tat_id].update(status="graduated", times_read=3,
                                    text="いぬが はしる。ねこが ねる。" * 8, source="tatoeba")
        cur["shelf"][stuck_id].update(status="shelved_stuck", times_read=6,
                                      text="ぜんぶかなでよめないはなし。" * 12, source="ja")
        narrative_urls = {cur["shelf"][narr_id]["url"]}
        corpus = reader._narrative_rnn_corpus(cur, narrative_urls, set())
        self.assertIn(cur["shelf"][narr_id]["url"], corpus)
        self.assertNotIn(cur["shelf"][tat_id]["url"], corpus)      # tatoeba excluded
        self.assertNotIn(cur["shelf"][stuck_id]["url"], corpus)    # not a recognised narrative
        # the GENERAL corpus still takes all three (raw characters)
        gen = reader._rnn_corpus(cur, set())
        self.assertIn(cur["shelf"][tat_id]["url"], gen)
        self.assertIn(cur["shelf"][stuck_id]["url"], gen)
        # a forbidden collection is dropped from the narrative corpus too
        forbidden = {reader.jb.collection(cur["shelf"][narr_id]["url"])}
        self.assertNotIn(cur["shelf"][narr_id]["url"],
                         reader._narrative_rnn_corpus(cur, narrative_urls, forbidden))

    def test_run_once_writes_a_separate_narrative_sequence_state_and_status(self):
        # P1-1: the two models live in different files; the general model's file
        # is never repurposed, and status carries both.
        st = reader.run_once(self.runtime)
        self.assertTrue((self.runtime / reader.SEQUENCE_FILE).exists())
        self.assertTrue((self.runtime / reader.NARRATIVE_SEQUENCE_FILE).exists())
        gen = reader._read(self.runtime / reader.SEQUENCE_FILE)
        narr = reader._read(self.runtime / reader.NARRATIVE_SEQUENCE_FILE)
        self.assertEqual(gen.get("training_regime"), reader.sequence.TRAINING_REGIME)
        self.assertEqual(narr.get("training_regime"), reader.sequence.NARRATIVE_REGIME)
        self.assertIn("narrative_sequence", st)
        self.assertIn("narrative_sequence_training", st)
        # the retelling benchmark is not stuck on a permanent corpus mismatch:
        # with this tiny corpus it is simply "insufficient", never invalid
        self.assertNotIn(st["retelling"].get("status"),
                         ("measurement_invalid", "measurement_invalid_baseline_corpus_mismatch"))

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

    def test_vocab_fuel_buffer_refreshes_from_tatoeba_pools_and_never_touches_the_shelf(self):
        calls = []
        counter = [0]

        def fake_readers(level, n_readers=4, skip=0, **k):
            calls.append((level, skip))
            counter[0] += 1
            tag = counter[0]
            return [FakeText(f"drill{tag}_{i}", f"tatoeba://reader/{level}/{tag}_{i}",
                             f"犬が道を走る{tag}。子供が水を飲む{tag}。") for i in range(n_readers)]

        reader.corpus.tatoeba_readers = fake_readers
        cur = {"cycle": 500, "level": 6.0, "shelf": {"x": {"status": "in_rotation"}},
               "known_words": {}}
        n = reader._refresh_vocab_fuel(cur, 500)
        self.assertEqual(n, reader.VOCAB_FUEL_FRESH)            # fuelled despite level 6.0
        self.assertEqual(len(cur["_vocab_fuel_texts"]), reader.VOCAB_FUEL_FRESH)
        self.assertEqual(cur["shelf"], {"x": {"status": "in_rotation"}})   # shelf untouched
        self.assertTrue(any(v > 0 for v in cur["_vocab_fuel_cursors"].values()))
        self.assertEqual(reader._refresh_vocab_fuel(cur, 503), 0)          # cooldown
        # a later top-up brings NEW texts and trims to the buffer cap
        cur["_vocab_fuel_cycle"] = -999
        for c in range(520, 520 + 40 * reader.VOCAB_FUEL_COOLDOWN, reader.VOCAB_FUEL_COOLDOWN):
            reader._refresh_vocab_fuel(cur, c)
        self.assertLessEqual(len(cur["_vocab_fuel_texts"]), reader.VOCAB_FUEL_BUFFER)
        self.assertTrue(all(lvl in [f"{x:.1f}" for x in reader.VOCAB_FUEL_LEVELS]
                            for lvl in cur["_vocab_fuel_cursors"]))

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

    def test_dialogue_feedback_is_merged_into_the_prediction_feedback_channel(self):
        # last cycle's japanese_dialogue_v1 self-correction signal (a word
        # Noise kept failing to be understood about) must reach word_meaning
        # through the SAME `prediction_feedback` parameter japanese_prediction_v1
        # already feeds -- one revisable-belief channel, two first-person sources.
        reader._write(self.runtime / reader.JA_DIALOGUE_FILE, {
            "version": reader.ja_dialogue.VERSION,
            "dialogue_feedback": {"きつね": {"against": "道具", "strength": 0.2,
                                            "hits": 1, "misses": 6, "cycle": 3}}})
        reader._write(self.runtime / reader.PREDICTION_FILE, {
            "prediction_feedback": {"からす": {"against": "場所", "strength": 0.1,
                                             "hits": 2, "misses": 5, "cycle": 3}}})
        captured = {}

        def fake_learn(*a, **k):
            captured.update(k)
            return {"status": "measured", "vocab": 0}
        reader.word_meaning.learn_and_evaluate = fake_learn
        reader.run_once(self.runtime)
        fb = captured.get("prediction_feedback", {})
        self.assertIn("きつね", fb)                 # from dialogue
        self.assertIn("からす", fb)                 # from prediction
        self.assertEqual(fb["きつね"]["against"], "道具")

    def test_higher_strength_feedback_wins_on_a_shared_word(self):
        reader._write(self.runtime / reader.JA_DIALOGUE_FILE, {
            "version": reader.ja_dialogue.VERSION,
            "dialogue_feedback": {"きつね": {"against": "道具", "strength": 0.25,
                                            "hits": 1, "misses": 8, "cycle": 3}}})
        reader._write(self.runtime / reader.PREDICTION_FILE, {
            "prediction_feedback": {"きつね": {"against": "場所", "strength": 0.05,
                                             "hits": 4, "misses": 5, "cycle": 3}}})
        captured = {}

        def fake_learn(*a, **k):
            captured.update(k)
            return {"status": "measured", "vocab": 0}
        reader.word_meaning.learn_and_evaluate = fake_learn
        reader.run_once(self.runtime)
        self.assertEqual(captured["prediction_feedback"]["きつね"]["against"], "道具")


if __name__ == "__main__":
    unittest.main()
