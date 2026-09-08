import json
import re
import unittest

import reading_curriculum_v1 as rc


def book(title, url, level_text, events=8, verbs=None):
    return {"title": title, "url": url, "source": "test",
            "text": level_text, "event_count": events, "verbs": verbs or []}


SIMPLE = "きつねがぶどうを見つけました。きつねはとびあがりました。きつねはすっぱいと言いました。"
MID = ("むかしむかし、おじいさんは山へしばかりに行きました。おばあさんは川へせんたくに行きました。"
       "おばあさんは大きな桃を家へ持って帰りました。二人は桃を切りました。"
       "桃の中から元気な男の子が生まれました。男の子は鬼を島でやっつけました。")
HARDER = ("これは私が小さいときに村の茂平というおじいさんからきいたお話です。"
          "むかしは私たちの村のちかくの中山というところに小さなお城があって、"
          "中山さまというおとのさまがおられたそうです。")

GOOD_EVENTS = [{"subject": "きつね", "verb": v, "obj": "", "confidence": 0.9,
                "provenance": "heuristic_self"}
               for v in ("見つける", "とびあがる", "言う", "帰る")]


def _self(events):
    """Stamp a list of test events as Noise's own heuristic parse."""
    return [{**e, "provenance": "heuristic_self"} for e in events]


class ReadingCurriculumTest(unittest.TestCase):
    def test_difficulty_ranks_a_retelling_below_literary_prose(self):
        easy = rc.text_difficulty(SIMPLE, 3, set())
        hard = rc.text_difficulty(HARDER, 3, set())
        self.assertLess(easy["estimated_level"], hard["estimated_level"])
        self.assertLessEqual(easy["estimated_level"], 2.2)

    def test_cold_start_always_leaves_one_book_in_rotation(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", HARDER), book("B", "b", SIMPLE)], cycle=1)
        self.assertEqual(rc.summary(cur)["in_rotation"] + rc.summary(cur)["graduated"] >= 1, True)
        self.assertIsNotNone(rc.select_next_book(cur))

    def test_a_passing_comprehension_score_graduates_the_book(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE)], cycle=1)
        bid = rc.select_next_book(cur)
        out = rc.record_reading(cur, bid, GOOD_EVENTS, cycle=2, comprehension=0.9)
        self.assertEqual(out["status"], "graduated")
        self.assertEqual(cur["shelf"][bid]["status"], "graduated")

    def test_record_reading_is_fail_closed_on_provenance(self):
        # only events EXPLICITLY stamped heuristic_self become vocabulary / schema:
        # a teacher event AND an unstamped (unknown-provenance) event are both dropped
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE)], cycle=1)
        bid = rc.select_next_book(cur)
        mixed = ([{"subject": "きつね", "verb": v, "obj": "ぶどう",
                   "provenance": "heuristic_self"} for v in ("みつける", "とる", "たべる")]
                 + [{"subject": "ようせい", "verb": "たすける", "obj": "きつね",
                     "provenance": "morphological_teacher"},
                    {"subject": "だれか", "verb": "あるく", "obj": ""}])   # NO provenance
        rc.record_reading(cur, bid, mixed, cycle=2, comprehension=0.5)
        self.assertIn("きつね", cur["known_words"])
        self.assertIn("ぶどう", cur["known_words"])
        self.assertNotIn("ようせい", cur["known_words"])     # teacher token
        self.assertNotIn("たすける", cur["known_words"])
        self.assertNotIn("だれか", cur["known_words"])        # unstamped -> not promoted
        self.assertNotIn("あるく", cur["known_words"])
        self.assertEqual(set(cur["shelf"][bid].get("schema", [])), {"みつける", "とる", "たべる"})

    def test_all_unstamped_events_do_not_graduate_a_book(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE)], cycle=1)
        bid = rc.select_next_book(cur)
        unstamped = [{"subject": "きつね", "verb": v, "obj": ""} for v in ("みつける", "とる", "たべる", "なく")]
        out = rc.record_reading(cur, bid, unstamped, cycle=2, comprehension=0.95)
        self.assertEqual(out["status"], "unparsable")           # < 3 heuristic_self events
        self.assertNotEqual(cur["shelf"][bid]["status"], "graduated")

    def test_cold_start_proxy_rewards_a_coherent_parse(self):
        clean = [{"subject": "きつね", "verb": v, "obj": "", "confidence": 0.9}
                 for v in ("みつける", "とる", "たべる", "かえる", "なく")]
        self.assertGreaterEqual(rc._self_consistency(clean), rc.GRADUATE_COMPREHENSION)

    def test_cold_start_proxy_does_not_graduate_a_broken_parse(self):
        # every clause "それが<壊れた動詞>" -- high protagonist share, but not Japanese
        broken = [{"subject": "それ", "verb": v, "obj": "", "confidence": 0.9}
                  for v in ("うめる", "まもなく", "しぬ", "もでてく", "でる")]
        self.assertLess(rc._self_consistency(broken), rc.GRADUATE_COMPREHENSION)

    def test_stale_parse_books_return_to_rotation_on_a_parser_version_bump(self):
        import japanese_event_v1 as jevent
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE)], cycle=1)
        bid = rc._book_id("a", "A")
        cur["shelf"][bid].update(status="shelved_stuck", times_read=6,
                                 comprehension_history=[0.2, 0.2, 0.2],
                                 parser_version=jevent.PARSER_VERSION - 1)
        reset = rc.reevaluate_stale_parses(cur)
        self.assertEqual(reset, [bid])
        self.assertEqual(cur["shelf"][bid]["status"], "in_rotation")
        self.assertEqual(cur["shelf"][bid]["times_read"], 0)
        self.assertEqual(cur["shelf"][bid]["comprehension_history"], [0.2, 0.2, 0.2])
        # current-version books are left alone; no repeat reset
        self.assertEqual(rc.reevaluate_stale_parses(cur), [])

    def test_parser_bump_resets_an_inflated_level_and_re_walks_the_corpus(self):
        import japanese_event_v1 as jevent
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("easy", "e", SIMPLE), book("mid", "m", MID),
                                book("hard", "h", HARDER)], cycle=1)
        cur["level"] = 4.0                       # inflated by the old parser
        cur["level_parser_version"] = jevent.PARSER_VERSION - 1
        for b in cur["shelf"].values():
            b["status"] = "shelved_above_level"
        result = rc.reset_level_for_new_parser(cur, cycle=9)
        self.assertTrue(result["reset"])
        self.assertLess(cur["level"], 4.0)
        self.assertGreaterEqual(sum(b["status"] == "in_rotation"
                                    for b in cur["shelf"].values()), 1)
        self.assertEqual(rc.reset_level_for_new_parser(cur, cycle=10)["reset"], False)

    def test_rereading_the_same_book_does_not_inflate_word_book_counts(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE)], cycle=1)
        bid = rc._book_id("a", "A")
        for c in range(2, 6):
            rc.record_reading(cur, bid, GOOD_EVENTS, cycle=c, comprehension=0.5)
        # "きつね" was seen in exactly ONE book, however many times it was re-read
        self.assertEqual(cur["known_words"]["きつね"]["books"], 1)
        rc.register_books(cur, [book("B", "b", SIMPLE)], cycle=10)
        rc.record_reading(cur, rc._book_id("b", "B"), GOOD_EVENTS, cycle=11, comprehension=0.5)
        self.assertEqual(cur["known_words"]["きつね"]["books"], 2)

    def test_difficulty_coverage_uses_event_tokens_not_glued_runs(self):
        known = {"きつね", "ぶどう"}
        events = [{"subject": "きつね", "obj": "ぶどう", "verb": "みつける"}]
        d = rc.text_difficulty("きつねがぶどうを見つけました。", 1, known, events=events)
        self.assertGreater(d["known_word_coverage"], 0.5)     # 2/3 known
        # without events the whole run is one unknown "word" -> 0
        self.assertEqual(rc.text_difficulty("きつねがぶどうを見つけました。", 1, known)
                         ["known_word_coverage"], 0.0)

    def test_schema_migration_is_idempotent_and_audits_aided_graduates(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE), book("B", "b", MID),
                                book("S", "s", "むかし。")], cycle=1)
        a, b, s = rc._book_id("a", "A"), rc._book_id("b", "B"), rc._book_id("s", "S")
        cur["shelf"][a].update(status="graduated", comprehension_history=[0.8])
        cur["shelf"][s].update(status="graduated", scaffolded=True,
                               comprehension_history=[0.9])
        # legacy inflated known_words -- one real, one that no re-parse recovers
        cur["known_words"] = {"きつね": {"books": 47, "first_cycle": 1},
                              "ぶどう": {"books": 12, "used": True, "use_tested": True},
                              "ようせい": {"books": 8, "first_cycle": 2}}
        store = {}
        events = {a: _self([{"subject": "きつね", "verb": "みつける", "obj": "ぶどう"}] * 4),
                  b: _self([{"subject": "おじいさん", "verb": "行く", "obj": ""}] * 4)}

        def fake_extract(text):
            if "きつね" in text or text == SIMPLE:
                return events[a]
            if text == MID:
                return events[b]
            return []                                  # "むかし。" -> unparsable

        m1 = rc.migrate_reading_state(cur, store, fake_extract)
        self.assertTrue(m1["migrated"])
        self.assertEqual(m1["from_schema"], 0)
        # A (graduated) and S (graduated) count as read; B was only registered
        self.assertEqual(m1["books_read"], 2)
        self.assertEqual(m1["books_unread"], 1)
        self.assertNotIn(b, store)                     # unread book: not in the shared store
        # a scaffold-graduated book is returned for re-evaluation, history kept
        self.assertEqual(cur["shelf"][s]["status"], "in_rotation")
        self.assertTrue(cur["shelf"][s]["graduated_via_aid"])
        self.assertEqual(cur["shelf"][s]["comprehension_history"], [0.9])
        # a heuristically-graduated book stays graduated
        self.assertEqual(cur["shelf"][a]["status"], "graduated")
        # known_words rebuilt: distinct-book counts, tiers preserved
        self.assertEqual(cur["known_words"]["きつね"]["books"], 1)
        self.assertEqual(cur["known_words"]["きつね"]["book_ids"], [a])
        self.assertIn("ぶどう", cur["known_words"])
        self.assertTrue(cur["known_words"]["ぶどう"].get("used"))
        # a legacy word no re-parse recovers is quarantined, not trusted at its
        # inflated count
        self.assertEqual(cur["known_words"]["ようせい"]["books"], 0)
        self.assertTrue(cur["known_words"]["ようせい"]["books_unverified"])
        self.assertEqual(m1["known_words_quarantined"], 1)
        # the re-parse never MINTS vocabulary: no key that was not already known
        self.assertEqual(set(cur["known_words"]), {"きつね", "ぶどう", "ようせい"})
        self.assertNotIn("みつける", cur["known_words"])
        # a second run changes nothing
        snapshot = json.loads(json.dumps(cur))
        m2 = rc.migrate_reading_state(cur, store, fake_extract)
        self.assertFalse(m2["migrated"])
        self.assertEqual(json.loads(json.dumps(cur)), snapshot)

    def test_v3_migration_drops_unread_books_as_vocabulary_evidence(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("READ", "r", SIMPLE), book("UNREAD", "u", MID)], cycle=1)
        rid, uid = rc._book_id("r", "READ"), rc._book_id("u", "UNREAD")
        cur["shelf"][rid].update(times_read=1, last_read_cycle=2, comprehension_history=[0.7])
        # UNREAD stays as-registered: no times_read, no history, no post-read status
        cur["known_words"] = {
            "きつね": {"books": 2, "book_ids": [rid, uid], "first_cycle": 1},  # some unread support
            "ようせい": {"books": 1, "book_ids": [uid], "first_cycle": 1},      # ONLY unread support
        }
        store = {rid: _self([{"subject": "きつね", "verb": "みつける", "obj": ""}]),
                 uid: _self([{"subject": "ようせい", "verb": "でる", "obj": ""}])}

        def fake_extract(text):
            if text == SIMPLE:
                return _self([{"subject": "きつね", "verb": "みつける", "obj": ""}] * 4)
            return _self([{"subject": "きつね", "verb": "にげる", "obj": ""},
                          {"subject": "ようせい", "verb": "あらわれる", "obj": ""}] * 2)

        self.assertTrue(rc.book_was_read(cur["shelf"][rid]))
        self.assertFalse(rc.book_was_read(cur["shelf"][uid]))
        m = rc.migrate_reading_state(cur, store, fake_extract)
        self.assertTrue(m["migrated"])
        self.assertEqual(m["books_read"], 1)
        self.assertEqual(m["books_unread"], 1)
        self.assertNotIn(uid, store)                       # unread book left the shared store
        self.assertIn(rid, store)
        # きつね keeps ONLY the read book
        self.assertEqual(cur["known_words"]["きつね"]["book_ids"], [rid])
        self.assertEqual(cur["known_words"]["きつね"]["books"], 1)
        # ようせい loses all support -> quarantined, not deleted, provenance kept
        self.assertEqual(cur["known_words"]["ようせい"]["books"], 0)
        self.assertEqual(cur["known_words"]["ようせい"]["book_ids"], [])
        self.assertTrue(cur["known_words"]["ようせい"]["books_unverified"])
        self.assertGreaterEqual(m["unread_book_ids_removed"], 2)   # きつね:uid + ようせい:uid
        self.assertGreaterEqual(m["known_words_zeroed_by_v3"], 1)
        self.assertEqual(set(cur["known_words"]), {"きつね", "ようせい"})   # nothing minted
        # a fresh shelf book alone does not change known_words on the next migrate
        snap = json.loads(json.dumps(cur["known_words"]))
        rc.register_books(cur, [book("NEW", "n", SIMPLE)], cycle=3)
        rc.migrate_reading_state(cur, store, fake_extract)         # already v3 -> no-op
        self.assertEqual(cur["known_words"], snap)

    def test_books_far_above_the_level_are_not_dragged_in_the_loop_idles(self):
        cur = rc.empty_curriculum()
        cur["level"] = 2.0
        rc.register_books(cur, [book("Easy", "e", SIMPLE)], cycle=1)
        easy_id = next(iter(cur["shelf"]))
        cur["shelf"][easy_id]["status"] = "graduated"          # already read
        rc.register_books(cur, [book("Hard1", "h1", HARDER), book("Hard2", "h2", HARDER)], cycle=2)
        for b in cur["shelf"].values():
            if b["title"].startswith("Hard"):
                self.assertEqual(b["status"], "shelved_above_level")
        # nothing within reach -> idle honestly rather than fake-read a novel
        self.assertIsNone(rc.select_next_book(cur))
        self.assertEqual(rc.summary(cur)["readable_unread"], 0)

    def test_parser_bump_files_stuck_books_by_level_not_all_into_rotation(self):
        cur = rc.empty_curriculum()
        cur["level"] = 2.0
        rc.register_books(cur, [book("Easy", "e", SIMPLE), book("Hard", "h", HARDER)], cycle=1)
        for b in cur["shelf"].values():
            b.update(status="shelved_stuck", times_read=6, parser_version=1)
            b["estimated_level"] = 2.0 if b["title"] == "Easy" else 4.0
        rc.reevaluate_stale_parses(cur)
        by_title = {b["title"]: b["status"] for b in cur["shelf"].values()}
        self.assertEqual(by_title["Easy"], "in_rotation")
        self.assertEqual(by_title["Hard"], "shelved_above_level")

    def test_a_stuck_book_is_shelved_but_retried_as_a_last_resort(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("A", "a", SIMPLE), book("B", "b", SIMPLE)], cycle=1)
        stuck = rc._book_id("a", "A")
        for c in range(2, 2 + rc.MAX_REREADS + 1):
            rc.record_reading(cur, stuck, GOOD_EVENTS, cycle=c, comprehension=0.5)
        self.assertEqual(cur["shelf"][stuck]["status"], "shelved_stuck")
        # while B is in rotation the stuck book is left alone
        self.assertEqual(rc.select_next_book(cur), rc._book_id("b", "B"))
        for c in range(20, 20 + rc.MAX_REREADS + 1):
            rc.record_reading(cur, rc._book_id("b", "B"), GOOD_EVENTS, cycle=c, comprehension=0.5)
        # now nothing else is readable -> retry the stuck book rather than idle
        self.assertIsNotNone(rc.select_next_book(cur))
        self.assertEqual(cur["shelf"][stuck]["times_read"], 0)

    def test_an_earned_level_is_never_surrendered_when_its_band_gets_stuck(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("easy", "e", SIMPLE)], cycle=1)
        bid = rc._book_id("e", "easy")
        rc.record_reading(cur, bid, GOOD_EVENTS, cycle=2, comprehension=0.9)   # graduate
        rc.maybe_advance_level(cur, cycle=3)
        earned = cur["level"]
        self.assertGreater(earned, 1.5)
        self.assertEqual(cur["floor_level"], earned)
        # a much easier book appears and the band's books are all stuck
        rc.register_books(cur, [book("tiny", "t", "ねこがきた。")], cycle=4)
        for b in cur["shelf"].values():
            if b["status"] != "graduated":
                b["status"] = "shelved_stuck"
        rc.register_books(cur, [book("tiny2", "t2", "いぬがきた。")], cycle=5)
        self.assertGreaterEqual(cur["level"], earned)          # not dropped below floor

    def test_level_advances_when_the_band_is_exhausted_and_unshelves_reachable_books(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("easy", "e", SIMPLE), book("mid", "m", MID)], cycle=1)
        start = cur["level"]
        mid = cur["shelf"][rc._book_id("m", "mid")]
        self.assertEqual(mid["status"], "shelved_above_level")
        mid["shelved_at_level"] = start + rc.LEVEL_STEP       # just out of reach now
        bid = rc.select_next_book(cur)
        rc.record_reading(cur, bid, GOOD_EVENTS, cycle=2, comprehension=0.9)
        result = rc.maybe_advance_level(cur, cycle=3)
        self.assertTrue(result["advanced"])
        self.assertGreater(cur["level"], start)
        self.assertEqual(result["unshelved"], 1)              # "mid" now reachable
        self.assertEqual(mid["status"], "in_rotation")

    def test_advancement_pauses_if_current_band_comprehension_drops(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("g", "g", SIMPLE), book("r", "r", SIMPLE)], cycle=1)
        # graduate one, but the other is in rotation with a poor recent score
        rc.record_reading(cur, rc._book_id("g", "g"), GOOD_EVENTS, cycle=2, comprehension=0.9)
        rc.record_reading(cur, rc._book_id("r", "r"), GOOD_EVENTS, cycle=3, comprehension=0.4)
        self.assertFalse(rc.maybe_advance_level(cur, cycle=4)["advanced"])

    def test_vocabulary_has_three_tiers_and_used_gates_known_once_tested(self):
        cur = rc.empty_curriculum()
        for c in range(1, 4):
            rc.record_reading(cur, "x", [], cycle=c)          # unknown book, no-op events
        cur["known_words"]["きつね"] = {"books": 3}
        cur["known_words"]["ぶどう"] = {"books": 3}
        self.assertEqual(rc.summary(cur)["vocabulary_seen"], 2)
        self.assertEqual(len(rc._known_set(cur)), 2)          # seen tier while untested
        rc.record_word_test(cur, "きつね", used=True, cycle=5)
        rc.record_word_test(cur, "ぶどう", used=False, cycle=5)
        self.assertEqual(rc.summary(cur)["vocabulary_used"], 1)
        self.assertEqual(rc._known_set(cur), {"きつね"})       # used tier once any word tested
        rc.record_word_test(cur, "きつね", explained=True, cycle=9)
        self.assertEqual(rc.summary(cur)["vocabulary_explained"], 1)

    def test_retention_check_returns_a_lower_level_graduate_after_the_interval(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [book("low", "l", SIMPLE)], cycle=1)
        bid = rc.select_next_book(cur)
        rc.record_reading(cur, bid, GOOD_EVENTS, cycle=2, comprehension=0.9)
        cur["level"] = 3.0                                    # reader has moved on
        self.assertIsNone(rc.retention_check_due(cur, cycle=10))
        self.assertEqual(rc.retention_check_due(cur, cycle=2 + rc.RETENTION_INTERVAL), bid)

    def test_reading_age_is_monotonic_in_level(self):
        self.assertLess(rc.reading_age(1.5), rc.reading_age(4.0))

    def test_level_scale_runs_from_picture_book_to_middle_school(self):
        # the endpoint goal is middle-school reading, not toddler books
        self.assertEqual(rc.current_milestone(1.5), "絵本 (4歳・第一目標)")
        self.assertEqual(rc.current_milestone(7.0), "中学生")
        self.assertIn("最終目標", rc.current_milestone(9.5))
        self.assertGreaterEqual(rc.MAX_LEVEL, 9.0)
        # ~15 years old at the top of the scale
        self.assertGreaterEqual(int(re.search(r"\d+", rc.reading_age(9.0)).group()), 14)

    def test_explained_tier_only_gates_advancement_at_higher_levels(self):
        self.assertEqual(rc.explained_ratio_target(2.0), 0.0)   # a 4-year-old defines nothing
        self.assertEqual(rc.explained_ratio_target(3.5), 0.0)
        self.assertGreater(rc.explained_ratio_target(7.0), 0.3)  # middle-school: explain most words

    def test_advancement_at_a_high_level_requires_the_explained_tier(self):
        cur = rc.empty_curriculum()
        cur["level"] = 7.0
        # a graduated band book with high comprehension, but recent vocabulary
        # is `used` without being `explained`
        bid = rc._book_id("b", "b")
        cur["shelf"][bid] = {"title": "b", "url": "b", "source": "t",
                             "difficulty": {}, "estimated_level": 7.0, "times_read": 1,
                             "comprehension_history": [0.9], "status": "graduated",
                             "shelved_at_level": None, "first_seen_cycle": 1,
                             "last_read_cycle": 2, "graduated_cycle": 2}
        for i in range(10):
            rc.record_word_test(cur, f"語{i}", used=True, cycle=2)
        blocked = rc.maybe_advance_level(cur, cycle=3)
        self.assertFalse(blocked["advanced"])
        self.assertIn("explained", blocked["reason"])
        for i in range(6):
            rc.record_word_test(cur, f"語{i}", explained=True, cycle=3)
        self.assertTrue(rc.maybe_advance_level(cur, cycle=4)["advanced"])

    def test_select_prefers_a_book_whose_schema_the_reader_already_knows(self):
        cur = rc.empty_curriculum()
        rc.register_books(cur, [
            book("known-shape", "k", SIMPLE, verbs=["みつける", "たべる", "かえる"]),
            book("new-shape", "n", SIMPLE, verbs=["さがす", "つくる", "おどる"]),
        ], cycle=1)
        # graduate a book that shares its verbs with "known-shape"
        gid = rc._book_id("g", "g")
        cur["shelf"][gid] = {"title": "g", "url": "g", "source": "t", "difficulty": {},
                             "estimated_level": 1.5, "schema": ["みつける", "たべる"],
                             "times_read": 1, "comprehension_history": [0.9],
                             "status": "graduated", "shelved_at_level": None,
                             "first_seen_cycle": 0, "last_read_cycle": 1}
        self.assertEqual(cur["shelf"][rc.select_next_book(cur)]["title"], "known-shape")

    def test_schema_signature_is_the_distinct_verbs(self):
        self.assertEqual(rc._schema_signature(["a", "b", "a", "", "c"]), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
