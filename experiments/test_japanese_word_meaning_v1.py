import unittest

import japanese_word_meaning_v1 as wm


def _self(events):
    return [{**e, "provenance": "heuristic_self"} for e in events]


STORY = _self([
    {"subject": "きつね", "verb": "みつける", "obj": "ぶどう"},
    {"subject": "きつね", "verb": "とびあがる", "obj": ""},
    {"subject": "きつね", "verb": "あきらめる", "obj": "ぶどう"},
    {"subject": "からす", "verb": "とぶ", "obj": ""},
])


class WordMeaningTest(unittest.TestCase):
    def test_genus_regex_takes_the_head_noun_before_the_period(self):
        g = wm._GENUS.search("日本全国に生息するイヌ科に属する哺乳動物。")
        self.assertEqual(g.group(1), "哺乳動物")
        g2 = wm._GENUS.search("時刻を示す道具である。")
        self.assertEqual(g2.group(1), "道具")

    def test_bad_genus_and_stopwords_are_rejected(self):
        self.assertIn("日本語", wm._BAD_GENUS)
        self.assertFalse(wm._is_content("する"))
        self.assertFalse(wm._is_content("こと"))
        self.assertTrue(wm._is_content("きつね"))

    def test_held_out_split_is_deterministic(self):
        a = {w: wm._held_out(w) for w in ("きつね", "からす", "ぶどう", "つき", "うみ")}
        b = {w: wm._held_out(w) for w in ("きつね", "からす", "ぶどう", "つき", "うみ")}
        self.assertEqual(a, b)

    def test_observe_builds_entity_contexts_from_heuristic_events_only(self):
        state = wm._blank()
        wm._observe(state, [{"events": STORY + [
            {"subject": "たぬき", "verb": "みる", "obj": "", "provenance": "teacher"}]}])
        self.assertIn("きつね", state["contexts"])
        self.assertGreaterEqual(state["entities"].get("きつね", 0), 3)
        self.assertNotIn("たぬき", state["entities"])          # teacher-stamped, dropped
        self.assertIn("ぶどう", state["contexts"]["きつね"])    # co-occurs

    def test_explain_is_model_free_and_uses_the_taxonomy(self):
        state = wm._blank()
        wm._observe(state, [{"events": STORY}] * 3)
        state["taxonomy"] = {"からす": "鳥"}
        # きつね co-occurs with からす -> genus propagates via the neighbour
        e = wm.explain("きつね", state, allow_self=False)
        self.assertIn("ぶどう", e["assoc"])
        self.assertEqual(e["genus"], "鳥")                     # propagated (toy data)

    def test_learn_and_evaluate_reports_a_stable_shape_when_offline(self):
        # no network -> no research, but the pipeline must not crash and the
        # returned dict must round-trip as `previous`
        r1 = wm.learn_and_evaluate([{"events": STORY}] * 12, None, 1,
                                   known_words={"きつね", "からす", "ぶどう"})
        self.assertIn(r1["status"], ("measured", "insufficient_test_words"))
        r2 = wm.learn_and_evaluate([{"events": STORY}] * 12, dict(r1), 2,
                                   known_words={"きつね", "からす", "ぶどう"})
        self.assertEqual(r2["version"], 1)
        self.assertIsInstance(r2["researched"], list)

    def test_a_held_out_word_is_never_researched(self):
        state = wm._blank()
        wm._observe(state, [{"events": STORY}] * 5)
        state["selection_words"] = ["きつね"]
        # even if きつね is held out, learn_and_evaluate must not add it to researched
        r = wm.learn_and_evaluate([{"events": STORY}] * 5, state, 3,
                                  known_words={"きつね", "からす", "ぶどう"})
        for w in state["selection_words"]:
            self.assertNotIn(w, [x for x in r["researched"] if wm._held_out(x) and x == w])


if __name__ == "__main__":
    unittest.main()
