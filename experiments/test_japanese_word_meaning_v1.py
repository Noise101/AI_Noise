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


class _FakeLLM:
    """Offline stand-in for the local model: answers from a fixed table."""

    def __init__(self, table=None, up=True):
        self.table = table or {}
        self.up = up
        self.asked = []

    def available(self):
        return self.up

    def ask_class(self, word):
        self.asked.append(word)
        return self.table.get(word)


class CoarseOntologyTest(unittest.TestCase):
    def test_fine_genus_folds_onto_the_closed_set(self):
        self.assertEqual(wm._coarse("哺乳動物"), "生き物")
        self.assertEqual(wm._coarse("道具"), "道具")
        self.assertEqual(wm._coarse("登場人物"), "人")
        self.assertEqual(wm._coarse("大きな川"), "場所")
        self.assertEqual(wm._coarse("よく分からない語"), "")

    def test_person_is_compatible_with_creature(self):
        self.assertTrue(wm._compatible("人", "生き物"))
        self.assertTrue(wm._compatible("生き物", "生き物"))
        self.assertFalse(wm._compatible("道具", "生き物"))
        self.assertFalse(wm._compatible("", "生き物"))


class ReadingClassTest(unittest.TestCase):
    def test_acting_under_its_own_power_reads_as_a_creature(self):
        prof = {"subj": 3, "obj": 1,
                "subj_verbs": {"あるく": 2, "たべる": 1}, "obj_verbs": {"みつける": 1}}
        cls, strength = wm._reading_class(prof)
        self.assertEqual(cls, "生き物")
        self.assertGreater(strength, 0.0)

    def test_only_ever_handled_reads_as_a_tool(self):
        prof = {"subj": 0, "obj": 4,
                "subj_verbs": {}, "obj_verbs": {"つくる": 2, "つかう": 2}}
        self.assertEqual(wm._reading_class(prof)[0], "道具")

    def test_too_little_usage_is_uninformative(self):
        self.assertEqual(wm._reading_class({"subj": 1, "obj": 0,
                                            "subj_verbs": {"あるく": 1}, "obj_verbs": {}}),
                         ("", 0.0))

    def test_moves_and_is_handled_reads_as_a_vehicle_not_a_creature(self):
        prof = {"subj": 5, "obj": 5,
                "subj_verbs": {"着く": 3, "走る": 2},          # motion only, no mind
                "obj_verbs": {"止める": 3, "引く": 2}}
        cls, _ = wm._reading_class(prof)
        self.assertEqual(cls, "道具")

    def test_worn_object_reads_as_a_tool(self):
        prof = {"subj": 0, "obj": 6, "subj_verbs": {},
                "obj_verbs": {"着る": 4, "かける": 2}}
        self.assertEqual(wm._reading_class(prof)[0], "道具")

    def test_x_wo_suru_reads_as_an_event(self):
        prof = {"subj": 0, "obj": 5, "subj_verbs": {}, "obj_verbs": {"する": 5}}
        self.assertEqual(wm._reading_class(prof)[0], "出来事")


class ObserveTest(unittest.TestCase):
    def test_profiles_and_entities_come_from_heuristic_events_only(self):
        state = wm._blank()
        wm._observe(state, [{"events": STORY + [
            {"subject": "たぬき", "verb": "みる", "obj": "", "provenance": "teacher"}]}])
        self.assertIn("きつね", state["entities"])
        self.assertNotIn("たぬき", state["entities"])          # teacher-stamped, dropped
        prof = state["profiles"]["きつね"]
        self.assertEqual(prof["subj"], 3)
        self.assertIn("みつける", prof["subj_verbs"])
        self.assertIn("ぶどう", state["contexts"]["きつね"])


class BeliefRevisionTest(unittest.TestCase):
    def test_testimony_alone_never_makes_a_word_understood(self):
        state = wm._blank()
        state["taxonomy"] = {"つるぎ": "武器"}
        state["llm_class"] = {"つるぎ": "道具"}
        state["entities"] = {"つるぎ": 2}
        # no profile / no neighbours -> only testimony speaks
        b = wm._revise_belief(state, "つるぎ", 10)
        self.assertEqual(b["genus"], "道具")
        self.assertLessEqual(b["confidence"], wm.TESTIMONY_CAP + 1e-9)
        self.assertFalse(b["understood"])
        self.assertEqual(b["support"]["evidence"], 0.0)

    def test_reading_evidence_grounds_understanding(self):
        state = wm._blank()
        state["taxonomy"] = {"うさぎ": "哺乳動物"}
        state["profiles"] = {"うさぎ": {"subj": 4, "obj": 0,
                                        "subj_verbs": {"はしる": 2, "たべる": 2},
                                        "obj_verbs": {}}}
        state["entities"] = {"うさぎ": 4}
        b = wm._revise_belief(state, "うさぎ", 5)
        self.assertEqual(b["genus"], "生き物")
        self.assertGreaterEqual(b["confidence"], wm.UNDERSTOOD_CONF)
        self.assertTrue(b["understood"])
        self.assertGreater(b["support"]["evidence"], 0.0)

    def test_reading_evidence_overrules_an_earlier_testimony_genus(self):
        state = wm._blank()
        state["entities"] = {"こま": 3}
        # earlier belief: someone's testimony said こま is a creature
        state["beliefs"] = {"こま": {"genus": "生き物", "confidence": 0.3,
                                     "understood": False,
                                     "support": {"testimony": 0.3, "evidence": 0.0},
                                     "sources": ["local_model"], "revisions": [],
                                     "last_cycle": 1}}
        state["llm_class"] = {"こま": "生き物"}
        # now Noise has only ever seen こま made and spun -- a thing
        state["profiles"] = {"こま": {"subj": 0, "obj": 4, "subj_verbs": {},
                                      "obj_verbs": {"つくる": 2, "つかう": 2}}}
        b = wm._revise_belief(state, "こま", 9)
        self.assertEqual(b["genus"], "道具")
        self.assertTrue(b["revisions"])
        self.assertEqual(b["revisions"][-1]["from"], "生き物")
        self.assertEqual(b["revisions"][-1]["to"], "道具")

    def test_neighbour_vote_only_counts_understood_beliefs(self):
        state = wm._blank()
        state["contexts"] = {"けもの": {"いぬ": 3, "ねこ": 3}}
        state["beliefs"] = {
            "いぬ": {"genus": "生き物", "understood": True},
            "ねこ": {"genus": "生き物", "understood": False},   # not grounded -> no vote
        }
        self.assertEqual(wm._neighbour_class(state, "けもの"), ("", 0.0))
        state["beliefs"]["ねこ"]["understood"] = True
        cls, w = wm._neighbour_class(state, "けもの")
        self.assertEqual(cls, "生き物")
        self.assertGreater(w, 0.0)


class ExplainTest(unittest.TestCase):
    def test_held_out_explanation_ignores_the_words_own_testimony(self):
        state = wm._blank()
        state["taxonomy"] = {"かえる": "両生類"}         # testimony that must be ignored
        state["profiles"] = {"かえる": {"subj": 3, "obj": 0,
                                        "subj_verbs": {"とぶ": 2, "なく": 1}, "obj_verbs": {}}}
        state["contexts"] = {"かえる": {"いけ": 2}}
        e = wm.explain("かえる", state, allow_self=False)
        self.assertEqual(e["genus"], "生き物")            # from reading usage, not 両生類
        self.assertIn("いけ", e["assoc"])


class LearnAndEvaluateTest(unittest.TestCase):
    def test_stable_shape_offline_and_round_trips(self):
        llm = _FakeLLM(up=False)
        r1 = wm.learn_and_evaluate([{"events": STORY}] * 12, None, 1,
                                   known_words={"きつね", "からす", "ぶどう"}, llm=llm)
        self.assertIn(r1["status"], ("measured", "insufficient_test_words"))
        self.assertEqual(r1["version"], 2)
        self.assertEqual(r1["llm_status"], "skipped")
        r2 = wm.learn_and_evaluate([{"events": STORY}] * 12, dict(r1), 2,
                                   known_words={"きつね", "からす", "ぶどう"}, llm=llm)
        self.assertEqual(r2["version"], 2)
        self.assertIsInstance(r2["researched"], list)
        self.assertIsInstance(r2["llm_asked"], list)
        self.assertIn("beliefs", r2)

    def test_v1_state_migrates_without_losing_the_taxonomy(self):
        v1 = {"version": 1, "contexts": {"きつね": {"ぶどう": 3}},
              "entities": {"きつね": 4}, "taxonomy": {"きつね": "哺乳動物"},
              "researched": ["きつね"], "selection_words": [], "selection_refs": {},
              "learning_curve": []}
        r = wm.learn_and_evaluate([{"events": STORY}] * 6, v1, 3,
                                  known_words={"きつね"}, llm=_FakeLLM(up=False))
        self.assertEqual(r["version"], 2)
        self.assertEqual(r["taxonomy"]["きつね"], "哺乳動物")
        self.assertIn("きつね", r["researched"])

    def test_held_out_words_are_never_researched_or_asked(self):
        # force きつね into the held-out set and give the LLM a table for it
        held = "きつね" if wm._held_out("きつね") else "からす"
        state = wm._blank()
        state["selection_version"] = wm.SELECTION_VERSION
        state["selection_words"] = [held]
        llm = _FakeLLM(table={held: "道具"}, up=True)
        r = wm.learn_and_evaluate([{"events": STORY}] * 8, state, 4,
                                  known_words={"きつね", "からす", "ぶどう"}, llm=llm)
        self.assertNotIn(held, r["researched"])
        self.assertNotIn(held, r["llm_asked"])
        self.assertNotIn(held, llm.asked)

    def test_held_out_deterministic(self):
        a = {w: wm._held_out(w) for w in ("きつね", "からす", "ぶどう", "つき", "うみ")}
        b = {w: wm._held_out(w) for w in ("きつね", "からす", "ぶどう", "つき", "うみ")}
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
