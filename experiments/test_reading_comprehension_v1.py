import random
import unittest
from collections import Counter

import reading_comprehension_v1 as rcp


def structured_stories(n=120, seed=1):
    """Folktales with a real narrative schema: find -> want -> approach ->
    fail -> think -> decide."""
    rng = random.Random(seed)
    animals = ["きつね", "うさぎ", "たぬき", "ねこ", "いぬ", "くま", "ねずみ", "さる"]
    goals = ["ぶどう", "にんじん", "かき", "さかな", "ほね", "はちみつ", "くり", "みず"]
    schema = ["みつける", "ほしくなる", "ちかづく", "しっぱいする", "かんがえる", "きめる"]
    out = []
    for i in range(n):
        a, o = rng.choice(animals), rng.choice(goals)
        events = [{"subject": a, "verb": v, "obj": o if j == 0 else "", "confidence": 0.9}
                  for j, v in enumerate(schema)]
        out.append({"url": f"http://s/{i}", "events": events})
    return out


def unstructured_stories(n=120, seed=2):
    rng = random.Random(seed)
    animals = ["きつね", "うさぎ", "たぬき", "ねこ", "いぬ", "くま"]
    verbs = ["みつける", "ほしくなる", "ちかづく", "しっぱいする", "かんがえる",
             "きめる", "はしる", "ねむる"]
    return [{"url": f"http://n/{i}",
             "events": [{"subject": rng.choice(animals), "verb": v, "obj": "", "confidence": 0.9}
                        for v in rng.sample(verbs, 6)]}
            for i in range(n)]


class ReadingComprehensionTest(unittest.TestCase):
    def _grow(self, prev, sizes):
        r = prev
        for i, n in enumerate(sizes):
            r = rcp.evaluate_comprehension(structured_stories(n=n, seed=1), r)
        return r

    def test_selection_alone_never_confirms_capability(self):
        # SELECTION is diagnostic: significant there is NOT a capability claim,
        # no matter how many times it is measured.
        r = rcp.evaluate_comprehension(structured_stories(n=160), {})
        self.assertEqual(r["status"], "measured")
        self.assertTrue(r["selection"]["significant"])           # structured data does beat freq
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)
        for _ in range(4):                                       # re-measure the same selection
            r = rcp.evaluate_comprehension(structured_stories(n=160), r)
        self.assertFalse(r["capability_confirmed"])              # still not confirmed
        self.assertGreaterEqual(r["selection_measurements"], 5)

    def test_capability_needs_an_unopened_final_to_also_pass(self):
        r = self._grow({}, [160, 260])                           # grow -> sel streak >= 2
        self.assertGreaterEqual(r["selection_significant_streak"], 2)
        self.assertGreaterEqual(r["final_opened_count"], 1)
        self.assertEqual(r["final_status"], "opened")
        self.assertTrue(r["final_result"]["significant"])
        self.assertTrue(r["capability_confirmed"])
        # the final's preconditions were frozen from the selection side
        pre = r["final_preconditions"]
        for k in ("model_fingerprint", "parser_version", "eval_regime", "scoring_version",
                  "baseline_definition", "significance_z", "selection_result",
                  "final_snapshot_fingerprint", "final_query_index"):
            self.assertIn(k, pre)

    def test_re_measuring_the_same_final_is_not_a_second_confirmation(self):
        r = self._grow({}, [160, 260])
        opened = r["final_opened_count"]
        self.assertGreaterEqual(opened, 1)
        r2 = rcp.evaluate_comprehension(structured_stories(n=260, seed=1), r)  # same model + data
        self.assertEqual(r2["final_opened_count"], opened)       # no extra open
        self.assertEqual(r2["final_history"], r["final_history"])

    def test_budget_caps_independent_finals_then_reports_stale(self):
        r = self._grow({}, [160, 260, 460, 820])                 # 4 big retrainings
        self.assertLessEqual(r["final_opened_count"], rcp.jb.FINAL_QUERY_BUDGET)
        if r["final_opened_count"] >= rcp.jb.FINAL_QUERY_BUDGET and r["final_stale_for_current_model"]:
            self.assertEqual(r["final_status"], "stale_needs_fresh_final")
            self.assertFalse(r["capability_confirmed"])

    def test_train_selection_final_collections_are_disjoint(self):
        r = self._grow({}, [160, 260])
        d = r["collection_disjointness"]
        self.assertEqual(d["train_x_selection"], [])
        self.assertEqual(d["train_x_final"], [])
        self.assertEqual(d["selection_x_final"], [])
        self.assertTrue(r["collections_disjoint"])

    def test_eval_regime_change_does_not_inherit_the_old_streak_or_finals(self):
        old = {"eval_regime": "legacy_regime_v0", "significant_streak": 2,
               "selection_significant_streak": 2, "capability_confirmed": True,
               "beats_baseline": True, "final_history": [{"tier": "final", "result": {"significant": True}}],
               "test_snapshot": [{"url": s["url"], "events": s["events"]}
                                 for s in structured_stories(120) if rcp._held_out(s["url"])],
               "learning_curve": [{"train_stories": 25}]}
        r = rcp.evaluate_comprehension(structured_stories(n=160), old)
        self.assertEqual(r["eval_regime"], rcp.EVAL_REGIME)
        self.assertEqual(r["regime_reset_from"], "legacy_regime_v0")
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)             # old finals not inherited
        self.assertLessEqual(r["selection_significant_streak"], 1)

    def test_insufficient_final_stories_is_reported_honestly(self):
        # too few collections in the 'final' tier -> no fake final
        r = rcp.evaluate_comprehension(structured_stories(n=45), {})
        if r["status"] == "measured":
            self.assertIn(r["final_status"], ("unopened", "insufficient_final_stories"))
            self.assertFalse(r["capability_confirmed"])

    def test_snapshot_fingerprint_depends_on_event_content(self):
        a = [{"url": "http://x/1", "events": [{"subject": "a", "verb": "b", "obj": ""}]}]
        b = [{"url": "http://x/1", "events": [{"subject": "a", "verb": "CHANGED", "obj": ""}]}]
        self.assertNotEqual(rcp._fingerprint(a), rcp._fingerprint(b))

    def test_unstructured_stories_do_not_beat_the_baseline(self):
        report = rcp.evaluate_comprehension(unstructured_stories(), {})
        self.assertLess(report["consequence_z"], rcp.SIGNIFICANCE_Z)
        self.assertFalse(report["beats_baseline"])

    def test_held_out_split_is_deterministic_and_source_disjoint(self):
        stories = structured_stories()
        train_urls = {s["url"] for s in stories if not rcp._held_out(s["url"])}
        test_urls = {s["url"] for s in stories if rcp._held_out(s["url"])}
        self.assertFalse(train_urls & test_urls)
        self.assertGreater(len(test_urls), 0)
        # re-running gives the same split
        self.assertEqual(test_urls, {s["url"] for s in stories if rcp._held_out(s["url"])})

    def test_insufficient_stories_reports_cleanly(self):
        report = rcp.evaluate_comprehension(structured_stories(n=10), {})
        self.assertEqual(report["status"], "insufficient_stories")
        self.assertIsNone(report["comprehension_score"])
        self.assertFalse(report["beats_baseline"])

    def test_book_comprehension_scores_a_coherent_story_higher_than_a_jumbled_one(self):
        model = rcp.ComprehensionModel().fit([s["events"] for s in structured_stories(n=80)])
        coherent = structured_stories(n=1, seed=99)[0]["events"]
        jumbled = list(reversed(coherent))
        good = rcp.book_comprehension(coherent, model, set())
        bad = rcp.book_comprehension(jumbled, model, set())
        self.assertGreater(good["score"], bad["score"])
        self.assertGreater(good["tests"]["consequence"], 0.4)

    def test_incoherent_parse_is_gated_below_a_coherent_one(self):
        model = rcp.ComprehensionModel().fit([s["events"] for s in structured_stories(n=80)])
        coherent = structured_stories(n=1, seed=7)[0]["events"]
        # same narrative, but the extractor collapsed every subject to a
        # deictic placeholder and mangled the verbs
        incoherent = [{"subject": "それ", "verb": v, "obj": "", "confidence": 0.9}
                      for v in ("のをみつける", "まもなく", "だろう", "です", "う", "もなる")]
        good = rcp.book_comprehension(coherent, model, set())
        bad = rcp.book_comprehension(incoherent, model, set())
        self.assertGreater(good["tests"]["coherence"], 0.9)
        self.assertLess(bad["tests"]["coherence"], 0.3)
        self.assertGreater(good["score"], bad["score"])

    def test_ordering_test_reconstructs_a_known_schema(self):
        model = rcp.ComprehensionModel().fit([s["events"] for s in structured_stories(n=80)])
        events = structured_stories(n=1, seed=5)[0]["events"]
        result = rcp.book_comprehension(events, model, set())
        self.assertGreaterEqual(result["tests"]["ordering"], 0.75)

    def _vocab_setup(self):
        stories = structured_stories(n=40)
        model = rcp.ComprehensionModel().fit([s["events"] for s in stories])
        cooc = rcp.build_cooccurrence(stories)
        target_events = [{"subject": "きつね", "obj": "ぶどう", "verb": "みつける",
                          "sentence": "きつねがぶどうをみつけました。"},
                         {"subject": "きつね", "obj": "ぶどう", "verb": "ほしくなる",
                          "sentence": "きつねはぶどうがほしくなりました。"}]
        return model, cooc, target_events

    def test_vocabulary_use_test_passes_only_with_positive_context_evidence(self):
        model, cooc, events = self._vocab_setup()
        good = rcp.vocabulary_use_test("ぶどう", events, ["いし", "くも", "かぜ"], model,
                                       cooccurrence=cooc, url="u")
        self.assertTrue(good["tested"])
        self.assertGreaterEqual(good["cloze_rate"], 0.5)

    def test_vocabulary_use_test_fails_when_all_candidates_have_no_evidence(self):
        model, cooc, events = self._vocab_setup()
        # a target the co-occurrence table has never seen -> no positive evidence
        blind = rcp.vocabulary_use_test("たからばこ",
            [{"subject": "たからばこ", "obj": "ぶどう", "verb": "みつける",
              "sentence": "たからばこがぶどうをみつけました。"}] * 3,
            ["いし", "くも"], model, cooccurrence=cooc, url="u")
        self.assertEqual(blind["cloze_rate"], 0.0)

    def test_vocabulary_use_test_is_order_independent(self):
        model, cooc, events = self._vocab_setup()
        a = rcp.vocabulary_use_test("ぶどう", events, ["いし", "くも", "かぜ"], model,
                                    cooccurrence=cooc, url="u")
        b = rcp.vocabulary_use_test("ぶどう", events, ["かぜ", "くも", "いし"], model,
                                    cooccurrence=cooc, url="u")
        self.assertEqual(a["cloze_rate"], b["cloze_rate"])

    def test_vocabulary_use_test_target_in_sentence_alone_is_not_a_pass(self):
        model, _cooc, _ = self._vocab_setup()
        empty_cooc = {}
        r = rcp.vocabulary_use_test("ぶどう",
            [{"subject": "きつね", "obj": "ぶどう", "verb": "みつける",
              "sentence": "きつねがぶどうをみつけました。"}] * 3,
            ["いし", "くも"], model, cooccurrence=empty_cooc, url="u")
        self.assertFalse(r["tested"])

    def test_vocabulary_use_test_fails_on_a_tie_with_a_distractor(self):
        model, _c, events = self._vocab_setup()
        # a table where the target and a distractor co-occur equally with context
        tie = {"ぶどう": Counter({"きつね": 1}), "いし": Counter({"きつね": 1})}
        r = rcp.vocabulary_use_test("ぶどう", events, ["いし", "くも"], model,
                                    cooccurrence=tie, url="u")
        self.assertEqual(r["cloze_rate"], 0.0)   # strict > required

    def test_cooccurrence_is_word_by_context_word_from_events(self):
        stories = [{"url": "u", "events": [{"subject": "きつね", "obj": "ぶどう",
                                            "verb": "みつける"}]}]
        table = rcp.build_cooccurrence(stories)
        self.assertIn("ぶどう", table["きつね"])
        self.assertNotIn("かぜ", table.get("きつね", {}))

    def test_learning_curve_records_the_comprehension_score_over_cycles(self):
        state = {}
        for _ in range(3):
            state = rcp.evaluate_comprehension(structured_stories(), state)
        self.assertGreaterEqual(len(state["learning_curve"]), 1)
        self.assertIn("comprehension_score", state["learning_curve"][-1])


if __name__ == "__main__":
    unittest.main()
