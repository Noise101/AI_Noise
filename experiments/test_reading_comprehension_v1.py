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
        for n in sizes:
            r = rcp.evaluate_comprehension(structured_stories(n=n, seed=1), r, cycle=n)
        return r

    def test_selection_alone_never_confirms_capability(self):
        r = rcp.evaluate_comprehension(structured_stories(n=160), {}, cycle=1)
        self.assertEqual(r["status"], "measured")
        self.assertTrue(r["selection"]["significant"])
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)
        for i in range(4):                                       # re-measure the same selection
            r = rcp.evaluate_comprehension(structured_stories(n=160), r, cycle=2 + i)
        self.assertFalse(r["capability_confirmed"])              # still not confirmed
        self.assertGreaterEqual(r["selection_measurements"], 5)
        # streak never advances without training growth (anchor-based, P1-3)
        self.assertLessEqual(r["selection_significant_streak"], 1)

    def test_gradual_growth_reaches_streak_2_and_confirms(self):
        # re-audit #6 P1-3: 160->180->...->430 gradual growth must eventually
        # advance the anchor-based streak to 2 and register + pass a final.
        r = self._grow({}, [160, 180, 200, 230])
        self.assertGreaterEqual(r["selection_significant_streak"], 2)
        self.assertGreaterEqual(r["final_opened_count"], 1)
        self.assertTrue(r["final_result"]["significant"])
        self.assertTrue(r["capability_confirmed_ever"])
        pre = r["final_preconditions"]
        for k in ("model_fingerprint", "identity", "significance_z", "selection_fingerprint",
                  "final_snapshot_fingerprint", "registered_at_train"):
            self.assertIn(k, pre)

    def test_ordinary_continued_training_does_not_burn_the_reserve(self):
        # re-audit #6 P1-4: after a pass, a one-book / few-cycle model change must
        # NOT open the reserve.  The reserve opens only for a NEW candidate.
        r = self._grow({}, [160, 180, 200, 230])
        n_after_first = r["final_opened_count"]
        self.assertGreaterEqual(n_after_first, 1)
        r2 = rcp.evaluate_comprehension(structured_stories(n=235, seed=1), r, cycle=235)  # +1-ish
        self.assertEqual(r2["final_opened_count"], n_after_first)           # no new final
        self.assertTrue(r2["capability_confirmed_ever"])                    # still confirmed ever
        self.assertFalse(r2["capability_confirmed_current_model"])          # but model moved on
        r3 = self._grow(r2, [400])                                         # a real retraining
        self.assertGreater(r3["final_opened_count"], n_after_first)         # NOW a fresh final
        self.assertLessEqual(r3["final_opened_count"], rcp.jb.FINAL_QUERY_BUDGET)

    def test_confirmed_checkpoint_is_permanent_and_separate_from_current(self):
        r = self._grow({}, [160, 180, 200, 230, 300])
        self.assertTrue(r["capability_confirmed_ever"])
        self.assertGreaterEqual(len(r["confirmed_checkpoints"]), 1)
        cp = r["confirmed_checkpoints"][0]
        self.assertIn("model_fingerprint", cp)
        self.assertIn("final_snapshot_fingerprint", cp)

    def test_train_selection_final_collections_are_disjoint(self):
        r = self._grow({}, [160, 260])
        d = r["collection_disjointness"]
        self.assertEqual(d["train_x_selection"], [])
        self.assertEqual(d["train_x_final"], [])
        self.assertEqual(d["selection_x_final"], [])
        self.assertTrue(r["collections_disjoint"])

    def test_eval_regime_change_does_not_inherit_the_old_streak_or_finals(self):
        old = {"eval_regime": "legacy_regime_v0",
               "selection_significant_streak": 2, "capability_confirmed": True,
               "capability_confirmed_ever": True,
               "candidate_checkpoints": [{"final_result": {"significant": True}}],
               "test_snapshot": [{"url": s["url"], "events": s["events"]}
                                 for s in structured_stories(120) if rcp._held_out(s["url"])],
               "learning_curve": [{"train_stories": 25}]}
        r = rcp.evaluate_comprehension(structured_stories(n=160), old, cycle=1)
        self.assertEqual(r["eval_regime"], rcp.EVAL_REGIME)
        self.assertEqual(r["regime_reset_from"], "legacy_regime_v0")
        self.assertFalse(r["capability_confirmed"])
        self.assertFalse(r["capability_confirmed_ever"])
        self.assertEqual(r["final_opened_count"], 0)             # old finals not inherited
        self.assertLessEqual(r["selection_significant_streak"], 1)

    def test_insufficient_final_or_selection_is_reported_honestly(self):
        r = rcp.evaluate_comprehension(structured_stories(n=45), {}, cycle=1)
        self.assertFalse(r["capability_confirmed"])
        if r["status"] == "measured":
            self.assertIn(r["final_status"],
                          ("awaiting_selection_streak_2", "registerable_next_cycle",
                           "candidate_registered_final_snapshot_unavailable"))
        else:
            self.assertEqual(r["status"], "insufficient_selection_stories")

    def test_snapshot_fingerprint_depends_on_event_content(self):
        a = [{"url": "http://x/1", "events": [{"subject": "a", "verb": "b", "obj": ""}]}]
        b = [{"url": "http://x/1", "events": [{"subject": "a", "verb": "CHANGED", "obj": ""}]}]
        self.assertNotEqual(rcp._fingerprint(a), rcp._fingerprint(b))

    def test_comprehension_model_fingerprint_tracks_event_content_and_parser(self):
        # re-audit #6 P1-5: same URL, changed events -> different fingerprint
        a = [{"url": "u1", "events": [{"subject": "a", "verb": "b", "obj": "c"}]},
             {"url": "u2", "events": [{"subject": "d", "verb": "e", "obj": "f"}]}]
        b = [{"url": "u1", "events": [{"subject": "a", "verb": "ZZZ", "obj": "c"}]},
             {"url": "u2", "events": [{"subject": "d", "verb": "e", "obj": "f"}]}]
        self.assertNotEqual(rcp._comprehension_model_fingerprint(a),
                            rcp._comprehension_model_fingerprint(b))
        tf = rcp.comprehension_training_fingerprint(a)
        for k in ("identity", "identity_fingerprint", "training_set_fingerprint",
                  "training_source_count", "training_urls"):
            self.assertIn(k, tf)
        self.assertIn("parser_version", tf["identity"])

    def test_a_parser_change_makes_old_selection_snapshot_not_reused_as_current(self):
        # a regime bump on a parser change means the old test_snapshot is demoted
        # to selection and the confirmation resets (never flows as current ability)
        old = {"eval_regime": "tiered_frozen_v1", "capability_confirmed_ever": True,
               "candidate_checkpoints": [{"final_result": {"significant": True}}],
               "test_snapshot": [{"url": s["url"], "events": s["events"]}
                                 for s in structured_stories(120) if rcp._held_out(s["url"])]}
        r = rcp.evaluate_comprehension(structured_stories(160), old, cycle=1)
        self.assertEqual(r["regime_reset_from"], "tiered_frozen_v1")
        self.assertFalse(r["capability_confirmed"])
        self.assertFalse(r["capability_confirmed_ever"])

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
        report = rcp.evaluate_comprehension(structured_stories(n=10), {}, cycle=1)
        self.assertEqual(report["status"], "insufficient_selection_stories")
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
