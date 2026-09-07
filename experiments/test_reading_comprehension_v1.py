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
    def test_structured_stories_beat_the_frequency_baseline_significantly(self):
        report = rcp.evaluate_comprehension(structured_stories(n=120), {})
        self.assertEqual(report["status"], "measured")
        self.assertGreater(report["consequence"], report["consequence_baseline"])
        self.assertGreater(report["consequence_z"], rcp.SIGNIFICANCE_Z)
        self.assertTrue(report["beats_baseline_significant"])
        self.assertFalse(report["beats_baseline"])
        fp1 = report["snapshot_fingerprint"]
        # re-measuring the SAME frozen snapshot with no training growth is NOT
        # an independent replication -- the streak does not advance
        same = rcp.evaluate_comprehension(structured_stories(n=120), report)
        self.assertEqual(same["snapshot_fingerprint"], fp1)   # snapshot frozen
        self.assertFalse(same["beats_baseline"])
        # a second measurement at a LARGER training size does count
        grown = structured_stories(n=120) + structured_stories(n=80, seed=9)
        second = rcp.evaluate_comprehension(grown, same)
        self.assertEqual(second["snapshot_fingerprint"], fp1)   # still frozen
        self.assertTrue(second["beats_baseline"])
        self.assertGreaterEqual(second["significant_streak"], 2)

    def test_eval_regime_change_does_not_inherit_the_old_streak(self):
        stories = structured_stories(n=120)
        held = [{"url": s["url"], "events": s["events"]}
                for s in stories if rcp._held_out(s["url"])]
        # a fixture that was "confirmed" under a previous regime with streak 2
        old = {"eval_regime": "legacy_regime_v0", "significant_streak": 2,
               "beats_baseline": True, "beats_baseline_significant": True,
               "last_significant_train": 25, "test_snapshot": held,
               "learning_curve": [{"train_stories": 25}]}
        r = rcp.evaluate_comprehension(stories, old)
        self.assertEqual(r["eval_regime"], rcp.EVAL_REGIME)
        self.assertEqual(r["regime_reset_from"], "legacy_regime_v0")
        # the first measurement under the new regime is streak 1 at most, never a pass
        self.assertFalse(r["beats_baseline"])
        self.assertLessEqual(r["significant_streak"], 1)
        # only a SECOND measurement at a larger training size can reach streak 2
        grown = structured_stories(n=120) + structured_stories(n=90, seed=11)
        r2 = rcp.evaluate_comprehension(grown, r)
        self.assertGreaterEqual(r2["significant_streak"], 2)
        self.assertTrue(r2["beats_baseline"])

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
