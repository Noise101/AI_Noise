import random
import unittest

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
        report = rcp.evaluate_comprehension(structured_stories(), {})
        self.assertEqual(report["status"], "measured")
        self.assertGreater(report["consequence"], report["consequence_baseline"])
        self.assertGreater(report["consequence_z"], rcp.SIGNIFICANCE_Z)
        self.assertTrue(report["beats_baseline_significant"])
        # "beats_baseline" needs it to hold for two consecutive measurements
        self.assertFalse(report["beats_baseline"])
        second = rcp.evaluate_comprehension(structured_stories(), report)
        self.assertTrue(second["beats_baseline"])
        self.assertEqual(second["significant_streak"], 2)

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

    def test_vocabulary_use_test_promotes_a_word_that_fits_its_sentences(self):
        model = rcp.ComprehensionModel().fit([s["events"] for s in structured_stories(n=40)])
        result = rcp.vocabulary_use_test(
            "ぶどう",
            ["きつねはぶどうをみつけました。", "きつねはぶどうがほしくなりました。"],
            distractors=["いし", "くも", "かぜ"], model=model)
        self.assertTrue(result["tested"])
        self.assertGreaterEqual(result["cloze_rate"], 0.5)

    def test_learning_curve_records_the_comprehension_score_over_cycles(self):
        state = {}
        for _ in range(3):
            state = rcp.evaluate_comprehension(structured_stories(), state)
        self.assertGreaterEqual(len(state["learning_curve"]), 1)
        self.assertIn("comprehension_score", state["learning_curve"][-1])


if __name__ == "__main__":
    unittest.main()
