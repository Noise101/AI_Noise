import random
import unittest

import japanese_retell_v1 as jr


def _story(subjects, verbs, objs, url):
    events = [{"subject": s, "verb": v, "obj": o, "confidence": 0.9, "roles": {}}
              for s, v, o in zip(subjects, verbs, objs)]
    return {"url": url, "events": events}


def folktale_stories(n=140, seed=1):
    rng = random.Random(seed)
    animals = ["きつね", "うさぎ", "たぬき", "ねこ", "いぬ", "くま", "ねずみ"]
    goals = ["ぶどう", "にんじん", "かき", "さかな", "はちみつ", "くり"]
    verbs = ["みつける", "とる", "たべる", "かえる", "ねむる", "なく"]
    out = []
    for i in range(n):
        a, o = rng.choice(animals), rng.choice(goals)
        subs = [a] * len(verbs)
        objs = [o, o, o, "", "", ""]
        out.append(_story(subs, verbs, objs, f"http://s/{i}"))
    return out


class RetellTest(unittest.TestCase):
    def test_polite_past_conjugation(self):
        self.assertEqual(jr.polite_past("行く"), "行きました")
        self.assertEqual(jr.polite_past("食べる"), "食べました")
        self.assertEqual(jr.polite_past("見る"), "見ました")
        self.assertEqual(jr.polite_past("とる"), "とりました")
        self.assertEqual(jr.polite_past("する"), "しました")

    def test_verbalise_orders_particles_subject_oblique_object_verb(self):
        event = {"subject": "いぬ", "verb": "みつける", "obj": "さかな",
                 "roles": {"で": "かわ"}}
        self.assertEqual(jr.verbalise_event(event), "いぬがかわでさかなをみつけました。")

    def test_clean_events_round_trip_with_high_fidelity(self):
        events = _story(["いぬ", "いぬ", "ねこ"], ["みつける", "たべる", "なく"],
                        ["さかな", "さかな", ""], "http://x/1")["events"]
        score = jr.score_retelling(events, jr.retell(events))
        self.assertGreaterEqual(score["recall"], 0.99)
        self.assertGreaterEqual(score["order_correlation"], 0.99)

    def test_coherent_retelling_keeps_full_fidelity(self):
        events = _story(["いぬ", "いぬ", "ねこ"], ["みつける", "たべる", "なく"],
                        ["さかな", "さかな", ""], "http://x/1")["events"]
        score = jr.score_retelling(events, jr.retell(events))
        self.assertGreaterEqual(score["coherence"], 0.99)
        self.assertAlmostEqual(score["fidelity"], score["structural_fidelity"], places=3)

    def test_deictic_placeholder_on_every_clause_is_incoherent(self):
        # the "それが<壊れた動詞>" degeneracy a caregiver flags as 意味不明
        events = [{"subject": "それ", "verb": v, "obj": "", "confidence": 0.9, "roles": {}}
                  for v in ("うめる", "まもなく", "しぬ", "でる", "もでてく")]
        self.assertLess(jr.retelling_coherence(events), 0.4)
        score = jr.score_retelling(events, jr.retell(events))
        self.assertLess(score["fidelity"], score["structural_fidelity"])
        self.assertLess(score["fidelity"], 0.4)

    def test_stranded_particle_in_the_verb_slot_lowers_coherence(self):
        broken = [{"subject": "いったいまいにちどんな", "verb": "のをたべているんです",
                   "obj": "", "confidence": 0.9, "roles": {}},
                  {"subject": "いったいまいにちどんな", "verb": "きく",
                   "obj": "", "confidence": 0.9, "roles": {}},
                  {"subject": "いったいまいにちどんな", "verb": "う",
                   "obj": "", "confidence": 0.9, "roles": {}}]
        self.assertLess(jr.retelling_coherence(broken), 0.3)

    def test_in_order_retelling_beats_shuffled_on_held_out_stories(self):
        stories = folktale_stories()
        report = jr.evaluate_retelling(stories, {})
        self.assertEqual(report["status"], "measured")
        self.assertGreater(report["fidelity"], report["fidelity_baseline"])
        self.assertGreater(report["gain_z"], jr.SIGNIFICANCE_Z)
        self.assertTrue(report["beats_baseline_significant"])
        self.assertFalse(report["beats_baseline"])          # needs two cycles
        second = jr.evaluate_retelling(stories, report)
        self.assertTrue(second["beats_baseline"])
        self.assertEqual(second["significant_streak"], 2)

    def test_insufficient_stories_reports_cleanly(self):
        report = jr.evaluate_retelling(folktale_stories(n=10), {})
        self.assertEqual(report["status"], "insufficient_stories")
        self.assertIsNone(report["fidelity"])
        self.assertFalse(report["beats_baseline"])

    def test_free_retell_is_a_noop_without_a_trained_rnn_state(self):
        self.assertEqual(jr.free_retell(None, "むかしむかし"), "")
        self.assertEqual(jr.free_retell({}, "むかしむかし"), "")

    def test_free_retell_generates_japanese_from_a_state(self):
        import japanese_sequence_v1 as js
        model = js.TinyRNN(sorted("むかしあおじいさんやまへ行きました。犬が"))
        text = jr.free_retell(model.state(), "むかし", length=30)
        self.assertEqual(len(text), 30)

    def test_held_out_split_is_deterministic_and_disjoint(self):
        stories = folktale_stories()
        train = {s["url"] for s in stories if not jr._held_out(s["url"])}
        test = {s["url"] for s in stories if jr._held_out(s["url"])}
        self.assertFalse(train & test)
        self.assertGreater(len(test), 0)


if __name__ == "__main__":
    unittest.main()
