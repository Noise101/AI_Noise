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

    def test_body_part_doing_an_action_verb_lowers_coherence(self):
        # "おなかが言いました" -- a body part cannot be the agent of 言う
        events = [{"subject": "おなか", "verb": v, "obj": "", "confidence": 0.9,
                   "roles": {}, "sentence": ""}
                  for v in ("すいてたまる", "しんだまねをする", "言う")]
        self.assertLess(jr.retelling_coherence(events), 0.6)

    def test_a_protagonist_absent_from_every_source_sentence_lowers_coherence(self):
        # "いっぴき" (a counter) threaded as the subject, never in the actual text
        events = [{"subject": "いっぴき", "verb": v, "obj": "", "confidence": 0.9,
                   "roles": {}, "sentence": s}
                  for v, s in (("はしる", "ねこがはしりました。"),
                               ("たべる", "ねずみをたべました。"),
                               ("なく", "おおごえでなきました。"))]
        self.assertLess(jr.retelling_coherence(events), 0.6)

    def test_template_roundtrip_is_a_diagnostic_not_a_capability(self):
        # the in-order-vs-shuffled TEMPLATE comparison used to earn beats_baseline
        # -- it only measured "can I serialise a list in order", so it must not
        stories = folktale_stories()
        report = jr.evaluate_retelling(stories, {})
        self.assertEqual(report["status"], "no_generation_model")   # no RNN state
        self.assertFalse(report["beats_baseline"])
        self.assertGreater(report["roundtrip_fidelity"], 0.0)       # still reported
        second = jr.evaluate_retelling(stories, report)
        self.assertFalse(second["beats_baseline"])

    def test_generation_capability_needs_the_rnn_to_beat_the_shuffled_template(self):
        import japanese_sequence_v1 as js
        stories = folktale_stories()
        state = js.TinyRNN(sorted("むかしきつねうさぎぶどうをみつけるたべるなく。")).state()
        report = jr.evaluate_retelling(stories, {}, rnn_state=state)
        self.assertEqual(report["status"], "measured")
        self.assertIn("generation_gain", report)
        # an untrained RNN will not beat the baseline -- that is the honest result
        self.assertFalse(report["beats_baseline"])

    def test_insufficient_stories_reports_cleanly(self):
        report = jr.evaluate_retelling(folktale_stories(n=10), {})
        self.assertEqual(report["status"], "insufficient_stories")
        self.assertIsNone(report["roundtrip_fidelity"])
        self.assertFalse(report["beats_baseline"])

    def test_the_held_out_test_set_is_a_frozen_snapshot(self):
        first = jr.evaluate_retelling(folktale_stories(140), {})
        fp = first["snapshot_fingerprint"]
        n_test = first["test_stories"]
        grown = folktale_stories(140) + folktale_stories(120, seed=7)
        second = jr.evaluate_retelling(grown, first)
        self.assertEqual(second["snapshot_fingerprint"], fp)     # unchanged
        self.assertEqual(second["test_stories"], n_test)          # did not grow
        self.assertGreater(second["train_stories"], first["train_stories"])  # training grew

    def test_retelling_capability_baseline_has_the_same_information(self):
        # the baseline is the SAME generator on a SHUFFLED event representation --
        # identical (subject, verb, object) info, ordering removed
        import japanese_sequence_v1 as js
        stories = folktale_stories(140)
        state = js.TinyRNN(sorted("むかしきつねうさぎぶどうをみつけるとるたべるなく。")).state()
        report = jr.evaluate_retelling(stories, {}, rnn_state=state)
        self.assertEqual(report["status"], "measured")
        self.assertIn("generation_gain", report)
        self.assertFalse(report["beats_baseline"])      # untrained RNN -> honest fail

    def test_re_measuring_the_same_snapshot_twice_is_not_a_replication(self):
        import japanese_sequence_v1 as js
        stories = folktale_stories(140)
        state = js.TinyRNN(sorted("むかしきつねうさぎぶどうみつけるたべる。" * 3)).state()
        r1 = jr.evaluate_retelling(stories, {}, rnn_state=state)
        r2 = jr.evaluate_retelling(stories, r1, rnn_state=state)   # same data
        self.assertLessEqual(r2["significant_streak"], max(1, r1["significant_streak"]))

    _EVENTS = [{"subject": "きつね", "verb": "みつける", "obj": "ぶどう"},
               {"subject": "きつね", "verb": "たべる", "obj": ""}]

    def test_free_retell_is_a_noop_without_a_trained_rnn_state_or_events(self):
        import japanese_sequence_v1 as js
        self.assertEqual(jr.free_retell(None, self._EVENTS), "")
        self.assertEqual(jr.free_retell({}, self._EVENTS), "")
        state = js.TinyRNN(sorted("むかしきつねぶどうをみつけるたべる。")).state()
        self.assertEqual(jr.free_retell(state, []), "")

    def test_free_retell_is_conditioned_on_events_not_the_gold_text(self):
        import japanese_sequence_v1 as js
        state = js.TinyRNN(sorted("むかしきつねぶどうをみつけるたべるなく。")).state()
        text = jr.free_retell(state, self._EVENTS)
        self.assertTrue(text)
        # the generator only ever sees subject/verb/object -- not any sentence
        self.assertNotIn("gold", str(self._EVENTS))
        # deterministic for the same event sequence
        self.assertEqual(text, jr.free_retell(state, self._EVENTS))

    def test_held_out_split_is_deterministic_and_disjoint(self):
        stories = folktale_stories()
        train = {s["url"] for s in stories if not jr._held_out(s["url"])}
        test = {s["url"] for s in stories if jr._held_out(s["url"])}
        self.assertFalse(train & test)
        self.assertGreater(len(test), 0)


if __name__ == "__main__":
    unittest.main()
