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

    def _rnn(self, extra=""):
        import japanese_sequence_v1 as js
        vocab = sorted(set("むかしきつねうさぎたぬきねこいぬくまねずみさるぶどうにんじんかき"
                           "さかなほねはちみつくりみずみつけるとるたべるかえるねむるなく"
                           "ほしくなるちかづくしっぱいするかんがえるきめるがをはにでました。" + extra))
        st = js.TinyRNN(vocab).state()
        st["model_fingerprint"], st["steps_trained"] = "test-untrained", 0
        return st

    def test_template_roundtrip_is_a_diagnostic_not_a_capability(self):
        stories = folktale_stories()
        report = jr.evaluate_retelling(stories, {})
        self.assertEqual(report["status"], "no_generation_model")   # no RNN state
        self.assertFalse(report["beats_baseline"])
        self.assertGreater(report["roundtrip_fidelity"], 0.0)       # still reported
        second = jr.evaluate_retelling(stories, report)
        self.assertFalse(second["beats_baseline"])

    def test_untrained_rnn_cannot_pass_narrative_order_recovery(self):
        stories = folktale_stories(180)
        report = jr.evaluate_retelling(stories, {}, rnn_state=self._rnn())
        self.assertEqual(report["status"], "measured")
        self.assertIn("order_gain", report)
        self.assertIn("rnn_pairwise_accuracy", report)
        # the verb-position baseline (built from training) IS informative here,
        # so it is a real bar ...
        self.assertGreater(report["position_baseline_accuracy"], 0.6)
        # ... and an untrained likelihood model scores ~chance and does not clear it
        self.assertFalse(report["beats_baseline"])
        self.assertLess(report["gain_z"], jr.SIGNIFICANCE_Z)

    def test_a_scorer_that_ignores_order_cannot_pass(self):
        from unittest.mock import patch
        stories = folktale_stories(180)
        state = {"vocab": list("abc"), "model_fingerprint": "flat", "steps_trained": 1}
        with patch.object(jr, "_rnn_scorer", lambda s: (lambda text: 1.0)):
            report = jr.evaluate_retelling(stories, {}, rnn_state=state)
        self.assertEqual(report["status"], "measured")
        # a constant score ties on every pair -> exactly 0.5, no order signal
        self.assertEqual(report["rnn_pairwise_accuracy"], 0.5)
        self.assertLess(report["order_gain"], 0)          # loses to the informative baseline
        self.assertFalse(report["beats_baseline"])

    def test_the_simple_baseline_does_not_beat_itself(self):
        # feeding the position baseline in as the "scorer" must still not pass:
        # order_gain is rnn_acc - baseline_acc, and here they are ~equal
        from unittest.mock import patch
        stories = folktale_stories(180)
        pos = jr._position_model([s for s in stories if not jr._held_out(s["url"])])

        def scorer_from_positions(_state):
            def score(text):                     # lower = earlier-preferred
                return sum(pos.get(v, 0.5) for v in ("みつける", "とる", "たべる",
                                                     "かえる", "ねむる", "なく") if v in text)
            return score
        with patch.object(jr, "_rnn_scorer", scorer_from_positions):
            report = jr.evaluate_retelling(
                stories, {}, rnn_state={"vocab": list("a"), "model_fingerprint": "p", "steps_trained": 1})
        self.assertFalse(report["beats_baseline"])

    def test_free_retell_does_not_transcribe_the_event_order(self):
        state = self._rnn()
        ev = [{"subject": "きつね", "verb": "みつける", "obj": "ぶどう"},
              {"subject": "きつね", "verb": "たべる", "obj": ""},
              {"subject": "きつね", "verb": "なく", "obj": ""}]
        a = jr.free_retell(state, ev)
        b = jr.free_retell(state, list(reversed(ev)))
        self.assertTrue(a)
        self.assertEqual(a, b)          # permutation-invariant prime -> same output

    def test_reeval_is_gated_on_the_rnn_fingerprint(self):
        stories = folktale_stories(180)
        st = self._rnn()
        st["model_fingerprint"], st["steps_trained"] = "fp-1", 500_000
        r1 = jr.evaluate_retelling(stories, {}, rnn_state=st)
        self.assertTrue(r1["recomputed"])
        r2 = jr.evaluate_retelling(stories, r1, rnn_state=st)     # identical model
        self.assertFalse(r2["recomputed"])
        self.assertFalse(r2["rnn_model_changed"])
        self.assertEqual(r2["order_gain"], r1["order_gain"])
        st2 = dict(st); st2["model_fingerprint"], st2["steps_trained"] = "fp-2", 700_000
        r3 = jr.evaluate_retelling(stories, r2, rnn_state=st2)    # +200k steps
        self.assertTrue(r3["recomputed"])

    def test_a_tiny_step_change_does_not_force_a_full_reeval(self):
        stories = folktale_stories(180)
        st = self._rnn(); st["model_fingerprint"], st["steps_trained"] = "fp-a", 100000
        r1 = jr.evaluate_retelling(stories, {}, rnn_state=st)
        st2 = dict(st); st2["model_fingerprint"], st2["steps_trained"] = "fp-b", 101000  # +1%
        r2 = jr.evaluate_retelling(stories, r1, rnn_state=st2)
        self.assertFalse(r2["recomputed"])
        self.assertTrue(r2["rnn_model_changed"])                 # changed, but not re-evaluated
        self.assertEqual(r2["rnn_steps_now"], 101000)

    def test_snapshot_fingerprint_depends_on_event_content_not_just_urls(self):
        a = [{"url": "http://x/1", "events": [{"subject": "a", "verb": "b", "obj": "c"}]}]
        b = [{"url": "http://x/1", "events": [{"subject": "a", "verb": "ZZ", "obj": "c"}]}]
        self.assertNotEqual(jr._fingerprint(a), jr._fingerprint(b))

    def test_eval_regime_change_resets_the_streak(self):
        stories = folktale_stories(180)
        old = {"eval_regime": "narrative_order_recovery_v1",   # a superseded regime
               "significant_streak": 2, "beats_baseline": True,
               "beats_baseline_significant": True, "selection_significant_streak": 2,
               "selection_last_significant_train": 40, "final_history": [{"tier": "final"}],
               "learning_curve": [{"train_stories": 40}],
               "test_snapshot": [{"url": s["url"], "events": s["events"]}
                                 for s in stories[:30]]}
        r = jr.evaluate_retelling(stories, old, rnn_state=self._rnn())
        self.assertEqual(r["eval_regime"], jr.EVAL_REGIME)
        self.assertEqual(r["regime_reset_from"], "narrative_order_recovery_v1")
        self.assertFalse(r["beats_baseline"])
        self.assertEqual(r["final_opened_count"], 0)          # old finals not inherited
        self.assertLessEqual(r["selection_significant_streak"], 1)

    def test_insufficient_stories_reports_cleanly(self):
        report = jr.evaluate_retelling(folktale_stories(n=10), {})
        self.assertEqual(report["status"], "insufficient_selection_stories")
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

    def test_selection_alone_is_never_a_capability_claim(self):
        stories = folktale_stories(220)
        r = jr.evaluate_retelling(stories, {}, rnn_state=self._rnn())
        self.assertEqual(r["status"], "measured")
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)
        for _ in range(3):
            r = jr.evaluate_retelling(stories, r, rnn_state=self._rnn())
        self.assertFalse(r["capability_confirmed"])           # selection is diagnostic only

    def test_final_needs_selection_significance_which_an_untrained_rnn_lacks(self):
        stories = folktale_stories(260)
        r = jr.evaluate_retelling(stories, {}, rnn_state=self._rnn())
        # untrained RNN -> selection not significant -> no candidate, no final
        self.assertFalse((r.get("selection") or {}).get("significant"))
        self.assertEqual(r["final_status"], "awaiting_selection_streak_2")
        self.assertFalse(r["capability_confirmed"])
        self.assertFalse(r["capability_confirmed_ever"])
        self.assertEqual(r["final_opened_count"], 0)

    def test_train_selection_final_are_collection_disjoint(self):
        r = jr.evaluate_retelling(folktale_stories(260), {}, rnn_state=self._rnn())
        self.assertTrue(r["collections_disjoint"], r["collection_disjointness"])

    def test_baseline_uses_exactly_the_rnn_training_sources(self):
        # re-audit #6 P1-6: pass the RNN's training URLs; the position baseline is
        # built from EXACTLY those, and a mismatch invalidates the measurement.
        stories = folktale_stories(260)
        rnn_urls = {s["url"] for s in stories[:120]}
        r = jr.evaluate_retelling(stories, {}, rnn_state=self._rnn(),
                                  rnn_training_urls=rnn_urls)
        self.assertEqual(r["rnn_training_url_count"], len(rnn_urls))
        self.assertTrue(r["baseline_corpus_matches_rnn"])
        self.assertIn("baseline_source_fingerprint", r)
        # a URL the RNN trained on that is missing from the corpus -> invalid
        bad = jr.evaluate_retelling(stories, {}, rnn_state=self._rnn(),
                                    rnn_training_urls=rnn_urls | {"http://not/in/corpus"})
        self.assertFalse(bad["baseline_corpus_matches_rnn"])
        self.assertEqual(bad["status"], "measurement_invalid")
        self.assertFalse(bad["capability_confirmed"])
        self.assertIn("baseline corpus", bad["capability_pending_reason"])

    def test_re_measuring_the_same_snapshot_twice_is_not_a_replication(self):
        stories = folktale_stories(140)
        state = self._rnn(); state["model_fingerprint"], state["steps_trained"] = "s", 500
        r1 = jr.evaluate_retelling(stories, {}, rnn_state=state)
        r2 = jr.evaluate_retelling(stories, r1, rnn_state=state)   # same data + model
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
