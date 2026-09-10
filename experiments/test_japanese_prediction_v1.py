import random
import unittest

import japanese_prediction_v1 as jp
import japanese_word_meaning_v1 as wmn


def _b(genus, conf=0.75):
    return {"genus": genus, "understood": True, "confidence": conf,
            "support": {"testimony": 0.1, "evidence": 0.5}, "sources": ["reading_usage"],
            "revisions": [], "last_cycle": 1}


# a world where the genus of the arguments really constrains the event:
#   生き物 subjects do mental / motion / ingest;  they ingest 食べ物;
#   道具 subjects only change;  場所 subjects only change/other
WM = {"beliefs": {**{f"けもの{i}": _b("生き物") for i in range(14)},
                  **{f"たべもの{i}": _b("食べ物") for i in range(10)},
                  **{f"どうぐ{i}": _b("道具") for i in range(10)},
                  **{f"ばしょ{i}": _b("場所") for i in range(10)}}}

_CREATURE_V = ["見る", "言う", "思う", "走る", "食べる"]
_TOOL_V = ["こわれる", "ひらく", "かわる"]
_PLACE_V = ["過ぎる", "かかる"]


def _ev(s, v, o=""):
    return {"subject": s, "verb": v, "obj": o, "provenance": "heuristic_self"}


def structured_stories(n=220, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        evs = []
        for _ in range(7):
            kind = rng.choice(["けもの", "けもの", "どうぐ", "ばしょ"])
            w = f"{kind}{rng.randint(0, 9)}"
            if kind == "けもの":
                v = rng.choice(_CREATURE_V)
                o = f"たべもの{rng.randint(0,9)}" if v == "食べる" else ""
            elif kind == "どうぐ":
                v, o = rng.choice(_TOOL_V), ""
            else:
                v, o = rng.choice(_PLACE_V), ""
            evs.append(_ev(w, v, o))
        out.append({"url": f"http://s/{i}", "events": evs})
    return out


def shuffled_stories(n=220, seed=1):
    base = structured_stories(n, seed)
    rng = random.Random(seed + 99)
    verbs = [e["verb"] for s in base for e in s["events"]]
    rng.shuffle(verbs)
    it = iter(verbs)
    for s in base:
        for e in s["events"]:
            e["verb"] = next(it)
            e["obj"] = ""
    return base


class VerbClassTest(unittest.TestCase):
    def test_no_stem_collisions(self):
        self.assertEqual(jp.verb_class("こわれる"), "change")   # not mental (こわがる)
        self.assertEqual(jp.verb_class("ある"), "")             # not motion (あるく)
        self.assertEqual(jp.verb_class("言う"), "mental")
        self.assertEqual(jp.verb_class("食べる"), "ingest")
        self.assertEqual(jp.verb_class("死ぬ"), "change")


class SignalTest(unittest.TestCase):
    def test_concept_plausibility_beats_the_frequency_baseline_when_signal_exists(self):
        r = jp.run_prediction(1, structured_stories(260), [], WM["beliefs"], [], {})
        self.assertEqual(r["status"], "measured")
        sel = r["selection"]
        self.assertIsNotNone(sel)
        self.assertGreater(sel["model_accuracy"], sel["baseline_accuracy"])
        self.assertGreater(sel["discrimination_gain"], 0.05)
        self.assertGreaterEqual(sel["gain_z"], jp.SIGNIFICANCE_Z)

    def test_no_signal_on_shuffled_events(self):
        r = jp.run_prediction(1, shuffled_stories(260), [], WM["beliefs"], [], {})
        sel = r["selection"]
        self.assertLess(sel["discrimination_gain"], 0.05)
        self.assertFalse(sel["significant"])
        self.assertFalse(r["capability_confirmed"])

    def test_next_verb_class_is_only_a_secondary_diagnostic(self):
        r = jp.run_prediction(1, structured_stories(260), [], WM["beliefs"], [], {})
        d = r["secondary_diagnostic_next_verb_class"]
        self.assertIn("note", d)
        self.assertIn("NOT a capability", d["note"])
        # it must not feed capability_confirmed
        self.assertFalse(r["capability_confirmed"])

    def test_only_heuristic_self_events_are_scored(self):
        s = structured_stories(30)
        s[0]["events"].append({"subject": "けもの0", "verb": "食べる", "obj": "どうぐ0",
                               "provenance": "local_llm_scaffold"})   # implausible + not self
        ts = jp._triples(s[0]["events"], WM["beliefs"])
        self.assertTrue(all(not (t["subj"] == "けもの0" and t["og"] == "道具") for t in ts))


class CapabilityGateTest(unittest.TestCase):
    def test_selection_alone_is_not_confirmed(self):
        r = jp.run_prediction(1, structured_stories(260), [], WM["beliefs"], [], {})
        self.assertTrue((r["selection"] or {}).get("significant"))
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)

    def test_selection_snapshot_is_frozen_across_growth(self):
        r1 = jp.run_prediction(1, structured_stories(260), [], WM["beliefs"], [], {})
        fp = r1["selection_fingerprint"]
        r2 = jp.run_prediction(2, structured_stories(260) + structured_stories(160, seed=7),
                               [], WM["beliefs"], [], r1)
        self.assertEqual(r2["selection_fingerprint"], fp)

    def test_capability_only_via_an_unopened_final(self):
        r = jp.run_prediction(1, structured_stories(320), [], WM["beliefs"], [], {})
        for c in range(2, 9):
            r = jp.run_prediction(c, structured_stories(320 + c * 140, seed=c),
                                  [], WM["beliefs"], [{"x": c}], r)
        if r["capability_confirmed"]:
            self.assertGreaterEqual(r["final_opened_count"], 1)
            self.assertTrue((r["final_result"] or {}).get("significant"))
        else:
            self.assertIn(r["final_status"], ("awaiting_selection_streak_2",
                          "registerable_next_cycle", "final_below_threshold",
                          "candidate_registered_final_snapshot_unavailable"))

    def test_regime_reset_does_not_inherit_a_pass(self):
        old = {"eval_regime": "old_regime", "capability_confirmed": True,
               "candidate_checkpoints": [{"final_result": {"significant": True}, "tier": "final"}],
               "selection_significant_streak": 2}
        r = jp.run_prediction(1, structured_stories(260), [], WM["beliefs"], [], old)
        self.assertEqual(r["regime_reset_from"], "old_regime")
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)


class FeedbackTest(unittest.TestCase):
    def _misses(self, cycles):
        r = {}
        for c in range(1, cycles + 1):
            book = ([_ev("けもの5", "見る")]
                    + [_ev("けもの5", "こわれる") for _ in range(4)])   # a "creature" that only breaks
            r = jp.run_prediction(c, structured_stories(260), book, WM["beliefs"], [], r)
        return r

    def test_persistent_implausible_real_events_produce_capped_feedback(self):
        r = self._misses(12)
        fb = r["prediction_feedback"]
        self.assertIn("けもの5", fb)
        self.assertEqual(fb["けもの5"]["against"], "生き物")
        self.assertGreater(fb["けもの5"]["strength"], 0)
        self.assertLessEqual(fb["けもの5"]["strength"], jp.FEEDBACK_MAX_PENALTY)

    def test_a_one_off_does_not_produce_feedback(self):
        self.assertNotIn("けもの5", self._misses(1)["prediction_feedback"])

    def test_word_meaning_uses_the_penalty_as_one_capped_channel(self):
        st = wmn._blank()
        st["contexts"]["椅子"] = {"人": 3}
        st["taxonomy"]["椅子"] = "動物"                        # bogus testimony -> 生き物
        st["profiles"]["椅子"] = {"subj": 1, "obj": 4, "subj_verbs": {},
                                  "obj_verbs": {"こわす": 2, "つかう": 2}}
        st["prediction_feedback"] = {}
        b0 = wmn._revise_belief(st, "椅子", 5)
        st["prediction_feedback"] = {"椅子": {"against": b0["genus"], "strength": 0.3}}
        b1 = wmn._revise_belief(st, "椅子", 6)
        self.assertTrue(b1["genus"] != b0["genus"]
                        or b1["confidence"] < b0["confidence"]
                        or (b0["understood"] and not b1["understood"])
                        or "prediction_miss" in b1["sources"])

    def test_a_held_out_word_never_takes_the_penalty(self):
        st = wmn._blank()
        st["contexts"]["机"] = {"人": 3}
        st["profiles"]["机"] = {"subj": 0, "obj": 5, "subj_verbs": {},
                                "obj_verbs": {"つかう": 3, "はこぶ": 2}}
        st["prediction_feedback"] = {"机": {"against": "道具", "strength": 0.3}}
        b = wmn._revise_belief(st, "机", 5, suppress_testimony=True)
        self.assertNotIn("prediction_miss", b["sources"])


if __name__ == "__main__":
    unittest.main()
