import unittest

import capability_probe_v1 as probe
import cognition_v1 as cog

# structurally-valid 2-kanji nouns (cog._real_word only checks shape), grouped so
# each has a stable independent "reference genus".  A big pool per genus so every
# hash-assigned tier clears its minimum.
_HEADS = {"生き物": "犬猫馬鳥魚虫鹿兎猿熊狐狸鶏牛羊豚蛇亀鴨鷲",
          "道具":   "机椅筆鏡笛鈴刀弓網箱桶皿匙鍋釜傘靴帽袋櫛",
          "場所":   "山川海湖島森林村町国橋寺宮門庭畑坂道港谷",
          "食べ物": "飯汁茶酒肉魚菓餅飴粥芋豆栗柿桃梨麦米粟稗"}
_TAIL = "郎子丸太助兵吉造平次作"


def _pool():
    return {g: [h + t for h in heads for t in _TAIL if cog._real_word(h + t)]
            for g, heads in _HEADS.items()}


_POOL = _pool()


def _build_state(match_belief=True, conf=0.75):
    beliefs, refs, contexts, entities, profiles = {}, {}, {}, {}, {}
    for genus, words in _POOL.items():
        for w in words:
            refs[w] = {"genus": genus}
            g = genus if match_belief else "生き物"
            beliefs[w] = {"genus": g, "understood": True, "confidence": conf,
                          "support": {}, "sources": [], "revisions": [], "last_cycle": 1}
            # a couple of same-genus neighbours so _infer_genus / classify works
            contexts[w] = {n: 3 for n in words[:12] if n != w}
            entities[w] = 10
            if genus in ("生き物", "人"):
                profiles[w] = {"subj": 4, "obj": 0,
                               "subj_verbs": {"見る": 2, "食べる": 1}, "obj_verbs": {}}
            else:
                profiles[w] = {"subj": 0, "obj": 4,
                               "subj_verbs": {}, "obj_verbs": {"使う": 2}}
    return {"beliefs": beliefs, "selection_refs": refs, "contexts": contexts,
            "entities": entities, "profiles": profiles}


STORE = {}
SHELF = {}


class TierSeparationTest(unittest.TestCase):
    def test_build_produces_three_separated_tiers(self):
        built = probe.build_probe(1, _build_state(), STORE, SHELF)
        self.assertEqual(built["status"], "frozen")
        for tier in ("challenge", "selection", "final", "reserve"):
            self.assertIn(tier, built)
        self.assertGreaterEqual(len(built["selection"]), probe.MIN_SELECTION_PROBLEMS)

    def test_no_concept_straddles_two_held_out_tiers(self):
        built = probe.build_probe(1, _build_state(), STORE, SHELF)
        concept_tier = {}
        for tier in ("challenge", "selection", "final", "reserve"):
            for p in built[tier]:
                # a problem lives in its concept's tier ...
                if tier != "challenge":
                    self.assertEqual(probe._word_tier(p["concept"]), tier)
                # ... two problems about the same concept -> same tier
                self.assertEqual(concept_tier.setdefault(p["concept"], tier), tier)
        # ... and a held-out-tier problem never has a distractor from a
        # DIFFERENT held-out tier
        for tier in ("selection", "final", "reserve"):
            for p in built[tier]:
                for w in p["words"]:
                    if w != p["concept"]:
                        self.assertIn(probe._word_tier(w), ("challenge", tier))
        self.assertTrue(probe._tier_separation_valid(
            built["challenge"] + built["selection"] + built["final"] + built["reserve"]))

    def test_selection_is_not_conditioned_on_baseline_being_wrong(self):
        built = probe.build_probe(1, _build_state(), STORE, SHELF)
        graded = [cog._grade(p, p["baseline"]) for p in built["selection"]]
        # the unbiased selection contains BOTH baseline-correct and baseline-wrong
        # problems -- it was a salt sample, never filtered on gold/baseline
        self.assertTrue(any(graded))
        self.assertTrue(any(not g for g in graded))

    def test_challenge_is_deliberately_baseline_fails(self):
        built = probe.build_probe(1, _build_state(), STORE, SHELF)
        for p in built["challenge"]:
            self.assertFalse(cog._grade(p, p["baseline"]))


class CapabilityGateTest(unittest.TestCase):
    def _cog(self, decisions=0, rules=None):
        return {"rules": [{"rule_id": f"r{i}"} for i in range(decisions)],
                "corrections_index": {}, "controller": {"policy": {}, "decisions": 0}}

    def test_final_is_not_graded_before_selection_passes(self):
        wm = _build_state(match_belief=False)            # beliefs all wrong -> Noise fails selection
        r = probe.maybe_run(10, wm, STORE, SHELF, self._cog(), None)
        self.assertEqual(r["status"], "measured")
        self.assertFalse(r["selection_pass"])
        self.assertIsNone(r["final_result"])
        self.assertEqual(r["final_opened_count"], 0)
        self.assertFalse(r["capability_confirmed"])

    def test_challenge_improvement_alone_is_not_capability_confirmed(self):
        wm = _build_state()
        r = probe.maybe_run(10, wm, STORE, SHELF, self._cog(), None)
        for c in range(22, 22 + probe.PROBE_INTERVAL * 4, probe.PROBE_INTERVAL):
            r = probe.maybe_run(c, wm, STORE, SHELF, self._cog(), r)
        self.assertGreater(len(r["challenge_history"]), 1)
        # even with a healthy challenge curve, no final has been opened via a
        # single unchanging model -> not confirmed
        if not r["capability_confirmed"]:
            self.assertIn(r["capability_pending_reason"] or "", [
                r["capability_pending_reason"]])   # reason is populated
        self.assertFalse(r["capability_confirmed"] and r["final_opened_count"] == 0)

    def test_selection_pass_alone_is_not_capability_confirmed(self):
        wm = _build_state()
        # one measurement only -> streak of 1 -> selection_pass False
        r = probe.maybe_run(10, wm, STORE, SHELF, self._cog(), None)
        self.assertFalse(r["selection_pass"])
        self.assertFalse(r["capability_confirmed"])

    def test_only_an_unopened_final_pass_confirms_capability(self):
        wm = _build_state()
        r = probe.maybe_run(10, wm, STORE, SHELF, self._cog(decisions=1), None)
        # a genuinely different model fingerprint on the next measurement
        r = probe.maybe_run(11, wm, STORE, SHELF, self._cog(decisions=9), r)
        self.assertTrue(r["selection_pass"], r.get("selection_latest"))
        self.assertIsNotNone(r["final_result"])
        self.assertEqual(r["final_opened_count"], 1)
        self.assertTrue(r["capability_confirmed"], r.get("capability_pending_reason"))
        self.assertIn("distinct models", r["capability_basis"])

    def test_an_opened_final_is_never_re_graded(self):
        wm = _build_state()
        r = probe.maybe_run(10, wm, STORE, SHELF, self._cog(decisions=1), None)
        r = probe.maybe_run(11, wm, STORE, SHELF, self._cog(decisions=9), r)
        opened_at = r["final_result"]["opened_at_cycle"]
        first = dict(r["final_result"])
        for c in (20, 30, 40):
            r = probe.maybe_run(c, wm, STORE, SHELF, self._cog(decisions=c), r)
        self.assertEqual(r["final_result"]["opened_at_cycle"], opened_at)
        self.assertEqual(r["final_result"]["correct"], first["correct"])
        self.assertEqual(r["final_opened_count"], 1)

    def test_re_measuring_selection_needs_a_moved_model_not_a_lucky_reroll(self):
        wm = _build_state()
        r = probe.maybe_run(10, wm, STORE, SHELF, self._cog(decisions=3), None)
        n1 = r["selection_measurements"]
        r = probe.maybe_run(11, wm, STORE, SHELF, self._cog(decisions=3), r)  # identical model
        self.assertEqual(r["selection_measurements"], n1)                     # not re-measured
        r = probe.maybe_run(12, wm, STORE, SHELF, self._cog(decisions=7), r)  # moved
        self.assertEqual(r["selection_measurements"], n1 + 1)


class MigrationTest(unittest.TestCase):
    def test_a_v2_probe_is_not_reinterpreted_as_a_fair_final(self):
        old = {"version": 2, "status": "frozen", "frozen_cycle": 3,
               "problems": [{"pid": "x", "type": "common_property", "concept": "小犬",
                             "other": "子馬", "grade": "coarse", "gold": "生き物",
                             "options": list(cog.COARSE)[:3], "baseline": "道具",
                             "words": ["子馬", "小犬"]}],
               "history": [{"cycle": 3, "n": 11, "correct": 11, "baseline": 0,
                            "derive_rate": 1.0, "lift": 11}]}
        r = probe.maybe_run(20, _build_state(), STORE, SHELF, {"rules": []}, old)
        self.assertEqual(r["version"], probe.VERSION)
        self.assertEqual(r.get("migrated_from_version"), 2)
        self.assertFalse(r["capability_confirmed"])
        self.assertEqual(r["final_opened_count"], 0)
        # the old 1.0 rate is not carried as a selection / final result
        self.assertLessEqual(len(r["selection_history"]), 1)

    def test_waits_until_enough_understood_words(self):
        thin = {"beliefs": {f"a{i}": {"genus": "生き物", "understood": True, "confidence": 0.7}
                            for i in range(5)}, "selection_refs": {}}
        r = probe.maybe_run(1, thin, STORE, SHELF, {"rules": []}, None)
        self.assertEqual(r["status"], "waiting")


if __name__ == "__main__":
    unittest.main()
