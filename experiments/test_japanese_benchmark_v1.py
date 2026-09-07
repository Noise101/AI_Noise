import unittest

import japanese_benchmark_v1 as jb


def stories(n, salt_prefix="col"):
    return [{"url": f"https://ja.wikisource.org/wiki/{salt_prefix}{i}/story{i}",
             "events": [{"subject": "a", "verb": f"v{j}", "obj": ""} for j in range(4)]}
            for i in range(n)]


class BenchmarkTest(unittest.TestCase):
    def test_tiers_are_collection_disjoint(self):
        t = jb.Tiers(stories(200), "salt:x", {})
        self.assertTrue(t.selection_frozen)
        self.assertTrue(t.disjoint, t.disjointness)
        for pair in t.disjointness.values():
            self.assertEqual(pair, [])

    def test_a_collection_keeps_its_tier_as_the_corpus_grows(self):
        t1 = jb.Tiers(stories(120), "salt:y", {})
        prev = {"selection_snapshot": t1.selection_snapshot}
        t2 = jb.Tiers(stories(240), "salt:y", prev)
        self.assertEqual([s["url"] for s in t2.selection_snapshot],
                         [s["url"] for s in t1.selection_snapshot])   # frozen, not re-drawn

    def test_fingerprint_depends_on_event_content(self):
        a = [{"url": "u", "events": [{"subject": "s", "verb": "v", "obj": "o"}]}]
        b = [{"url": "u", "events": [{"subject": "s", "verb": "CHANGED", "obj": "o"}]}]
        self.assertNotEqual(jb.fingerprint(a), jb.fingerprint(b))

    def test_next_unopened_final_walks_final_then_reserve_then_none(self):
        t = jb.Tiers(stories(400), "salt:z", {})
        first = t.next_unopened_final()
        self.assertIsNotNone(first)
        self.assertEqual(first["tier"], "final")
        t.record_final(first)
        second = t.next_unopened_final()
        if second is not None:
            self.assertEqual(second["tier"], "reserve")
            t.record_final(second)
        self.assertIsNone(t.next_unopened_final())

    def test_forbidden_collections_covers_every_non_train_tier(self):
        t = jb.Tiers(stories(200), "salt:f", {})
        train_cols = {jb.collection(s["url"]) for s in t.train_stories}
        self.assertFalse(train_cols & t.forbidden_train_collections)

    def test_train_tier_is_fully_persisted_and_survives_reload(self):
        # re-audit #6 P1-2: the ledger records EVERY collection, train included
        t1 = jb.Tiers(stories(200), "salt:p", {}, cycle=1)
        train_cols_1 = {c for c, v in t1.tier_assignments.items() if v["tier"] == "train"}
        self.assertGreater(len(train_cols_1), 0)
        self.assertTrue(all(isinstance(v, dict) and "assigned_at" in v
                            for v in t1.tier_assignments.values()))
        prev = {"tier_assignments": t1.tier_assignments,
                "selection_snapshot": t1.selection_snapshot}
        t2 = jb.Tiers(stories(260), "salt:p", prev, cycle=2)
        train_cols_2 = {c for c, v in t2.tier_assignments.items() if v["tier"] == "train"}
        self.assertTrue(train_cols_1 <= train_cols_2)          # no train collection lost

    def test_topup_only_touches_fresh_untrained_collections(self):
        # a previously-trained collection is NEVER moved into a held-out tier,
        # even if a held-out tier is short of its story minimum
        st = stories(60)
        et = {jb.collection(s["url"]) for s in st[:40]}        # 40 collections "ever trained"
        t = jb.Tiers(st, "salt:tu", {}, cycle=1, ever_trained_collections=et)
        for col in et:
            self.assertEqual(t.tier_assignments.get(col, {}).get("tier"), "train")
        moved = [c for c, v in t.tier_assignments.items()
                 if v.get("assignment_reason") == "topup_fresh_untrained"]
        self.assertFalse(set(moved) & et)

    def test_streak_anchor_reaches_2_only_with_growth_and_a_new_model(self):
        s = jb.advance_selection_streak(None, True, 100, "m0")
        self.assertEqual(s["streak"], 1)
        self.assertEqual(s["anchor_train"], 100)
        s = jb.advance_selection_streak(s, True, 120, "m0")     # not 1.4x yet
        self.assertEqual(s["streak"], 1)
        self.assertEqual(s["anchor_train"], 100)               # anchor NOT re-stamped
        s = jb.advance_selection_streak(s, True, 141, "m0")     # 1.4x but SAME model
        self.assertEqual(s["streak"], 1)
        s = jb.advance_selection_streak(s, True, 141, "m2")     # 1.4x AND new model
        self.assertEqual(s["streak"], 2)
        self.assertEqual(s["anchor_train"], 141)               # anchor moves forward
        s = jb.advance_selection_streak(s, False, 400, "m3")    # non-significant resets
        self.assertEqual(s["streak"], 0)
        self.assertIsNone(s["anchor_train"])

    def test_a_new_candidate_needs_growth_beyond_the_last_one(self):
        cands = [{"registered_at_train": 100}]
        self.assertFalse(jb.should_register_candidate(cands, 2, 120))   # only +20%
        self.assertTrue(jb.should_register_candidate(cands, 2, 160))    # +60% >= 1.5x
        self.assertFalse(jb.should_register_candidate(cands, 1, 300))   # streak < 2
        self.assertFalse(jb.should_register_candidate(cands * 2, 2, 999))  # budget spent


if __name__ == "__main__":
    unittest.main()
