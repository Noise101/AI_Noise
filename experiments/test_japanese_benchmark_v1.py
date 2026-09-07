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


if __name__ == "__main__":
    unittest.main()
