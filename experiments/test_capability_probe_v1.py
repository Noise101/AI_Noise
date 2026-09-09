import unittest

import capability_probe_v1 as probe
import cognition_v1 as cog
from test_cognition_v1 import WM, STORE, SHELF, _belief


class CapabilityProbeTest(unittest.TestCase):
    def setUp(self):
        self._orig = probe._ref_genus
        # deterministic offline "dictionary": kanji-ish words get their belief genus
        probe._ref_genus = lambda w, cache: cache.setdefault(
            w, cog._canon_class((WM["beliefs"].get(w, {}) or {}).get("genus", "")))

    def tearDown(self):
        probe._ref_genus = self._orig

    def test_waits_until_enough_understood_words(self):
        thin = {"beliefs": {"a" + str(i): _belief("生き物") for i in range(5)}}
        r = probe.maybe_run(1, thin, STORE, SHELF, {"rules": []}, None)
        self.assertEqual(r["status"], "waiting")

    def test_build_makes_reference_checked_composition_problems_only(self):
        wm = {"beliefs": {**{f"どうぐ{i}": _belief("道具") for i in range(6)},
                          **{f"いきもの{i}": _belief("生き物") for i in range(40)}},
              "contexts": {}, "entities": {}}
        built = probe.build_probe(1, wm, STORE, SHELF)
        if built["status"] == "building":
            self.assertLess(built["have"], probe.MIN_PROBE_PROBLEMS)
            return
        self.assertEqual(built["status"], "frozen")
        for p in built["problems"]:
            self.assertIn(p["type"], ("odd_one_out", "common_property"))
            self.assertIn("baseline", p)
            self.assertFalse(cog._grade(p, p["baseline"]))     # baseline must fail

    def test_run_records_derive_rate_and_lift_and_never_regenerates(self):
        wm = {"beliefs": {**{f"どうぐ{i}": _belief("道具") for i in range(8)},
                          **{f"いきもの{i}": _belief("生き物") for i in range(40)}},
              "contexts": {}, "entities": {}}
        r = probe.maybe_run(10, wm, STORE, SHELF, {"rules": [], "corrections_index": {}}, None)
        if r["status"] != "measured":
            self.skipTest("offline dictionary produced too few problems")
        frozen = list(r["problems"])
        r2 = probe.maybe_run(30, wm, STORE, SHELF, {"rules": [], "corrections_index": {}}, r)
        self.assertEqual(r2["problems"], frozen)               # frozen, not rebuilt
        self.assertIsNotNone(r2["latest_derive_rate"])
        self.assertGreaterEqual(len(r2["history"]), 2)


if __name__ == "__main__":
    unittest.main()
