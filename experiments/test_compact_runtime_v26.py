import tempfile
import unittest
from pathlib import Path

from compact_runtime_v26 import compact_curriculum, compact_runtime, compact_state
from local_worker_v21 import read_json, write_json


class CompactRuntimeTest(unittest.TestCase):
    def test_keeps_seed_cycles_and_only_their_curiosity(self):
        state = {"seed": "one", "completed_gap_ids": ["word:one"],
                 "cycles": [{"gap": {"gap_id": "word:one"}, "result": {"learned": True}}],
                 "curiosity_ledger": {"word:one": {"pressure": 2},
                                      "word:global-copy": {"pressure": 9, "seed_encounters": {"x": 3}}}}
        compacted, counts = compact_state(state)
        self.assertEqual(list(compacted["curiosity_ledger"]), ["word:one"])
        self.assertEqual(compacted["cycles"][0]["result"], {"learned": True})
        self.assertEqual(counts["curiosity_before"], 2)

    def test_global_ledger_keeps_aggregates_without_per_seed_map(self):
        curriculum = {"curiosity_ledger": {"word:x": {
            "encounters": 7, "contexts_seen": 2, "seed_encounters": {"a": 3, "b": 4}}}}
        compacted, counts = compact_curriculum(curriculum)
        self.assertEqual(compacted["curiosity_ledger"]["word:x"]["encounters"], 7)
        self.assertNotIn("seed_encounters", compacted["curiosity_ledger"]["word:x"])
        self.assertEqual(counts["removed_seed_links"], 2)

    def test_apply_preserves_state_and_report_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            state = {"seed": "one", "completed_gap_ids": [], "cycles": [],
                     "curiosity_ledger": {"copied": {"pressure": 1}}}
            write_json(runtime / "controller-state.json", state)
            write_json(runtime / "latest-report.json", {"state": state, "knowledge": {"kept": True}})
            write_json(runtime / "curriculum-state.json", {"curiosity_ledger": {},
                                                             "completed_seeds": [], "mastery_history": []})
            summary = compact_runtime(runtime, True)
            self.assertEqual(summary["status"], "applied")
            self.assertEqual(read_json(runtime / "latest-report.json")["knowledge"], {"kept": True})
            self.assertEqual(read_json(runtime / "controller-state.json")["seed"], "one")


if __name__ == "__main__":
    unittest.main()
