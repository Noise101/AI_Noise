import tempfile
import unittest
from pathlib import Path

from compact_runtime_v26 import (compact_curriculum, compact_historical_seed_reports,
                                 compact_runtime, compact_seed_report, compact_state,
                                 rebuild_global_curiosity)
from local_worker_v21 import read_json, write_json


class CompactRuntimeTest(unittest.TestCase):
    def test_historical_report_keeps_evidence_and_drops_copied_priors(self):
        report = {"state": {"seed": "fox", "cycles": [{"large": True}]}, "knowledge": {
            "bootstrap": {"sources": [{"url": "u", "event_extraction_audit": [
                {"sentence": "Fox ran.", "accepted": True, "event": "fox|ran|"}]}],
                "parallel_learning": {"copied": "x" * 1000}},
            "lexicon": {"word_forms": {"fox": 2},
                "grounded_meanings": [{"form": "fox", "roles": {"subject": 1}}],
                "phrase_candidates": [], "characters": {}, "conversation_cues": {},
                "researched_meanings": {"fox": {"accepted_sense": "animal"},
                    "unrelated": {"accepted_sense": "copied global belief"}},
                "researched_phrase_meanings": {}, "researched_conversation_acts": {}},
            "story": {"rules": [{"when": "a", "expect": "b"}]},
            "concepts": {"beliefs": [{"subject": "fox"}]}}}
        compacted, result = compact_seed_report(report)
        self.assertFalse(result["already_compacted"])
        self.assertNotIn("parallel_learning", compacted["knowledge"]["bootstrap"])
        self.assertEqual(list(compacted["knowledge"]["lexicon"]["researched_meanings"]), ["fox"])
        self.assertEqual(compacted["knowledge"]["story"]["rules"][0]["expect"], "b")
        self.assertEqual(compacted["knowledge"]["bootstrap"]["sources"][0][
            "event_extraction_audit"][0]["sentence"], "Fox ran.")

    def test_incremental_compaction_skips_current_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            for identity, seed in (("one", "fox"), ("two", "owl")):
                path = runtime / "seeds" / identity / "latest-report.json"
                path.parent.mkdir(parents=True)
                write_json(path, {"state": {"seed": seed}, "knowledge": {
                    "bootstrap": {"parallel_learning": {"copy": "x" * 1000}, "sources": []},
                    "lexicon": {}}})
            result = compact_historical_seed_reports(runtime, "fox", max_files=10)
            self.assertEqual(result["processed_reports"], 1)
            self.assertNotIn("storage_compaction", read_json(
                runtime / "seeds" / "one" / "latest-report.json"))
            self.assertIn("storage_compaction", read_json(
                runtime / "seeds" / "two" / "latest-report.json"))

    def test_keeps_seed_cycles_and_only_their_curiosity(self):
        state = {"seed": "one", "completed_gap_ids": ["word:one"],
                 "cycles": [{"gap": {"gap_id": "word:one"}, "result": {"learned": True}}],
                 "curiosity_ledger": {"word:one": {"pressure": 2},
                                      "word:global-copy": {"pressure": 9, "seed_encounters": {"x": 3}}}}
        compacted, counts = compact_state(state)
        self.assertEqual(list(compacted["curiosity_ledger"]), ["word:one"])
        self.assertEqual(compacted["cycles"][0]["result"], {"learned": True})
        self.assertEqual(counts["curiosity_before"], 2)
        self.assertEqual(compacted["curiosity_ledger"]["word:one"]["encounters"], 1)

    def test_rebuilds_real_counts_instead_of_propagating_corrupt_global_counts(self):
        one, _ = compact_state({"seed": "one", "cycles": [{
            "gap": {"gap_id": "word:x", "layer": "word", "query": "x", "observations": 2},
            "grounded": False}], "curiosity_ledger": {"word:x": {"encounters": 10 ** 50}}})
        two, _ = compact_state({"seed": "two", "cycles": [{
            "gap": {"gap_id": "word:x", "layer": "word", "query": "x", "observations": 3},
            "grounded": False}], "curiosity_ledger": {"word:x": {"encounters": 10 ** 60}}})
        global_ledger = rebuild_global_curiosity([one, two], 2)
        self.assertEqual(global_ledger["word:x"]["encounters"], 5)
        self.assertEqual(global_ledger["word:x"]["contexts_seen"], 2)

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
