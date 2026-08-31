import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_worker_v21 import (_seed_from_title, developmental_source_quality, discover_curriculum,
                              discover_from_developmental_shelves, enforce_storage_budget, is_transient_error,
                              compact_learning_history, merge_curiosity, read_json, status_record,
                              work, write_json)


class LocalWorkerTest(unittest.TestCase):
    def test_storage_guard_compacts_redundant_curiosity_over_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            state = {"seed": "one", "cycles": [], "completed_gap_ids": [],
                     "curiosity_ledger": {"copied": {"pressure": 2, "padding": "x" * 5000}}}
            write_json(runtime / "controller-state.json", state)
            write_json(runtime / "latest-report.json", {"state": state})
            write_json(runtime / "curriculum-state.json", {"curiosity_ledger": {},
                                                             "completed_seeds": [], "mastery_history": []})
            result = enforce_storage_budget(runtime, 1000)
            self.assertTrue(result["compacted"])
            self.assertLess(result["after_bytes"], result["before_bytes"])
    def test_classifies_network_timeout_but_not_programming_error(self):
        self.assertTrue(is_transient_error(TimeoutError("read timed out")))
        self.assertTrue(is_transient_error(RuntimeError("IncompleteRead(100 bytes read)")))
        self.assertFalse(is_transient_error(RuntimeError("HTTP Error 403: Forbidden")))
        self.assertFalse(is_transient_error(KeyError("broken schema")))

    def test_atomic_status_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            write_json(path, {"phase": "learning"})
            self.assertEqual(read_json(path), {"phase": "learning"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_status_exposes_falsifiable_causal_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_json(runtime / "causal-memory.json", {
                "supported_hypotheses": 0,
                "evaluation": {"accuracy": 0.1, "baseline_accuracy": 0.1},
                "limitations": ["event parser is still shallow"],
            })
            status = status_record("seed", runtime, "learning", 1)
            self.assertEqual(status["causal_evaluation"]["supported_hypotheses"], 0)
            self.assertEqual(status["causal_evaluation"]["evaluation"]["accuracy"], 0.1)

    @patch("local_worker_v21.discover_from_developmental_shelves", return_value=[])
    @patch("local_worker_v21.rediscover_from_history", return_value=[])
    @patch("local_worker_v21.discover_curriculum", return_value=[])
    @patch("local_worker_v21.run_cycle")
    def test_repeats_step_budgets_then_exhausts_frontier(
            self, run_cycle, _discover, _history, _shelves):
        run_cycle.side_effect = [
            {"state": {"completed_gap_ids": ["one"], "stop_reason": "step_budget_exhausted"},
             "current_gaps": [{"gap_id": "two"}], "web_usage": {"network_requests": 1}},
            {"state": {"completed_gap_ids": ["one", "two"],
                       "stop_reason": "no_unresolved_executable_gap"},
             "current_gaps": [], "web_usage": {"network_requests": 0}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = work("seed", Path(directory), 5, 0, 1, 1, 2, local_conversation=False)
            self.assertEqual(run_cycle.call_count, 2)
            self.assertEqual(result["phase"], "curriculum_exhausted")
            self.assertEqual(result["completed_gaps"], 2)
            self.assertEqual(result["codex_or_remote_llm_calls"], 0)

    @patch("local_worker_v21.discover_curriculum")
    @patch("local_worker_v21.run_cycle")
    def test_selects_a_new_seed_without_another_manual_run(self, run_cycle, discover):
        discover.return_value = [{"seed": "fox crow", "score": 3,
                                  "reason": "linked", "parent_url": "source"}]
        run_cycle.side_effect = [
            {"state": {"completed_gap_ids": ["one"],
                       "stop_reason": "no_unresolved_executable_gap"},
             "current_gaps": [], "knowledge": {}, "web_usage": {}},
            {"state": {"completed_gap_ids": [], "stop_reason": "network_budget_exhausted"},
             "current_gaps": [], "knowledge": {}, "web_usage": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            result = work("fox grapes", runtime, 2, 0, 1, 1, 1, local_conversation=False)
            self.assertEqual(run_cycle.call_args_list[1].args[0], "fox crow")
            curriculum = read_json(runtime / "curriculum-state.json")
            self.assertEqual(curriculum["current_seed"], "fox crow")
            self.assertEqual(result["phase"], "round_budget_exhausted")
            self.assertEqual(result["seed"], "fox crow")

    def test_derives_seed_from_an_observed_story_link(self):
        self.assertEqual(_seed_from_title("Three Hundred Æsop's Fables/The Fox and the Crow"),
                         "fox crow")

    def test_rejects_metadata_as_a_curriculum_seed(self):
        self.assertIsNone(_seed_from_title("Ivory Carving: Historical Notes"))
        self.assertIsNone(_seed_from_title("Book VII"))

    @patch("local_worker_v21.WEB_CACHE.get_json", return_value={})
    def test_repeated_japanese_chunks_can_enter_the_same_curriculum(self, _get):
        report = {"state": {"seed": "きつね つる"}, "knowledge": {"lexicon": {"phrase_candidates": [
            {"phrase": "きつね", "kind": "unsegmented_chunk_candidate"},
            {"phrase": "つる", "kind": "unsegmented_chunk_candidate"}]}}}
        candidates = discover_curriculum(report, set(), 0)
        self.assertEqual(candidates[0]["seed"], "きつね つる")

    @patch("local_worker_v21.WEB_CACHE.get_json", return_value={})
    def test_function_word_concept_pair_cannot_become_a_seed(self, _get):
        report = {"knowledge": {"concepts": {"beliefs": [
            {"subject": "of", "object": "and", "citations": []}]}}}
        self.assertEqual(discover_curriculum(report, set(), 0), [])

    @patch("local_worker_v21.WEB_CACHE.get_json", return_value={})
    def test_rejected_source_cannot_spawn_concept_seed(self, _get):
        report = {"knowledge": {"bootstrap": {"sources": [{
            "event_extraction_audit": [{"accepted": False}] * 3}]},
            "concepts": {"beliefs": [{"subject": "became", "object": "moon"}]}}}
        self.assertEqual(discover_curriculum(report, set(), 0), [])

    @patch("local_worker_v21.WEB_CACHE.get_json", return_value={})
    def test_single_source_concept_cannot_become_curriculum(self, _get):
        report = {"knowledge": {"bootstrap": {"sources": [{
            "event_extraction_audit": [
                {"accepted": True, "event": "brain|said|right", "sentence": "Brain said right."},
                {"accepted": True, "event": "brain|said|night", "sentence": "Brain said night."},
            ]}]}, "concepts": {"beliefs": [{"subject": "brain", "object": "right",
                "status": "single_source", "accepted_polarity": True,
                "citations": ["https://one"]}]}}}
        self.assertEqual(discover_curriculum(report, set(), 0), [])

    @patch("local_worker_v21.WEB_CACHE.get_json", return_value={})
    def test_corroborated_concept_can_become_curriculum(self, _get):
        audit = [{"accepted": True, "event": "fox|saw|moon", "sentence": "Fox saw moon."},
                 {"accepted": True, "event": "fox|waited|moon", "sentence": "Fox waited moon."}]
        report = {"knowledge": {"bootstrap": {"sources": [
            {"event_extraction_audit": audit}]}, "concepts": {"beliefs": [{
                "subject": "fox", "object": "moon", "status": "corroborated",
                "accepted_polarity": True, "citations": ["https://one", "https://two"]}]}}}
        candidates = discover_curriculum(report, set(), 0)
        self.assertEqual(candidates[0]["seed"], "fox moon")
        self.assertEqual(candidates[0]["independent_sources"], 2)

    @patch("local_worker_v21.WEB_CACHE.get_json")
    def test_developmental_shelf_supplies_unread_titles_only(self, get_json):
        get_json.return_value = {"query": {"categorymembers": [
            {"title": "The Fox and the Crow"},
            {"title": "The Hare and the Tortoise"},
            {"title": "Index"},
        ]}}
        candidates = discover_from_developmental_shelves({"fox crow"}, 4)
        self.assertEqual([item["seed"] for item in candidates], ["hare tortoise"])
        self.assertTrue(all(item["reason"].startswith("unread title") for item in candidates))

    def test_compacts_mastery_history_without_losing_summary(self):
        curriculum = {"mastery_history": [{"overall_score": 0.2,
            "weakest_dimension": "words"} for _ in range(510)]}
        compact_learning_history(curriculum)
        self.assertEqual(len(curriculum["mastery_history"]), 500)
        self.assertEqual(curriculum["mastery_history_summary"]["records"], 10)

    def test_rejects_a_page_whose_text_is_mostly_not_narrative(self):
        report = {"knowledge": {"bootstrap": {"sources": [{
            "event_extraction_audit": [{"accepted": False}] * 4 + [{"accepted": True}]
        }]}}}
        self.assertEqual(developmental_source_quality(report)["status"], "outside_current_level")

    @patch("local_worker_v21.WEB_CACHE.get_json")
    def test_low_narrative_page_cannot_spawn_more_web_curricula(self, get_json):
        report = {"knowledge": {"bootstrap": {"sources": [{
            "url": "https://en.wikisource.org/wiki/Index_Page",
            "event_extraction_audit": [{"accepted": False}] * 3}]}}}
        self.assertEqual(discover_curriculum(report, set(), 4), [])
        get_json.assert_not_called()

    def test_same_unknown_across_curricula_builds_global_pressure(self):
        curriculum = {}
        def report(encounters):
            return {"state": {"curiosity_ledger": {"conversation:said": {
                "layer": "conversation", "query": "said dialogue", "encounters": encounters,
                "status": "wanting_to_know"}}}}
        merge_curiosity(curriculum, "story one", report(2), 1)
        first = curriculum["curiosity_ledger"]["conversation:said"]["pressure"]
        merge_curiosity(curriculum, "story two", report(3), 3)
        entry = curriculum["curiosity_ledger"]["conversation:said"]
        self.assertGreater(entry["pressure"], first)
        self.assertEqual(entry["contexts_seen"], 2)
        self.assertEqual(entry["encounters"], 5)

    @patch("local_worker_v21.run_cycle")
    def test_stop_file_prevents_a_cycle(self, run_cycle):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            # Simulate a stop appearing immediately after startup writes its status.
            original_write = write_json
            def request_stop(path, value):
                original_write(path, value)
                if value.get("phase") == "starting":
                    (runtime / "STOP").touch()
            with patch("local_worker_v21.write_json", side_effect=request_stop):
                result = work("seed", runtime, 2, 0, 1, 1, 1, local_conversation=False)
            run_cycle.assert_not_called()
            self.assertEqual(result["phase"], "stopped_by_user")

    @patch("local_worker_v21.wait_for_retry", return_value=True)
    @patch("local_worker_v21.run_cycle")
    def test_transient_timeout_retries_without_manual_restart(self, run_cycle, _wait):
        run_cycle.side_effect = [TimeoutError("read operation timed out"), {
            "state": {"completed_gap_ids": ["one"], "stop_reason": "step_budget_exhausted"},
            "current_gaps": [], "knowledge": {}, "web_usage": {}}]
        with tempfile.TemporaryDirectory() as directory:
            result = work("seed", Path(directory), 2, 0, 1, 1, 1, local_conversation=False)
        self.assertEqual(run_cycle.call_count, 2)
        self.assertEqual(result["phase"], "round_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
