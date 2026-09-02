import json
import tempfile
import unittest
import urllib.parse
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from local_worker_v21 import (_seed_from_title, developmental_source_quality, discover_curriculum,
                              discover_from_developmental_shelves, enforce_storage_budget, is_transient_error,
                              compact_learning_history, merge_curiosity, read_json, render_human_status, status_record,
                              parser_counterexample_candidate, structural_counterexample_candidate,
                              repeated_grounding_candidate, curriculum_strategy_allowed,
                              learned_curriculum_score, supervise, update_autonomy_state,
                              update_curriculum_strategy, work, write_json)


class LocalWorkerTest(unittest.TestCase):
    def test_low_yield_curriculum_route_deprioritizes_itself(self):
        curriculum = {"transitions": [{"to": f"seed{i}", "reason": "broad shelf"}
                                       for i in range(10)]}
        for index in range(10):
            update_curriculum_strategy(curriculum, f"seed{index}", index == 0)
        self.assertFalse(curriculum_strategy_allowed(
            curriculum, {"reason": "broad shelf"}))
        self.assertTrue(curriculum_strategy_allowed(
            curriculum, {"reason": "targeted counterexample"}))

    def test_successful_curriculum_route_earns_higher_autonomous_priority(self):
        curriculum = {"strategy_performance": {
            "good": {"admitted": 8, "rejected": 2},
            "bad": {"admitted": 1, "rejected": 9}}}
        self.assertGreater(learned_curriculum_score(curriculum, {"reason": "good", "score": 2}),
                           learned_curriculum_score(curriculum, {"reason": "bad", "score": 2}))

    def test_autonomy_switches_to_counterexamples_when_more_tests_add_no_correct_prediction(self):
        curriculum = {}
        state = None
        for index in range(12):
            report = {"global_memory": {"curricula": 100 + index * 2},
                      "experience_revision": {"evaluation": {
                          "correct": 1, "total": 100 + index * 4, "coverage": 0.1},
                          "reusable_rules": 0, "failure_patterns": [{"pattern": "x"}]}}
            state = update_autonomy_state(curriculum, report)
        self.assertEqual(state["mode"], "counterexample_hunt")
        self.assertFalse(state["human_intervention_required"])

    def test_parser_failure_can_request_a_nearby_observation(self):
        audit = {"summary": {"rejection_reasons": {"invalid_structural_subject": 3}},
                 "records": {"x": {"audit_id": "x", "quarantined": True,
                    "curriculum_admitted": True,
                    "reason": "invalid_structural_subject", "sentence": "Through green woods birds flew.",
                    "source_url": "https://story"}}}
        candidate = parser_counterexample_candidate(audit, set())
        self.assertIn("green woods", candidate["seed"])
        self.assertEqual(candidate["parser_failure_reason"], "invalid_structural_subject")

    def test_repeated_structural_failure_can_drive_counterexample_search(self):
        report = {"summary": {"failure_patterns": [{"pattern": "x", "count": 8,
            "query_terms": ["sees", "leaves", "food"]}]}}
        candidate = structural_counterexample_candidate(report, set())
        self.assertEqual(candidate["seed"], "sees leaves food")
        self.assertEqual(candidate["failure_count"], 8)

    def test_partly_grounded_event_can_request_independent_repetition(self):
        verified = {"event_counts": {"fox|sees|grapes": 3, "bird|flies|sky": 1}}
        candidate = repeated_grounding_candidate(verified, set())
        self.assertEqual(candidate["seed"], "fox sees grapes")
        self.assertEqual(candidate["prior_observations"], 3)

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

    @patch("local_worker_v21.shutil.disk_usage",
           return_value=SimpleNamespace(free=100 * 1024 ** 3))
    @patch("local_worker_v21.runtime_bytes", return_value=4 * 1024 ** 3)
    @patch("local_worker_v21.time.time", return_value=7200)
    def test_abnormal_three_gib_per_hour_pauses_external_acquisition(
            self, _time, _bytes, _disk):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_json(runtime / "storage-status.json",
                       {"checked_epoch": 3600, "managed_bytes": 0})
            result = enforce_storage_budget(runtime, 20 * 1024 ** 3)
        self.assertTrue(result["abnormal_growth"])
        self.assertTrue(result["external_acquisition_paused"])
        self.assertIn("abnormal_growth", result["pause_reasons"])

    @patch("local_worker_v21.shutil.disk_usage",
           return_value=SimpleNamespace(free=40 * 1024 ** 3))
    def test_low_free_space_pauses_only_external_acquisition(self, _disk):
        with tempfile.TemporaryDirectory() as directory:
            result = enforce_storage_budget(Path(directory), 20 * 1024 ** 3)
        self.assertTrue(result["external_acquisition_paused"])
        self.assertIn("low_disk_free", result["pause_reasons"])
        self.assertIn("error-memory", result["protected_memories"])
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

    def test_human_status_explains_health_and_baselines_in_japanese(self):
        status = {"phase": "between_rounds", "seed": "fox grapes",
                  "heartbeat": "2026-09-01T00:00:00Z", "pid": 123, "rounds": 10,
                  "codex_or_remote_llm_calls": 0,
                  "global_memory": {"curricula": 12, "word_forms": 100,
                                    "grounded_word_forms": 40, "quality_events": 30},
                  "mastery": {"weakest_dimension": "associations"},
                  "association": {"evaluation": {"correct": 2, "baseline_correct": 4,
                                                    "total": 20}, "reinforced": 1, "weakened": 3},
                  "causal_evaluation": {"supported_hypotheses": 0,
                      "evaluation": {"correct": 4, "baseline_correct": 4, "total": 20}},
                  "representation": {"selected_evaluation": {"correct": 0, "total": 20,
                                                                 "coverage": 0.1}},
                  "error_memory": {"recognized_errors": 5, "unresolved_errors": 3},
                  "visual_memory": {"depictions_seen": 2, "pending_visual_curricula": 8},
                  "storage": {"managed_bytes": 1024 ** 3, "disk_free_bytes": 100 * 1024 ** 3}}
        rendered = render_human_status(status, now_epoch=1788220805, process_alive=True)
        self.assertIn("正常に稼働", rendered)
        self.assertIn("次の処理を準備中", rendered)
        self.assertIn("連想予測", rendered)
        self.assertIn("基準より下", rendered)
        self.assertIn("因果予測", rendered)
        self.assertIn("基準と同じ", rendered)
        self.assertIn("管理対象合計   : 1.0GB", rendered)

    def test_human_status_warns_when_process_is_dead_and_heartbeat_stale(self):
        status = {"phase": "learning", "heartbeat": "2026-09-01T00:00:00Z", "pid": 123}
        rendered = render_human_status(status, now_epoch=1788222000, process_alive=False)
        self.assertIn("確認が必要", rendered)
        self.assertIn("ワーカープロセスが停止", rendered)
        self.assertIn("最終更新が2分以上前", rendered)

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

    @patch("local_worker_v21.WEB_CACHE.get_json")
    def test_developmental_shelf_follows_pages_and_subcategories(self, get_json):
        def response(url, _agent):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            shelf = query["cmtitle"][0]
            continuation = query.get("cmcontinue", [None])[0]
            if shelf == "Category:Fables" and continuation is None:
                return {"continue": {"cmcontinue": "next"}, "query": {"categorymembers": [
                    {"ns": 14, "title": "Category:Animal fables"}]}}
            if shelf == "Category:Fables" and continuation == "next":
                return {"query": {"categorymembers": [{"ns": 0, "title": "The Fox and Crow"}]}}
            if shelf == "Category:Animal fables":
                return {"query": {"categorymembers": [
                    {"ns": 0, "title": "The Wolf and Lamb"}]}}
            return {"query": {"categorymembers": []}}
        get_json.side_effect = response
        candidates = discover_from_developmental_shelves(set(), 20)
        self.assertIn("fox crow", {item["seed"] for item in candidates})
        self.assertIn("wolf lamb", {item["seed"] for item in candidates})

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

    @patch("local_worker_v21.wait_for_retry", return_value=True)
    @patch("local_worker_v21.work")
    def test_supervisor_restarts_after_exhaustion_and_error(self, work_loop, _wait):
        work_loop.side_effect = [
            {"phase": "curriculum_exhausted", "seed": "one"},
            RuntimeError("unexpected parser failure"),
            {"phase": "stopped_by_user", "seed": "two"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = supervise("one", Path(directory), 0, 0, 1, 1, 1,
                               local_conversation=False)
        self.assertEqual(work_loop.call_count, 3)
        self.assertEqual(result["phase"], "stopped_by_user")

    @patch("local_worker_v21.work", return_value={"phase": "round_budget_exhausted"})
    def test_supervisor_respects_explicit_round_limit(self, work_loop):
        with tempfile.TemporaryDirectory() as directory:
            result = supervise("one", Path(directory), 2, 0, 1, 1, 1,
                               local_conversation=False)
        self.assertEqual(work_loop.call_count, 1)
        self.assertEqual(result["phase"], "round_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
