import gzip
import json
import tempfile
import unittest
import urllib.parse
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from local_worker_v21 import (_seed_from_title, append_events, developmental_source_quality,
                              discover_curriculum,
                              discover_from_developmental_shelves, enforce_storage_budget, is_transient_error,
                              compact_learning_history, merge_curiosity, read_json,
                              render_ability_report, render_detail_status, render_human_status, status_record,
                              parser_counterexample_candidate, structural_counterexample_candidate,
                              repeated_grounding_candidate, curriculum_strategy_allowed,
                              learned_curriculum_score, conversation_practice_summary,
                              rotate_events_log, EVENTS_LOG_ROTATE_LINES, EVENTS_LOG_KEEP_LINES,
                              supervise, update_autonomy_state, COLLECTION_STALL_ROUNDS,
                              update_collection_progress,
                              update_curriculum_strategy, work, write_json)


class LocalWorkerTest(unittest.TestCase):
    def test_legacy_null_conversation_metric_does_not_break_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_json(runtime / "dialogue-ledger.json", {"turns": [
                {"practice_metrics": {"formed_followup": None,
                                      "relevant_token_overlap": None}},
                {"practice_metrics": {"formed_followup": True,
                                      "relevant_token_overlap": 0.2}}]})
            result = conversation_practice_summary(runtime)
        self.assertEqual(result["evaluated_turns"], 2)
        self.assertEqual(result["successful_followups"], 1)

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

    @staticmethod
    def _es_report(curricula, *, lift=0, locked=True, selected=True, curve=None):
        """A minimal report shaped like work() builds it for update_autonomy_state."""
        selected_block = ({"task": "event_plausibility", "model_id": "smoothed_argument",
                           "lift": lift, "final": {"lift": lift}} if selected and lift else None)
        return {"global_memory": {"curricula": curricula},
                "association": {"selected_evaluation": {"correct": 1, "baseline_correct": 1},
                                "learning_curve": curve or []},
                "representation": {"selected_evaluation": {"correct": 0}},
                "event_structure": {"benchmark": {"locked": locked, "eligible_collection_count": 26},
                                    "selected": selected_block,
                                    "learning_curve": curve or []},
                "experience_revision": {"evaluation": {"correct": 1, "total": 100 + curricula},
                                        "reusable_rules": 0, "failure_patterns": [{"pattern": "x"}]}}

    def test_autonomy_switches_to_counterexamples_when_more_tests_add_no_correct_prediction(self):
        curriculum = {}
        state = None
        for index in range(12):
            state = update_autonomy_state(
                curriculum, self._es_report(100 + index * 2, lift=0))
        self.assertEqual(state["mode"], "counterexample_hunt")
        self.assertFalse(state["human_intervention_required"])

    def test_autonomy_stops_collection_when_bounded_intervention_has_no_gain(self):
        curriculum = {}
        for index in range(12):
            state = update_autonomy_state(
                curriculum, self._es_report(100 + index * 2, lift=0))
        self.assertEqual(state["mode"], "counterexample_hunt")
        start = state["mode_started_curricula"]
        for index in range(1, 22):
            state = update_autonomy_state(
                curriculum, self._es_report(start + index * 2, lift=0))
        self.assertEqual(state["mode"], "capability_plateau")
        self.assertTrue(state["human_intervention_required"])

    def test_autonomy_returns_to_normal_only_after_measured_gain(self):
        curriculum = {"autonomy_state": {"mode": "counterexample_hunt",
            "mode_started_curricula": 100,
            "intervention_start_snapshot": {"event_structure_lift": 0}}}
        state = update_autonomy_state(curriculum, self._es_report(110, lift=3))
        self.assertEqual(state["mode"], "normal_curriculum")

    def test_autonomy_state_exposes_a_per_subsystem_trend_without_gating_on_it(self):
        declining = [{"training_events": index, "lift": value}
                     for index, value in enumerate((8, 7, 6, 5, 2, 1, 0, -1, -2, -3))]
        state = update_autonomy_state({}, self._es_report(110, lift=0, curve=declining))
        self.assertEqual(state["trends"]["association_lift"], "declining")
        self.assertEqual(state["trends"]["event_structure_lift"], "declining")
        # A declining trend must not by itself force a plateau.
        self.assertNotEqual(state["mode"], "capability_plateau")

    def test_selection_lift_without_a_selected_model_cannot_clear_plateau(self):
        # best_rejected_candidate has a positive selection lift but nothing
        # cleared the FINAL gate, so event_structure_lift stays 0.
        curriculum = {"autonomy_state": {"mode": "counterexample_hunt",
            "mode_started_curricula": 100,
            "intervention_start_snapshot": {"event_structure_lift": 0}}}
        report = self._es_report(110, lift=0, selected=False)
        report["event_structure"]["best_rejected_candidate"] = {"selection": {"lift": 9}}
        state = update_autonomy_state(curriculum, report)
        self.assertEqual(state["mode"], "counterexample_hunt")

    def test_plateau_is_not_declared_while_benchmark_is_unmeasurable(self):
        curriculum = {}
        state = None
        for index in range(45):
            state = update_autonomy_state(
                curriculum, self._es_report(100 + index * 2, lift=0, locked=False))
        self.assertEqual(state["mode"], "normal_curriculum")
        self.assertFalse(state["human_intervention_required"])

    def test_plateau_detector_does_not_compare_across_the_redesign(self):
        # Pre-redesign snapshots (no schema:2) in the window must not be read as
        # a measurable plateau, even though they carry benchmark_locked.
        curriculum = {"capability_history": [
            {"curricula": 1400 + i, "benchmark_locked": True, "world_model_lift": 0}
            for i in range(15)]}
        for index in range(12):
            state = update_autonomy_state(
                curriculum, self._es_report(1500 + index * 2, lift=0))
        # only the new schema-2 snapshots count; not yet 10 spanning >= 20 curricula
        self.assertIn(state["mode"], {"normal_curriculum", "counterexample_hunt"})

    def test_pre_redesign_intervention_state_is_reset_not_escalated(self):
        # A counterexample_hunt recorded against the retired world_model_lift
        # signal (no event_structure_lift key) must be dropped, not carried
        # forward into a spurious capability_plateau after 40 curricula.
        curriculum = {"autonomy_state": {"mode": "counterexample_hunt",
            "mode_started_curricula": 1693,
            "intervention_start_snapshot": {"curricula": 1693, "world_model_lift": 0,
                                            "benchmark_locked": True}}}
        state = update_autonomy_state(curriculum, self._es_report(1760, lift=0))
        self.assertEqual(state["mode"], "normal_curriculum")
        self.assertIsNone(state["intervention_start_snapshot"])

    def test_persisted_plateau_is_released_when_benchmark_still_not_ready(self):
        curriculum = {"autonomy_state": {"mode": "capability_plateau",
            "mode_started_curricula": 1418,
            "intervention_start_snapshot": {"event_structure_lift": 0, "curricula": 1418},
            "human_intervention_required": True}}
        state = update_autonomy_state(
            curriculum, self._es_report(1460, lift=0, locked=False))
        self.assertEqual(state["mode"], "normal_curriculum")
        self.assertFalse(state["human_intervention_required"])
        self.assertIsNone(state["intervention_start_snapshot"])

    def test_collection_progress_detects_a_stall_and_recovers_on_growth(self):
        curriculum = {}
        event_structure = {"benchmark": {"locked": False, "eligible_collection_count": 5}}
        stalled = False
        for _ in range(COLLECTION_STALL_ROUNDS - 1):
            stalled = update_collection_progress(curriculum, event_structure)
        self.assertFalse(stalled)
        self.assertTrue(update_collection_progress(curriculum, event_structure))
        event_structure = {"benchmark": {"locked": False, "eligible_collection_count": 6}}
        self.assertFalse(update_collection_progress(curriculum, event_structure))
        self.assertEqual(curriculum["collection_progress"]["unchanged_rounds"], 0)

    def test_collection_progress_is_irrelevant_once_benchmark_is_locked(self):
        curriculum = {"collection_progress": {"count": 30, "unchanged_rounds": 999}}
        stalled = update_collection_progress(curriculum, {"benchmark": {"locked": True}})
        self.assertFalse(stalled)
        self.assertNotIn("collection_progress", curriculum)

    @patch("local_worker_v21.update_collection_progress", return_value=True)
    @patch("local_worker_v21.discover_from_developmental_shelves")
    @patch("local_worker_v21.discover_curriculum")
    @patch("local_worker_v21.run_cycle")
    def test_stalled_collection_growth_forces_a_shelf_search(
            self, run_cycle, discover, shelves, _stalled):
        # discover_curriculum alone already fills the frontier from a known
        # collection; a stall must still add the shelf route's candidates.
        discover.return_value = [{"seed": "known collection page", "score": 2.5,
                                  "reason": "unvisited page in an observed story collection",
                                  "parent_url": "source"}]
        shelves.return_value = [{"seed": "brand new shelf title", "score": 1.0,
                                 "reason": "unread title selected from a developmental shelf",
                                 "parent_url": "shelf"}]
        run_cycle.return_value = {"state": {"completed_gap_ids": [],
                                            "stop_reason": "no_unresolved_executable_gap"},
                                  "current_gaps": [], "knowledge": {}, "web_usage": {}}
        with tempfile.TemporaryDirectory() as directory:
            work("seed", Path(directory), 1, 0, 1, 1, 1, local_conversation=False)
        self.assertTrue(shelves.called)

    @patch("local_worker_v21.update_collection_progress", return_value=False)
    @patch("local_worker_v21.discover_from_developmental_shelves")
    @patch("local_worker_v21.discover_curriculum")
    @patch("local_worker_v21.run_cycle")
    def test_shelf_search_is_not_forced_while_collections_are_still_growing(
            self, run_cycle, discover, shelves, _not_stalled):
        discover.return_value = [{"seed": "known collection page", "score": 2.5,
                                  "reason": "unvisited page in an observed story collection",
                                  "parent_url": "source"}]
        run_cycle.return_value = {"state": {"completed_gap_ids": [],
                                            "stop_reason": "no_unresolved_executable_gap"},
                                  "current_gaps": [], "knowledge": {}, "web_usage": {}}
        with tempfile.TemporaryDirectory() as directory:
            work("seed", Path(directory), 1, 0, 1, 1, 1, local_conversation=False)
        self.assertFalse(shelves.called)

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

    def test_append_events_writes_one_jsonl_line_per_event_with_module_and_curricula(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            append_events(runtime, "event_structure_v1", 1500,
                          [{"event_type": "benchmark_locked", "before": {"locked": False},
                            "after": {"locked": True}, "reason": "example"}])
            append_events(runtime, "event_structure_v1", 1500, [])  # a no-op must not touch the file
            lines = (runtime / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["module"], "event_structure_v1")
            self.assertEqual(record["curricula"], 1500)
            self.assertEqual(record["event_type"], "benchmark_locked")
            self.assertIn("ts", record)

    def test_append_events_with_no_events_does_not_create_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            append_events(runtime, "event_structure_v1", 0, [])
            self.assertFalse((runtime / "events.jsonl").exists())

    def test_rotate_events_log_archives_the_oldest_lines_past_the_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            path = runtime / "events.jsonl"
            lines = [json.dumps({"n": index}) for index in range(EVENTS_LOG_ROTATE_LINES + 50)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = rotate_events_log(runtime)
            self.assertTrue(result["rotated"])
            live = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(live), EVENTS_LOG_KEEP_LINES)
            self.assertEqual(json.loads(live[0])["n"],
                             EVENTS_LOG_ROTATE_LINES + 50 - EVENTS_LOG_KEEP_LINES)
            archive_path = runtime / result["archive_path"]
            self.assertTrue(archive_path.exists())
            with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
                archived_lines = handle.read().splitlines()
            self.assertEqual(len(archived_lines), EVENTS_LOG_ROTATE_LINES + 50 - EVENTS_LOG_KEEP_LINES)
            self.assertEqual(json.loads(archived_lines[0])["n"], 0)
            # A second call below the threshold must be a no-op.
            self.assertFalse(rotate_events_log(runtime)["rotated"])

    def test_decision_replay_reconstructs_the_decision_sequence_from_the_log_alone(self):
        """Decision replay (not full state replay, see event_structure_v1's
        docstring): reading events.jsonl alone -- never touching the
        event-structure snapshot -- must recover when the benchmark locked and
        the selected model switched."""
        from event_structure_v1 import train_and_evaluate
        from test_event_structure_v1 import verified, learnable_sequences
        data = verified(learnable_sequences(60))
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            first = train_and_evaluate(data)
            append_events(runtime, "event_structure_v1", 100, first.pop("emitted_events", []))
            forced_previous = dict(first)
            forced_previous["selected_model_id"] = "some_other_model_id"
            second = train_and_evaluate(data, forced_previous)
            append_events(runtime, "event_structure_v1", 140, second.pop("emitted_events", []))

            events = [json.loads(line) for line in
                     (runtime / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        lock_events = [event for event in events if event["event_type"] == "benchmark_locked"]
        mode_events = [event for event in events if event["event_type"] == "selected_model_changed"]
        self.assertEqual(len(lock_events), 1)
        self.assertEqual(lock_events[0]["curricula"], 100)
        self.assertFalse(lock_events[0]["before"]["locked"])
        self.assertTrue(lock_events[0]["after"]["locked"])
        self.assertEqual(len(mode_events), 1)
        self.assertEqual(mode_events[0]["curricula"], 140)
        self.assertEqual(mode_events[0]["before"]["selected_model_id"], "some_other_model_id")
        self.assertEqual(mode_events[0]["after"]["selected_model_id"], second["selected_model_id"])

    def test_human_status_explains_health_and_baselines_in_japanese(self):
        status = {"phase": "between_rounds", "seed": "fox grapes",
                  "heartbeat": "2026-09-01T00:00:00Z", "pid": 123, "rounds": 10,
                  "codex_or_remote_llm_calls": 0,
                  "global_memory": {"curricula": 12, "word_forms": 100,
                                    "grounded_word_forms": 40, "quality_events": 30},
                  "mastery": {"weakest_dimension": "associations"},
                  "event_structure": {"selected_model_id": "frequency_baseline",
                      "selection_status": "no_model_beats_corrected_selection_baseline",
                      "benchmark": {"locked": True}, "selected": None,
                      "evaluations": [
                          {"task": "event_plausibility", "selection": {
                              "correct": 2, "baseline_correct": 4, "total": 20}},
                          {"task": "verb_cloze", "selection": {
                              "correct": 4, "baseline_correct": 4, "total": 20}}]},
                  "representation": {"selected_evaluation": {"correct": 0, "total": 20,
                                                                 "coverage": 0.1}},
                  "error_memory": {"recognized_errors": 5, "unresolved_errors": 3},
                  "visual_memory": {"depictions_seen": 2, "pending_visual_curricula": 8},
                  "storage": {"managed_bytes": 1024 ** 3, "disk_free_bytes": 100 * 1024 ** 3}}
        rendered = render_human_status(status, now_epoch=1788220805, process_alive=True)
        self.assertIn("正常に稼働", rendered)
        self.assertIn("次の処理を準備中", rendered)
        self.assertIn("妥当性判定     : 2/20、単純基準より -2件", rendered)
        self.assertIn("動詞クローズ   : 4/20、単純基準より +0件", rendered)
        self.assertIn("因果予測       : 評価不能", rendered)
        self.assertIn("管理対象合計   : 1.0GB", rendered)
        # the micro-world diagnostic block is gone (invariant 18: zero real credit)
        self.assertNotIn("限定実験世界", rendered)
        self.assertNotIn("第一段階", rendered)
        # so is the section that just duplicated 現在できること
        self.assertNotIn("現在の能力評価", rendered)
        self.assertIn("実用会話       : 未到達", rendered)

    def test_human_status_shows_learning_curve_trend(self):
        status = {"phase": "learning",
                  "event_structure": {"selected_model_id": "frequency_baseline",
                      "benchmark": {"locked": True}, "selected": None, "evaluations": [],
                      "learning_curve_trend": "improving"}}
        rendered = render_human_status(status, now_epoch=1788220805, process_alive=True)
        self.assertIn("改善傾向", rendered)

    def test_ability_report_shows_honest_claims_and_examples(self):
        status = {"global_memory": {"word_forms": 100, "grounded_word_forms": 40},
                  "verified_experience": {"accepted_sentences": 30},
                  "representation": {"selected_evaluation": {"correct": 0, "total": 5}},
                  "experience_revision": {"reusable_rules": 0},
                  "event_structure": {"benchmark": {"locked": True}, "selected": None,
                                      "selection_status": "no_model_beats_corrected_selection_baseline",
                                      "evaluations": []},
                  "abstraction_world": {"open_transfer_gates": {"a": True, "b": False}}}
        event_structure = {
            "selected_model_id": "event_plausibility:smoothed_argument",
            "evaluations": [
                {"task": "event_plausibility", "model_id": "smoothed_argument",
                 "selection": {"correct": 6, "baseline_correct": 4, "total": 10, "lift": 2}},
                {"task": "verb_cloze", "model_id": "smoothed_argument",
                 "selection": {"correct": 3, "baseline_correct": 3, "total": 10, "lift": 0}}],
            "final_attempt": {"task": "event_plausibility", "model_id": "smoothed_argument",
                              "evaluation": {"correct": 7, "baseline_correct": 5, "total": 10, "lift": 2}},
            "counterexamples": [{"subject": "fox", "object": "grapes",
                                 "predicted": "said", "observed": "ate", "correct": False}]}
        rendered = render_ability_report(status, event_structure)
        self.assertIn("実用会話       : 未到達", rendered)
        self.assertIn("妥当性判定", rendered)
        self.assertIn("lift +2", rendered)
        self.assertIn("Noise=said / 実際=ate", rendered)

    def test_human_status_warns_when_process_is_dead_and_heartbeat_stale(self):
        status = {"phase": "learning", "heartbeat": "2026-09-01T00:00:00Z", "pid": 123}
        rendered = render_human_status(status, now_epoch=1788222000, process_alive=False)
        self.assertIn("確認が必要", rendered)
        self.assertIn("ワーカープロセスが停止", rendered)
        self.assertIn("最終更新が2分以上前", rendered)

    def test_detail_status_consolidates_admissions_modules_and_strategy_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_json(runtime / "status.json", {"phase": "learning", "rounds": 42,
                                                 "seed": "the fox"})
            write_json(runtime / "curriculum-state.json", {
                "autonomy_state": {"mode": "normal_curriculum"},
                "admission_log": [
                    {"seed": "wolf tale", "at_curricula": 900, "admitted": True, "score": 0.71,
                     "metrics": {"narrative_ratio": 0.9, "short_sentence_ratio": 0.8,
                                 "vocabulary_fit": 1.0, "known_word_ratio": 0.95,
                                 "subject_recurrence": 0.3, "dialogue_ratio": 0.1},
                     "admission_reason": "developmental_passage",
                     "source_urls": ["https://example/wolf"]},
                    {"seed": "dense treatise", "at_curricula": 901, "admitted": False,
                     "score": 0.4, "metrics": {}, "reasons": ["combined developmental score below 0.68"],
                     "admission_reason": "outside_current_level", "source_urls": []},
                ],
                "strategy_performance": {
                    "unvisited page in an observed story collection": {
                        "attempts": 100, "admitted": 20, "rejected": 80,
                        "seed_outcomes": {"a": True}}}})
            write_json(runtime / "experience-revision.json", {"summary": {
                "selected_context": "one_event", "reusable_rules": 0,
                "evaluation": {"correct": 1, "baseline_correct": 1, "total": 30}}})
            write_json(runtime / "event-structure.json", {
                "benchmark": {"locked": True, "status": "ready", "collection_count": 90,
                              "source_count": 134, "selection_events": 267, "final_events": 266,
                              "selection_regime": "collection_disjoint_within_event_v1",
                              "fingerprint": "abcd"},
                "training": {"events": 1870, "verb_vocabulary": 79},
                "selected_model_id": "event_plausibility:smoothed_argument",
                "selection_status": "accepted_final_gain",
                "learning_curve_trend": "improving", "corrupters": ["verb_swap"],
                "final_queries_used": 1, "final_query_budget": 5,
                "selected": {"task": "event_plausibility", "final": {"lift": 24}},
                "final_attempt": {"task": "event_plausibility", "model_id": "smoothed_argument",
                                  "evaluation": {"correct": 160, "baseline_correct": 136,
                                                 "total": 266, "lift": 24, "one_sided_sign_p": 4e-06}},
                "evaluations": [
                    {"task": "verb_cloze", "model_id": "smoothed_argument",
                     "selection": {"correct": 26, "baseline_correct": 31, "total": 267,
                                   "lift": -5, "coverage": 0.6, "one_sided_sign_p": 0.9}},
                    {"task": "event_plausibility", "model_id": "smoothed_argument",
                     "selection": {"correct": 131, "baseline_correct": 118, "total": 267,
                                   "lift": 13, "coverage": 0.6, "one_sided_sign_p": 0.0004}}]})

            first = render_detail_status(runtime)
            self.assertIn("wolf tale", first)
            self.assertIn("vocabulary_fit=1.0", first)
            self.assertIn("combined developmental score below 0.68", first)
            self.assertIn("event_plausibility:smoothed_argument", first)
            self.assertIn("FINAL event_plausibility", first)
            self.assertIn("[PASS]", first)
            self.assertIn("locked=True", first)
            self.assertIn("初回実行", first)
            self.assertTrue((runtime / "status-detail-snapshot.json").exists())

            # Advance one strategy attempt; the second run must show the delta.
            state = read_json(runtime / "curriculum-state.json")
            bucket = state["strategy_performance"]["unvisited page in an observed story collection"]
            bucket["attempts"] += 3
            bucket["admitted"] += 1
            bucket["rejected"] += 2
            write_json(runtime / "curriculum-state.json", state)

            second = render_detail_status(runtime)
            self.assertIn("前回実行", second)
            self.assertIn("Δ試行+3 採用+1 不採用+2", second)

    def test_status_record_reports_honest_causal_and_event_structure_shims(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_json(runtime / "event-structure.json", {
                "benchmark": {"locked": True}, "selection_status": "accepted_final_gain",
                "selected_model_id": "event_plausibility:smoothed_argument",
                "learning_curve": [], "learning_curve_trend": "improving",
                "evaluations": [{"task": "event_plausibility",
                                 "selection": {"correct": 8, "baseline_correct": 5, "total": 20}}]})
            status = status_record("seed", runtime, "learning", 1)
            self.assertEqual(status["causal_evaluation"]["supported_hypotheses"], 0)
            self.assertIn("retired", status["causal_evaluation"]["limitations"][0])
            self.assertEqual(status["association"]["selected_evaluation"]["correct"], 8)

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
