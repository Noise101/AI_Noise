import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_worker_v21 import read_json, work, write_json


class LocalWorkerTest(unittest.TestCase):
    def test_atomic_status_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            write_json(path, {"phase": "learning"})
            self.assertEqual(read_json(path), {"phase": "learning"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    @patch("local_worker_v21.run_cycle")
    def test_repeats_step_budgets_then_becomes_idle(self, run_cycle):
        run_cycle.side_effect = [
            {"state": {"completed_gap_ids": ["one"], "stop_reason": "step_budget_exhausted"},
             "current_gaps": [{"gap_id": "two"}], "web_usage": {"network_requests": 1}},
            {"state": {"completed_gap_ids": ["one", "two"],
                       "stop_reason": "no_unresolved_executable_gap"},
             "current_gaps": [], "web_usage": {"network_requests": 0}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = work("seed", Path(directory), 5, 0, 1, 1, 2)
            self.assertEqual(run_cycle.call_count, 2)
            self.assertEqual(result["phase"], "idle")
            self.assertEqual(result["completed_gaps"], 2)
            self.assertEqual(result["codex_or_remote_llm_calls"], 0)

    @patch("local_worker_v21.run_cycle")
    def test_stop_file_prevents_a_cycle(self, run_cycle):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            # Simulate a stop appearing immediately after startup writes its status.
            original_write = write_json
            def request_stop(path, value):
                original_write(path, value)
                if value["phase"] == "starting":
                    (runtime / "STOP").touch()
            with patch("local_worker_v21.write_json", side_effect=request_stop):
                result = work("seed", runtime, 2, 0, 1, 1, 1)
            run_cycle.assert_not_called()
            self.assertEqual(result["phase"], "stopped_by_user")


if __name__ == "__main__":
    unittest.main()
