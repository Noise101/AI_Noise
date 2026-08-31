import unittest
from unittest.mock import patch

from error_memory_v35 import empty_error_memory, update_error_memory


class ErrorMemoryTest(unittest.TestCase):
    def association(self, count=1, weakened=False):
        return {"predictions": [
            {"prior": "fox|waited|", "prediction": "left", "observed": "jumped",
             "cues": ["subject:fox"], "correct": False, "count": count},
            {"prior": "bird|sang|", "prediction": "flew", "observed": "flew",
             "cues": ["subject:bird"], "correct": True, "count": 1}],
            "predictive_associations": ([{"cue": "subject:fox", "associated_outcome": "left",
                "status": "weakened", "prediction_successes": 0, "prediction_failures": 2}]
                if weakened else [])}

    @patch("error_memory_v35.now", return_value="2026-09-01T00:00:00Z")
    def test_remembers_wrong_prediction_but_not_correct_one(self, _now):
        ledger = update_error_memory(empty_error_memory(), self.association(), {}, 10)
        self.assertEqual(ledger["summary"]["recognized_errors"], 1)
        record = next(iter(ledger["records"].values()))
        self.assertEqual(record["asserted_or_predicted"], "left")
        self.assertEqual(record["observed"], "jumped")

    @patch("error_memory_v35.now", return_value="2026-09-01T00:00:00Z")
    def test_recalculation_does_not_duplicate_same_error(self, _now):
        ledger = empty_error_memory()
        update_error_memory(ledger, self.association(), {}, 10)
        update_error_memory(ledger, self.association(), {}, 11)
        self.assertEqual(ledger["summary"]["recognized_errors"], 1)
        self.assertEqual(len(next(iter(ledger["records"].values()))["revision_history"]), 1)

    @patch("error_memory_v35.now", return_value="2026-09-01T00:00:00Z")
    def test_preserves_history_when_error_repeats_and_changes_mechanism(self, _now):
        ledger = empty_error_memory()
        update_error_memory(ledger, self.association(), {}, 10)
        update_error_memory(ledger, self.association(3, True), {}, 20)
        record = next(iter(ledger["records"].values()))
        self.assertEqual(record["occurrences"], 3)
        self.assertEqual(record["status"], "recognized_and_corrected")
        self.assertEqual(len(record["revision_history"]), 2)

    def test_causal_error_retains_counterexample_without_claiming_truth(self):
        causal = {"preregistered_predictions": [{"prior": "rain|fell|", "prediction": "grew",
            "observed_after_registration": "dried", "correct": False, "count": 1,
            "confidence_basis": "prior_action=fall"}]}
        ledger = update_error_memory(empty_error_memory(), {}, causal, 5)
        record = next(iter(ledger["records"].values()))
        self.assertFalse(record["correction"]["causal_credit"])


if __name__ == "__main__":
    unittest.main()
