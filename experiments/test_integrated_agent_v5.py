import unittest

from integrated_agent_v5 import simulate


class IntegratedAgentTest(unittest.TestCase):
    def test_concept_survives_and_accelerates_temporal_relearning(self):
        # The full research evaluation uses 50+ seeds; keep CI reasonably fast.
        results = [simulate(seed) for seed in range(10)]
        retained = sum(float(result["concept_accuracy_after"]) for result in results) / len(results)
        transfer_recovery = sum(float(result["recovery_transfer"]) for result in results) / len(results)
        fresh_recovery = sum(float(result["recovery_fresh"]) for result in results) / len(results)
        self.assertGreaterEqual(retained, 0.9)
        self.assertLess(transfer_recovery, fresh_recovery)


if __name__ == "__main__":
    unittest.main()
