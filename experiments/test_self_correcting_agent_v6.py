import unittest

from self_correcting_agent_v6 import simulate


class SelfCorrectionTest(unittest.TestCase):
    def test_revises_incomplete_concept_from_prediction_errors(self):
        results = [simulate(seed) for seed in range(20)]
        accurate = sum(float(result["concept_accuracy"]) == 1.0 for result in results)
        revised = sum(bool(result["revisions"]) for result in results)
        self.assertGreaterEqual(accurate, 18)
        self.assertGreaterEqual(revised, 18)


if __name__ == "__main__":
    unittest.main()
