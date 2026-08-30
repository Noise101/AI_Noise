import unittest

from probabilistic_agent_v10 import simulate


class ProbabilisticAgentTest(unittest.TestCase):
    def test_recovers_noisy_causal_structure_and_calibrates(self):
        results = [simulate(seed, True) for seed in range(20)]
        recovered = sum(set(result["inputs"]) == {0, 3, 4} and result["lag"] == 5 for result in results)
        mean_brier = sum(float(result["brier"]) for result in results) / len(results)
        self.assertGreaterEqual(recovered, 17)
        self.assertLess(mean_brier, 0.035)


if __name__ == "__main__":
    unittest.main()

