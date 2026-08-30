import unittest

from unified_agent_v9 import simulate


class UnifiedAgentTest(unittest.TestCase):
    def test_integrated_agent_recovers_changed_concept_action_and_lag(self):
        results = [simulate(seed, True) for seed in range(20)]
        accurate = sum(float(result["accuracy"]) == 1.0 for result in results)
        recovered = sum(int(result["recovery"]) < 270 for result in results)
        correct_structure = sum(set(result["inputs"]) == {0, 3, 4} and result["lag"] == 4 for result in results)
        self.assertGreaterEqual(accurate, 18)
        self.assertGreaterEqual(recovered, 18)
        self.assertGreaterEqual(correct_structure, 18)


if __name__ == "__main__":
    unittest.main()
