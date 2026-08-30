import unittest

from active_falsification_v7 import run_trial


class ActiveFalsificationTest(unittest.TestCase):
    def test_active_experiments_correct_belief_faster(self):
        active = [run_trial(seed, True)["solved_at"] for seed in range(50)]
        passive = [run_trial(seed, False)["solved_at"] for seed in range(50)]
        self.assertLess(sum(active), sum(passive) * 0.7)
        self.assertGreaterEqual(sum(step <= 8 for step in active), 45)


if __name__ == "__main__":
    unittest.main()
