import unittest

from temporal_agent_v2 import simulate


class TemporalAgentTest(unittest.TestCase):
    def test_discovers_delayed_rules_across_seeds(self):
        results = [simulate(seed) for seed in range(30)]
        self.assertGreaterEqual(sum(bool(r["before"]) for r in results), 27)
        self.assertGreaterEqual(sum(bool(r["after"]) for r in results), 27)


if __name__ == "__main__":
    unittest.main()
