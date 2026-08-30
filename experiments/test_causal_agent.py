import random
import unittest

from causal_agent import CausalAgent, ChangingWorld


class CausalAgentTest(unittest.TestCase):
    def test_learns_and_relearns(self):
        agent = CausalAgent(random.Random(3))
        world = ChangingWorld("red", "square", 20)
        snapshots = {}
        for step in range(50):
            obj, _ = agent.choose()
            agent.learn(obj, world.act(obj, step))
            if step in (19, 49):
                snapshots[step] = agent.best_hypothesis()
        self.assertEqual(snapshots[19][0], "red")
        self.assertGreater(snapshots[19][1], 0.8)
        self.assertEqual(snapshots[49][0], "square")
        self.assertGreater(snapshots[49][1], 0.8)


if __name__ == "__main__":
    unittest.main()
