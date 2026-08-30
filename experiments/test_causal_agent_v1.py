import random
import unittest

from causal_agent_v1 import ChangingWorld, RuleInducer


class RuleInducerTest(unittest.TestCase):
    def test_synthesizes_both_hidden_rules(self):
        world = ChangingWorld(change_at=35)
        agent = RuleInducer(world.sensor_count, random.Random(11))
        rules = {}
        for step in range(80):
            observation, _ = agent.choose(world.candidates())
            agent.learn(observation, world.reward(observation, step))
            if step in (34, 79):
                rules[step] = str(agent.best_rule()[0])
        self.assertEqual(rules[34], "s0 & !s3")
        self.assertEqual(rules[79], "s2 | s5")


if __name__ == "__main__":
    unittest.main()
