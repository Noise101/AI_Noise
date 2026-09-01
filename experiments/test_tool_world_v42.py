import unittest

from tool_world_v42 import (ToolWorld, empty_tool_memory, initial_state, learn_episodes,
                            task_specs)


class ToolWorldTest(unittest.TestCase):
    def test_goal_requires_a_real_multi_step_sequence(self):
        task = {"goal_height": 3, "tool_capacities": [1, 2],
                "tool_locations": ["tool0", "tool1"]}
        state = initial_state(task)
        state, success, _ = ToolWorld.act(state, "take")
        self.assertFalse(success)
        for action in ("move:tool1", "pick:1", "move:target", "place", "climb", "take"):
            state, _, _ = ToolWorld.act(state, action)
        self.assertTrue(state["goal_taken"])

    def test_learning_receives_no_solution_plan(self):
        memory = empty_tool_memory()
        learn_episodes(memory, 1)
        self.assertFalse(memory["solution_plan_supplied"])
        self.assertEqual(memory["remote_llm_calls"], 0)

    def test_learns_tool_use_and_transfers_to_heldout_tasks(self):
        memory = empty_tool_memory()
        summary = learn_episodes(memory, 4000)
        self.assertGreaterEqual(summary["unseen_tasks"]["success_rate"], 0.9)
        self.assertGreater(summary["successful_training_plans"], 0)
        self.assertGreater(summary["remembered_action_failures"], 0)
        self.assertEqual(summary["status"], "stage_2_mastered")


if __name__ == "__main__":
    unittest.main()
