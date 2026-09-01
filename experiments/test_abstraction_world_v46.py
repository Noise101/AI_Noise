import unittest

from abstraction_world_v46 import (COMPETENCIES, assess_open_transfer,
                                   empty_abstraction_memory, learn_abstractions)


class AbstractionWorldTest(unittest.TestCase):
    def test_all_stage_five_competencies_and_required_gates_pass(self):
        memory = empty_abstraction_memory()
        summary = learn_abstractions(memory, 300)
        self.assertEqual(summary["status"], "stage_5_bounded_complete_open_transfer_pending")
        self.assertTrue(summary["bounded_world_complete"])
        self.assertEqual(summary["competencies_passed"], len(COMPETENCIES))
        self.assertTrue(all(summary["required_gates"].values()))
        self.assertGreater(summary["reusable_abstract_rules"], 0)

    def test_causal_and_association_models_beat_surface_baselines(self):
        memory = empty_abstraction_memory()
        summary = learn_abstractions(memory, 300)
        for name in ("causal_transfer", "structural_association"):
            result = summary["competencies"][name]
            self.assertGreater(result["correct"], result["baseline_correct"])

    def test_revision_improves_and_world_model_keeps_provenance(self):
        memory = empty_abstraction_memory()
        summary = learn_abstractions(memory, 300)
        revision = summary["competencies"]["self_revision"]
        self.assertGreater(revision["correct"], revision["baseline_correct"])
        self.assertGreater(summary["world_model"]["evidence_links"], 0)
        self.assertGreater(summary["world_model"]["revision_links"], 0)
        self.assertEqual(summary["remote_llm_calls"], 0)

    def test_open_curriculum_transfer_is_required_for_final_completion(self):
        memory = empty_abstraction_memory()
        learn_abstractions(memory, 300)
        summary = assess_open_transfer(
            memory,
            {"selected_scheme": "role_action", "selected_evaluation": {"correct": 8, "baseline_correct": 4}},
            {"evaluation": {"correct": 8, "baseline_correct": 5}},
            {"evaluation": {"correct": 7, "baseline_correct": 5}},
            {"summary": {"reusable_rules": 2,
                         "evaluation": {"correct": 9, "baseline_correct": 5}}})
        self.assertEqual(summary["status"], "stage_5_complete")
        self.assertTrue(summary["open_transfer_complete"])


if __name__ == "__main__":
    unittest.main()
