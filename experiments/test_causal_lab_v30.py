import unittest

from causal_lab_v30 import run_lab


class CausalLabTest(unittest.TestCase):
    def test_active_interventions_identify_unseen_procedural_rule(self):
        result = run_lab("unseen seed")
        self.assertTrue(result["identified"])
        self.assertTrue(result["prediction_test"])
        self.assertEqual(result["world_knowledge_credit"], 0)
        self.assertTrue(any(item["hypotheses_after"] < item["hypotheses_before"]
                            for item in result["interventions"]))


if __name__ == "__main__":
    unittest.main()
