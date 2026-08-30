import unittest

from truth_table_concepts_v4 import trial


class TruthTableConceptTest(unittest.TestCase):
    def test_learns_xor_without_xor_in_rule_grammar(self):
        results = [trial(seed) for seed in range(50)]
        concepts = sum(result["concept"] is not None for result in results)
        transfer = sum(float(result["transfer_accuracy"]) for result in results) / len(results)
        fresh = sum(float(result["fresh_accuracy"]) for result in results) / len(results)
        self.assertGreaterEqual(concepts, 45)
        self.assertGreater(transfer, fresh + 0.03)


if __name__ == "__main__":
    unittest.main()
