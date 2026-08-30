import unittest

from constructive_concepts_v8 import simulate


class ConstructiveConceptTest(unittest.TestCase):
    def test_grows_three_input_concept_from_residuals(self):
        results = [simulate(seed) for seed in range(30)]
        exact = sum(float(result["accuracy"]) == 1.0 for result in results)
        correct_inputs = sum(set(result["inputs"]) == {0, 3, 4} for result in results)
        self.assertGreaterEqual(exact, 27)
        self.assertGreaterEqual(correct_inputs, 27)


if __name__ == "__main__":
    unittest.main()
