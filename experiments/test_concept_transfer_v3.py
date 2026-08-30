import unittest

from concept_transfer_v3 import trial


class ConceptTransferTest(unittest.TestCase):
    def test_transfer_improves_mean_few_shot_accuracy(self):
        results = [trial(seed) for seed in range(50)]
        transfer = sum(float(result["transfer_accuracy"]) for result in results) / len(results)
        fresh = sum(float(result["fresh_accuracy"]) for result in results) / len(results)
        self.assertGreater(transfer, fresh + 0.05)

    def test_library_does_not_materially_harm_unrelated_task(self):
        results = [trial(seed) for seed in range(50)]
        transfer = sum(float(result["unrelated_transfer_accuracy"]) for result in results) / len(results)
        fresh = sum(float(result["unrelated_fresh_accuracy"]) for result in results) / len(results)
        self.assertGreaterEqual(transfer, fresh - 0.02)


if __name__ == "__main__":
    unittest.main()
