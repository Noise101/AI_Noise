import unittest

from narrative_event_v29 import NarrativeEventExtractor
from llm_tooluse_v1 import formulate_tasks, run, verify


class ScriptedWorker:
    """Stands in for OllamaWorker: canned responses, no network."""

    def __init__(self, responses, available=True):
        self.responses = list(responses)
        self._available = available
        self.model = "scripted"
        self.seen = []

    def available(self):
        return self._available

    def do_task(self, instruction):
        self.seen.append(instruction)
        return self.responses.pop(0) if self.responses else None


GROUNDED = {"fox", "wolf", "dog", "cat", "ate", "saw", "ran", "jumped", "grapes",
            "food", "bone", "tree", "hill", "wanted", "chased", "sheep", "bird",
            "the", "a", "an", "and", "big", "small", "hungry"}


class LlmToolUseTest(unittest.TestCase):
    def test_unavailable_worker_reports_cleanly_without_trials(self):
        report = run(GROUNDED, ["the fox chased the sheep."],
                     ScriptedWorker([], available=False), {})
        self.assertEqual(report["status"], "local_worker_unavailable")
        self.assertEqual(report["trials"], [])

    def test_use_word_task_verifies_a_grounded_parsing_sentence_with_the_word(self):
        task = {"task_type": "use_word", "template_index": 0, "target_word": "fox"}
        extractor = NarrativeEventExtractor("developmental_grounded_18")
        good = verify(task, "The fox ate the food.", GROUNDED, extractor)
        self.assertTrue(good["verified"])
        missing_word = verify(task, "The dog ate the bone.", GROUNDED, extractor)
        self.assertFalse(missing_word["verified"])
        self.assertFalse(missing_word["checks"]["contains_target"])

    def test_use_word_fails_on_ungrounded_vocabulary(self):
        task = {"task_type": "use_word", "template_index": 0, "target_word": "fox"}
        extractor = NarrativeEventExtractor("developmental_grounded_18")
        result = verify(task, "The fox perambulated the crepuscular thoroughfare.",
                        GROUNDED, extractor)
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["grounded_vocabulary"])

    def test_simplify_verifies_shorter_grounded_parsing_output(self):
        task = {"task_type": "simplify", "template_index": 1,
                "source_sentence": "The ravenous fox perceived the succulent grapes upon the vine."}
        extractor = NarrativeEventExtractor("developmental_grounded_18")
        result = verify(task, "The fox saw the grapes.", GROUNDED, extractor)
        self.assertTrue(result["checks"]["grounded_vocabulary"])
        self.assertTrue(result["checks"]["not_longer"])
        self.assertTrue(result["verified"])

    def test_run_records_verified_rate_per_template_and_accumulates(self):
        worker = ScriptedWorker(["The fox saw the food.", "The wolf chased the sheep."])
        first = run(GROUNDED, ["the ravenous fox perceived the grapes and wanted them badly today."],
                    worker, {})
        self.assertEqual(first["status"], "ran")
        self.assertEqual(first["tasks_this_run"], len(first["trials"]))
        for bucket in first["template_performance"].values():
            self.assertIn("verified_rate", bucket)
        worker2 = ScriptedWorker(["The cat ran up the tree.", "The bird saw the cat."])
        second = run(GROUNDED, ["a big hungry wolf came down the hill and saw the small sheep there."],
                     worker2, first)
        total = sum(b["attempts"] for b in second["template_performance"].values())
        self.assertGreater(total, sum(b["attempts"] for b in first["template_performance"].values()))

    def test_template_rotation_advances_each_run(self):
        previous = {}
        formulate_tasks(GROUNDED, ["the fox chased the sheep across the wide field today at dawn."], previous)
        first_rotation = dict(previous["template_rotation"])
        formulate_tasks(GROUNDED, ["the fox chased the sheep across the wide field today at dawn."], previous)
        self.assertNotEqual(previous["template_rotation"], first_rotation)

    def test_output_never_marked_as_a_belief(self):
        report = run(GROUNDED, ["the ravenous fox perceived the shining grapes and longed for them."],
                     ScriptedWorker(["The fox saw the grapes.", "The wolf saw the sheep."]), {})
        self.assertIn("never updates a belief", report["note"])


if __name__ == "__main__":
    unittest.main()
