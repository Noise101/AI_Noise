import unittest
import json
import tempfile
from pathlib import Path

from developmental_curriculum_v32 import assess_source_quality, rebuild_developmental_memory


def report(sentences, accepted):
    audit = [{"sentence": sentence, "accepted": ok,
              "event": (f"fox|acts|item{index}" if ok else None)}
             for index, (sentence, ok) in enumerate(zip(sentences, accepted))]
    return {"knowledge": {"bootstrap": {"sources": [{
        "url": "https://source/story", "event_extraction_audit": audit}]}}}


class DevelopmentalCurriculumTest(unittest.TestCase):
    def test_short_recurrent_story_is_admitted(self):
        item = report(["Fox sees food.", "Fox waits.", "Fox eats food."], [True, True, True])
        result = assess_source_quality(item, {"fox", "sees", "food", "waits", "eats"})
        self.assertEqual(result["status"], "developmental_passage")
        self.assertTrue(result["admit_to_global_memory"])

    def test_long_expository_page_is_preserved_but_not_learned(self):
        long = "The appointed minister administered ecclesiastical institutions throughout the extensive kingdom with several ceremonial authorities."
        item = report([long, long, "A king acted."], [False, False, True])
        result = assess_source_quality(item, {"the", "king"})
        self.assertEqual(result["status"], "outside_current_level")
        self.assertFalse(result["admit_to_global_memory"])
        self.assertTrue(result["reasons"])

    def test_short_fable_with_novel_words_is_not_penalized_for_learning(self):
        sentences = ["Young turkeys saw a fox near the tree.",
                     "The turkeys waited under the leaves.",
                     "The turkeys ran when the fox moved."]
        item = report(sentences, [True, True, True])
        result = assess_source_quality(item, {"the", "a", "near", "under", "when"})
        self.assertEqual(result["status"], "developmental_passage")
        self.assertGreaterEqual(result["metrics"]["subject_recurrence"], 0.15)

    def test_short_biographical_fragments_without_recurrence_are_rejected(self):
        item = {"knowledge": {"bootstrap": {"sources": [{"url": "https://bio",
            "event_extraction_audit": [
                {"sentence": "John painted a wall.", "accepted": True, "event": "john|painted|wall"},
                {"sentence": "Mary wrote a book.", "accepted": True, "event": "mary|wrote|book"},
                {"sentence": "Thomas built a house.", "accepted": True, "event": "thomas|built|house"},
            ]}]}}}
        result = assess_source_quality(item, {"a", "wall", "book", "house"})
        self.assertEqual(result["status"], "outside_current_level")
        self.assertIn("no recurring subject or dialogue structure", result["reasons"])

    def test_unaudited_legacy_source_cannot_enter_new_memory(self):
        result = assess_source_quality({"knowledge": {"bootstrap": {"sources": []}}})
        self.assertEqual(result["status"], "not_yet_audited")
        self.assertFalse(result["admit_to_global_memory"])

    def test_rebuild_archives_old_memory_and_keeps_only_admitted_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "global-language-memory.json").write_text(json.dumps({
                "words": {"fox": {"curricula": 3}}}), encoding="utf-8")
            good = report(["Fox sees food.", "Fox waits.", "Fox eats food."],
                          [True, True, True])
            good["state"] = {"seed": "fox food"}
            good["knowledge"].update({"lexicon": {}, "story": {}, "concepts": {}})
            (runtime / "latest-report.json").write_text(json.dumps(good), encoding="utf-8")
            (runtime / "curriculum-state.json").write_text(json.dumps({
                "completed_seeds": ["old bad"], "deferred_seeds": [],
                "frontier": [{"seed": "child", "parent_url": "https://source/story"}]}),
                encoding="utf-8")
            result = rebuild_developmental_memory(runtime)
            self.assertEqual(result["admitted_reports"], 1)
            self.assertTrue(Path(result["archive"]).exists())
            rebuilt = json.loads((runtime / "global-language-memory.json").read_text())
            self.assertEqual(rebuilt["merged_seeds"], ["fox food"])
            curriculum = json.loads((runtime / "curriculum-state.json").read_text())
            self.assertEqual(curriculum["completed_seeds"], ["fox food"])
            self.assertTrue((runtime / "archive" / "curriculum-state-pre-v32.json").exists())


if __name__ == "__main__":
    unittest.main()
