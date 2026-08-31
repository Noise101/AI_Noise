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
