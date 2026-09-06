import unittest
import json
import tempfile
from pathlib import Path

from developmental_curriculum_v32 import (DEVELOPMENTAL_SCORE_THRESHOLD, KNOWN_RATIO_FLOOR,
                                          assess_source_quality, rebuild_developmental_memory)


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

    def test_fully_familiar_vocabulary_is_not_penalised(self):
        # Every word already known: under the old tent curve (peak 0.65) this
        # would score vocabulary_fit ~0.46; the floor curve gives it full credit.
        item = report(["Fox sees food.", "Fox waits.", "Fox eats food."], [True, True, True])
        result = assess_source_quality(item, {"fox", "sees", "food", "waits", "eats"})
        self.assertEqual(result["metrics"]["known_word_ratio"], 1.0)
        self.assertEqual(result["metrics"]["vocabulary_fit"], 1.0)
        self.assertTrue(result["admit_to_global_memory"])

    def test_vocabulary_below_the_floor_is_still_penalised_proportionally(self):
        # Mostly-unknown text: known_ratio well below KNOWN_RATIO_FLOOR, so
        # vocabulary_fit scales down linearly rather than getting full credit.
        sentences = ["Zorg flonk wibble grib.", "Zorg zonk wibble grib.", "Zorg plork wibble grib."]
        item = {"knowledge": {"bootstrap": {"sources": [{"url": "https://x",
            "event_extraction_audit": [
                {"sentence": s, "accepted": True, "event": f"zorg|acts|grib{i}"}
                for i, s in enumerate(sentences)]}]}}}
        result = assess_source_quality(item, {"zorg"})  # only "zorg" known -> 3/12 = 0.25
        kr = result["metrics"]["known_word_ratio"]
        self.assertLess(kr, KNOWN_RATIO_FLOOR)
        self.assertAlmostEqual(result["metrics"]["vocabulary_fit"], kr / KNOWN_RATIO_FLOOR, places=3)
        self.assertLess(result["metrics"]["vocabulary_fit"], 1.0)

    def test_score_threshold_is_the_raised_value_and_named_in_the_reason(self):
        self.assertEqual(DEVELOPMENTAL_SCORE_THRESHOLD, 0.68)
        long = ("The appointed minister administered ecclesiastical institutions throughout "
                "the extensive kingdom with several ceremonial authorities.")
        result = assess_source_quality(report([long, long, "A king acted."],
                                              [False, False, True]), {"the", "king"})
        self.assertLess(result["score"], 0.68)
        self.assertIn("combined developmental score below 0.68", result["reasons"])

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
