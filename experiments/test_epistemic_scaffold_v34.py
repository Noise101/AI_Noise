import tempfile
import unittest
from pathlib import Path

from epistemic_scaffold_v34 import empty_scaffold, observe_report, rebuild_scaffold


def report(seed="fox grapes"):
    return {"state": {"seed": seed}, "knowledge": {"bootstrap": {"sources": [{
        "url": "https://source/story", "event_extraction_audit": [
            {"accepted": True, "event": "fox|saw|grapes", "sentence": "Fox saw grapes.",
             "quality": 0.8},
            {"accepted": True, "event": "fox|said|grapes", "sentence": "Fox said grapes.",
             "quality": 0.7},
        ]}]}}}


class EpistemicScaffoldTest(unittest.TestCase):
    def test_records_observation_but_invents_no_interpretation(self):
        scaffold = empty_scaffold()
        self.assertTrue(observe_report(scaffold, "fox grapes", report()))
        frames = list(scaffold["frames"].values())
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(item["interpretations"] == [] for item in frames))
        self.assertTrue(all(item["hypotheses"] == [] for item in frames))
        self.assertTrue(all(item["applicability"]["generalization_allowed"] is False
                            for item in frames))
        self.assertEqual(scaffold["summary"]["decision_influence"], False)

    def test_keeps_speech_content_unknown_and_links_sequence(self):
        scaffold = empty_scaffold()
        observe_report(scaffold, "fox grapes", report())
        frames = list(scaffold["frames"].values())
        self.assertEqual(frames[0]["sequence"]["next_frame"], frames[1]["frame_id"])
        self.assertIsNone(frames[1]["utterance"]["propositional_content"])

    def test_duplicate_curriculum_is_not_observed_twice(self):
        scaffold = empty_scaffold()
        observe_report(scaffold, "fox grapes", report())
        self.assertFalse(observe_report(scaffold, "fox grapes", report()))
        self.assertEqual(len(scaffold["frames"]), 2)

    def test_rebuild_uses_only_admitted_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            path = runtime / "seeds" / "one" / "latest-report.json"
            path.parent.mkdir(parents=True)
            import json
            path.write_text(json.dumps(report()), encoding="utf-8")
            self.assertEqual(rebuild_scaffold(runtime, set())["summary"]["observation_frames"], 0)
            self.assertEqual(rebuild_scaffold(runtime, {"fox grapes"})["summary"]["observation_frames"], 2)


if __name__ == "__main__":
    unittest.main()
