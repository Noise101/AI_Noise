import json
import tempfile
import unittest
from pathlib import Path

from autonomous_controller_v20 import AutonomousController, EnglishDevelopmentEnvironment, LearningGap, PersistentState


class FakeEnvironment:
    def __init__(self):
        self.executed = []
        self.available = [
            LearningGap("low", "word", "low query", 0.2, 1, 2, "low value"),
            LearningGap("high", "why", "why query", 1.0, 2, 0, "surprise"),
        ]

    def gaps(self):
        return self.available

    def execute(self, gap):
        self.executed.append(gap.gap_id)
        if gap.layer == "why":
            return {"learned": gap.gap_id, "why": {"candidate_cause": gap.gap_id}}
        return {"learned": gap.gap_id,
                "meaning_belief": {"accepted_sense": gap.gap_id}}

    def snapshot(self):
        return {"executed": self.executed}

    def restore(self, cycles):
        restored = [cycle["result"]["learned"] for cycle in cycles]
        self.available = [gap for gap in self.available if gap.gap_id not in restored]


class AutonomousControllerTest(unittest.TestCase):
    def test_global_transition_prior_is_used_but_not_duplicated_in_local_report(self):
        environment = EnglishDevelopmentEnvironment.__new__(EnglishDevelopmentEnvironment)
        class Story:
            def report(self):
                return {"events_seen": 12, "rules": [
                    {"when": "a", "expect": "b", "observations": 11, "confidence": 1.0},
                    {"when": "b", "expect": "c", "observations": 1, "confidence": 1.0}]}
        environment.agent = type("Agent", (), {"story": Story()})()
        environment.global_transition_baseline = {"a": {"b": 10}}
        environment.global_event_baseline = 10
        report = environment.local_story_report()
        self.assertEqual(report["events_seen"], 2)
        self.assertEqual([rule["observations"] for rule in report["rules"]], [1, 1])
        self.assertFalse(report["global_prior"]["included_in_local_rules"])
    def test_selects_expected_information_gain_not_fixed_order(self):
        environment = FakeEnvironment()
        controller = AutonomousController(environment, PersistentState("unknown seed"))
        report = controller.run(max_steps=1, max_seconds=1)
        self.assertEqual(environment.executed, ["high"])
        self.assertEqual(report["state"]["stop_reason"], "step_budget_exhausted")

    def test_state_resume_does_not_repeat_completed_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first_environment = FakeEnvironment()
            first = AutonomousController.load(first_environment, "seed", path)
            first.run(max_steps=1, max_seconds=1)
            second_environment = FakeEnvironment()
            second = AutonomousController.load(second_environment, "seed", path)
            second.run(max_steps=1, max_seconds=1)
            self.assertEqual(second_environment.executed, ["low"])
            self.assertEqual([gap.gap_id for gap in second_environment.available], ["low"])
            saved = json.loads(path.read_text())
            self.assertEqual(saved["completed_gap_ids"], ["high", "low"])

    def test_time_budget_stops_without_executing(self):
        environment = FakeEnvironment()
        controller = AutonomousController(environment, PersistentState("seed"))
        report = controller.run(max_steps=2, max_seconds=0)
        self.assertEqual(environment.executed, [])
        self.assertEqual(report["state"]["stop_reason"], "time_budget_exhausted")

    def test_ungrounded_curiosity_persists_without_repeating_same_evidence(self):
        environment = FakeEnvironment()
        environment.available = [LearningGap("word:mystery", "word", "query", 1.0, 3, 2, "unknown")]
        def unresolved(gap):
            environment.executed.append(gap.gap_id)
            return {"learned": gap.gap_id, "meaning_belief": {"accepted_sense": None}}
        environment.execute = unresolved
        report = AutonomousController(environment, PersistentState("seed")).run(3, 1)
        self.assertEqual(environment.executed, ["word:mystery"])
        self.assertEqual(report["state"]["stop_reason"], "no_new_evidence_for_unresolved_gap")
        self.assertEqual(report["state"]["curiosity_ledger"]["word:mystery"]["status"],
                         "wanting_to_know")


if __name__ == "__main__":
    unittest.main()
