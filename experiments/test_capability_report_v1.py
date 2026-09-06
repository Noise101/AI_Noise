import unittest

from capability_report_v1 import _slope, build_capability_report


def event_structure(*, confirmed=False, curve=None):
    selected = ({"task": "event_plausibility", "final": {"lift": 24}} if confirmed else None)
    return {
        "selected": selected,
        "learning_curve_trend": "improving" if curve else "insufficient_data",
        "learning_curve": curve or [],
        "evaluations": [
            {"task": "verb_cloze", "selection": {"correct": 30, "baseline_correct": 33,
                                                 "total": 260, "lift": -3, "coverage": 0.6,
                                                 "one_sided_sign_p": 0.9}},
            {"task": "event_plausibility", "selection": {"correct": 150, "baseline_correct": 130,
                                                        "total": 260, "lift": 20, "coverage": 0.6,
                                                        "one_sided_sign_p": 0.0002}},
        ]}


class CapabilityReportTest(unittest.TestCase):
    def test_slope_of_a_rising_line(self):
        self.assertAlmostEqual(_slope([(0, 0), (10, 5), (20, 10)]), 0.5)
        self.assertEqual(_slope([(5, 3)]), 0.0)

    def test_confirmed_task_sets_the_gate_and_headline(self):
        report = build_capability_report(
            event_structure(confirmed=True), {"accepted_sentences": 2800},
            {"summary": {"reusable_rules": 0, "evaluation": {}}})
        self.assertTrue(report["capability_gates"]["a_task_beats_baseline_on_frozen_final"])
        self.assertIn("event_plausibility", report["headline"])
        self.assertTrue(report["dimensions"]["event_plausibility"]["confirmed_on_final_split"])
        self.assertEqual(report["dimensions"]["event_plausibility"]["final_lift"], 24)

    def test_learning_efficiency_is_the_lift_slope_against_training_size(self):
        curve = [{"task": "event_plausibility", "training_events": 1000 + 100 * i, "lift": i}
                 for i in range(10)]
        report = build_capability_report(
            event_structure(curve=curve), {}, {})
        # +1 lift per +100 events -> +10 per 1k
        self.assertAlmostEqual(
            report["dimensions"]["event_plausibility"]["lift_per_1k_training_events"], 10.0, places=1)

    def test_flat_curve_over_growing_data_reports_zero_efficiency(self):
        curve = [{"task": "event_plausibility", "training_events": 1000 + 100 * i, "lift": 15}
                 for i in range(20)]
        report = build_capability_report(event_structure(curve=curve), {}, {})
        self.assertEqual(
            report["dimensions"]["event_plausibility"]["lift_per_1k_training_events"], 0.0)

    def test_extraction_delta_is_relative_to_the_previous_report(self):
        previous = {"extraction": {"accepted_sentences": 2700}}
        report = build_capability_report(
            event_structure(), {"accepted_sentences": 2830}, {}, previous)
        self.assertEqual(report["extraction"]["accepted_sentences_delta"], 130)

    def test_sequence_model_slot_is_present_and_defaults_cleanly(self):
        report = build_capability_report(event_structure(), {}, {})
        self.assertIsNone(report["sequence_model"]["held_out_bits_per_char"])
        self.assertEqual(report["sequence_model"]["trend"], "not_yet_measured")
        self.assertFalse(report["capability_gates"]["sequence_model_beats_char_baseline"])
        report2 = build_capability_report(
            event_structure(), {}, {},
            sequence_model={"held_out_bits_per_char": 2.1, "improvement_z": 5.0,
                            "beats_char_baseline": True, "perplexity_trend": "improving",
                            "can_sample": True})
        self.assertTrue(report2["capability_gates"]["sequence_model_beats_char_baseline"])
        self.assertEqual(report2["sequence_model"]["improvement_z"], 5.0)
        self.assertTrue(report2["sequence_model"]["generative"])


if __name__ == "__main__":
    unittest.main()
