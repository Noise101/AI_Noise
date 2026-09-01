import unittest

from narrative_event_v29 import NarrativeEventExtractor
from parser_self_revision_v38 import evaluate_policy, revise_parser


class ParserSelfRevisionTest(unittest.TestCase):
    def test_clause_head_avoids_prepositional_object_as_subject(self):
        sentence = "The beggar by the pool pulled his cowl down."
        self.assertEqual(NarrativeEventExtractor("baseline").extract(sentence).event.subject, "pool")
        self.assertEqual(NarrativeEventExtractor("clause_head").extract(sentence).event.subject,
                         "beggar")

    def test_compact_policy_keeps_first_observed_object_candidate(self):
        result = NarrativeEventExtractor("compact_roles").extract(
            "A fox saw ripe grapes on a vine.")
        self.assertEqual(result.event.key, "fox|saw|ripe")

    def test_no_policy_is_adopted_without_holdout_improvement(self):
        frames = {}
        previous = None
        for index in range(40):
            identity = f"f{index}"
            next_id = f"f{index + 1}" if index < 39 else None
            frames[identity] = {"observation": {"sentence": f"The fox saw food {index}."},
                                "sequence": {"next_frame": next_id}}
        report = revise_parser(frames, previous)
        self.assertEqual(report["selected_policy"], "baseline")
        self.assertEqual(report["selection_status"],
                         "baseline_retained_no_candidate_improved")
        self.assertGreaterEqual(evaluate_policy(frames, "baseline")["total"], 1)


if __name__ == "__main__":
    unittest.main()
