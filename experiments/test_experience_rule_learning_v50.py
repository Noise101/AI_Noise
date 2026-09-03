import unittest
from unittest.mock import patch

from experience_rule_learning_v50 import learn_experience_rules


class ExperienceRuleLearningTest(unittest.TestCase):
    def test_only_independently_verified_dialogue_structure_is_admitted(self):
        dialogue = {"expressions": {
            "with the": {"verification_id": "ok", "independent_sources": 2,
                "hypothesis_status": "supported_structural_candidate_not_meaning_proof",
                "structural_hypothesis": {"predicted_role": "relation_between_neighboring_entities"},
                "source_urls": ["a", "b"], "contexts": [{"before": ["walk"], "after": ["dog"]}]},
            "made up": {"verification_id": "no", "independent_sources": 0,
                "hypothesis_status": "unresolved"}}}
        result = learn_experience_rules({"sequences": []}, dialogue_verification=dialogue)
        self.assertEqual(result["summary"]["dialogue_structures"], 1)
        self.assertEqual(result["dialogue_frames"][0]["origin"],
                         "web_contexts_not_local_partner")

    @patch("experience_rule_learning_v50._source_holdout", side_effect=lambda source: source == "test")
    def test_structures_compares_tests_and_retains_counterexamples(self, _split):
        sequences = []
        for source in ("a", "b", "c"):
            sequences.append({"source_url": source, "seed": source,
                              "events": [f"fox_{source}|sees|food", f"fox_{source}|takes|food"]})
            sequences.append({"source_url": source, "seed": source,
                              "events": [f"fox_{source}|sees|water", f"fox_{source}|drinks|water"]})
        sequences.append({"source_url": "test", "seed": "test",
                          "events": ["fox|sees|food", "fox|runs|away"]})
        result = learn_experience_rules({"sequences": sequences})
        self.assertGreater(result["summary"]["structured_experiences"], 0)
        self.assertGreater(result["summary"]["comparison_groups"], 0)
        self.assertEqual(result["summary"]["evaluation"]["total"], 1)
        self.assertTrue(any(rule["known_counterexamples"] for rule in result["rules"]))
        self.assertFalse(result["causal_credit"])

    @patch("experience_rule_learning_v50._source_holdout", side_effect=lambda source: source == "test")
    def test_status_revision_is_remembered(self, _split):
        verified = {"sequences": [{"source_url": "a", "events": ["fox|sees|x", "fox|takes|x"]},
                                   {"source_url": "test", "events": ["fox|sees|x", "fox|runs|x"]}]}
        first = learn_experience_rules(verified)
        rule = first["rules"][0]
        old = {"rules": [{**rule, "status": "reusable"}], "revision_history": []}
        second = learn_experience_rules(verified, old)
        self.assertEqual(second["revision_history"][0]["before"], "reusable")


if __name__ == "__main__":
    unittest.main()
