import unittest

import semantic_representation_v1 as sem


def ev(subject, verb, obj=""):
    return {"subject": subject, "verb": verb, "obj": obj,
            "provenance": "heuristic_self"}


class SemanticRepresentationTests(unittest.TestCase):
    def test_learns_only_self_observed_usage_and_deduplicates_sources(self):
        store = {
            "a": [ev("きつね", "食べる", "ぶどう"), ev("きつね", "走る")],
            "teacher": [{"subject": "答え", "verb": "教える", "obj": "分類",
                         "provenance": "teacher"}],
        }
        state = sem.learn(store, {}, None, parser_version=9, cycle=1)
        self.assertIn("きつね", state["word_vectors"])
        self.assertNotIn("答え", state["word_vectors"])
        self.assertFalse(any("genus" in c or "分類" in c for c in state["context_vectors"]))
        pairs = state["pairs_seen"]
        again = sem.learn(store, {}, state, parser_version=9, cycle=2)
        self.assertEqual(again["pairs_seen"], pairs)
        self.assertEqual(again["sources_trained_this_cycle"], 0)

    def test_parser_change_resets_derived_vectors(self):
        state = sem.learn({"a": [ev("きつね", "走る")]}, {}, None, 8, 1)
        reset = sem.learn({}, {}, state, 9, 2)
        self.assertEqual(reset["parser_version"], 9)
        self.assertEqual(reset["word_count"], 0)
        self.assertIn("parser_version_changed", reset["reset_reason"])

    def test_nearest_anchor_is_a_bounded_proposal(self):
        state = {"word_vectors": {
            "仔犬": [1.0, 0.0], "子犬": [0.99, 0.01],
            "子猫": [0.96, 0.04], "狐猿": [0.93, 0.07], "机台": [0.0, 1.0]}}
        wm = {"beliefs": {
            "子犬": {"understood": True, "genus": "生き物", "confidence": .9},
            "子猫": {"understood": True, "genus": "生き物", "confidence": .9},
            "狐猿": {"understood": True, "genus": "生き物", "confidence": .9},
            "机台": {"understood": True, "genus": "道具", "confidence": .9}}}
        got = sem.infer_genus("仔犬", state, wm)
        self.assertEqual(got["genus"], "生き物")
        self.assertGreaterEqual(got["support"], 3)
        self.assertLessEqual(got["confidence"], .65)

    def test_diagnostic_explicitly_cannot_confer_capability(self):
        report = sem.learn({}, {}, None, 9, 1)
        self.assertEqual(report["diagnostic"]["status"], "diagnostic_only")
        self.assertIn("never confers capability", report["diagnostic"]["note"])


if __name__ == "__main__":
    unittest.main()
