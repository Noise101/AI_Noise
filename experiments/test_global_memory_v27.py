import unittest

from global_memory_v27 import empty_memory, merge_report, summarize


class GlobalMemoryTest(unittest.TestCase):
    def test_only_audited_events_enter_causal_evidence(self):
        memory = empty_memory()
        report = {"knowledge": {"bootstrap": {"sources": [{
            "learned_events": ["bad|was|metadata"],
            "event_extraction_audit": [
                {"accepted": True, "event": "fox|saw|grapes"},
                {"accepted": False, "event": None, "reason": "metadata"},
                {"accepted": True, "event": "fox|jumped|high"},
            ],
        }]}}}
        merge_report(memory, "one", report)
        self.assertIn("bad|was|metadata", memory["event_counts"])
        self.assertNotIn("bad|was|metadata", memory["quality_event_counts"])
        # Rejected sentences break a narrative sequence; no adjacency is invented across them.
        self.assertEqual(memory["quality_event_transitions"], {})

    def report(self, word_count, transition_count=1):
        return {"knowledge": {"lexicon": {"word_forms": {"fox": word_count},
            "grounded_meanings": [{"form": "fox", "roles": {"agent": word_count}}],
            "researched_meanings": {}, "phrase_candidates": [], "conversation_cues": {}},
            "story": {"rules": [{"when": "fox|sees|bird", "expect": "fox|waits|", 
                                   "observations": transition_count}]},
            "bootstrap": {"sources": [{"learned_events": ["fox|sees|bird"]}]},
            "concepts": {"beliefs": []}}}

    def test_merges_each_seed_once_and_accumulates_language(self):
        memory = empty_memory()
        self.assertTrue(merge_report(memory, "one", self.report(2)))
        self.assertFalse(merge_report(memory, "one", self.report(2)))
        self.assertTrue(merge_report(memory, "two", self.report(3, 2)))
        self.assertEqual(memory["words"]["fox"]["encounters"], 5)
        self.assertEqual(memory["words"]["fox"]["roles"]["agent"], 5)
        self.assertEqual(memory["event_transitions"]["fox|sees|bird"]["fox|waits|"], 3)
        self.assertEqual(summarize(memory)["curricula"], 2)

    def test_late_research_updates_belief_without_double_counting_seed(self):
        memory = empty_memory()
        first = self.report(2)
        merge_report(memory, "one", first)
        later = self.report(2)
        later["knowledge"]["lexicon"]["researched_meanings"] = {
            "fox": {"accepted_sense": "animal", "status": "corroborated"}}
        self.assertFalse(merge_report(memory, "one", later))
        self.assertEqual(memory["words"]["fox"]["encounters"], 2)
        self.assertEqual(memory["words"]["fox"]["accepted_belief"]["accepted_sense"], "animal")


if __name__ == "__main__":
    unittest.main()
