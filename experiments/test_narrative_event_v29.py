import unittest

from narrative_event_v29 import NarrativeEventExtractor


class NarrativeEventExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = NarrativeEventExtractor()

    def test_extracts_explicit_child_event(self):
        result = self.extractor.extract("A famished fox saw ripe grapes on a vine.")
        self.assertTrue(result.accepted)
        self.assertEqual(result.event.key, "fox|saw|ripe_grapes_on_vine")

    def test_normalizes_auxiliary_to_observable_action(self):
        result = self.extractor.extract("The fox was running toward the tree.")
        self.assertEqual(result.event.key, "fox|running|toward_tree")

    def test_rejects_bibliographic_sentence_with_reason(self):
        result = self.extractor.extract("Translated by George Fyler Townsend in 1867.")
        self.assertFalse(result.accepted)
        self.assertTrue(result.reason.startswith("metadata:"))

    def test_rejects_unresolved_pronoun_instead_of_merging_entities(self):
        result = self.extractor.extract("He jumped over the stream.")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "unresolved_pronoun_subject")

    def test_resolves_only_a_recent_explicit_subject(self):
        results = self.extractor.extract_sequence([
            "A fox saw a stream.", "He jumped over it."])
        self.assertEqual(results[1].event.subject, "fox")
        self.assertEqual(results[1].reason, "accepted_with_local_coreference")

    def test_rejection_is_serializable_for_audit(self):
        record = self.extractor.extract("The Complete Index:").record()
        self.assertFalse(record["accepted"])
        self.assertIsNone(record["event"])

    def test_preposition_cannot_be_promoted_to_subject(self):
        result = self.extractor.extract("At changed moon near water.")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "invalid_structural_subject")


if __name__ == "__main__":
    unittest.main()
