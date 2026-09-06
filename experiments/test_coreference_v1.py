import unittest

from coreference_v1 import resolve_document


class CoreferenceTest(unittest.TestCase):
    def test_pronoun_resolves_to_recent_compatible_animate_entity(self):
        doc = resolve_document([
            "The fox saw the grapes.",
            "He wanted the fruit.",
            "He jumped high."])
        self.assertEqual(doc.subject_hints[0], "fox")
        self.assertEqual(doc.subject_hints[1], "fox")
        self.assertEqual(doc.subject_hints[2], "fox")
        self.assertGreaterEqual(doc.resolutions, 2)

    def test_number_agreement_blocks_a_singular_antecedent_for_they(self):
        doc = resolve_document([
            "A wolf met the travellers.",
            "They ran away."])
        # "they" is plural; "wolf" is singular -> "travellers" (plural) wins
        self.assertEqual(doc.subject_hints[1], "travellers")

    def test_antecedent_survives_an_intervening_unparseable_sentence(self):
        doc = resolve_document([
            "The hunter set a trap.",
            "Deep in the ancient forest, beyond the river, under a grey and clouded sky.",
            "He waited."])
        self.assertEqual(doc.subject_hints[2], "hunter")

    def test_sentence_initial_adverb_is_not_registered_as_an_entity(self):
        doc = resolve_document([
            "The lion slept.",
            "Oftentimes he roared."])
        self.assertIsNone(doc.subject_hints[1] == "oftentimes" or None)
        self.assertNotIn("oftentimes", doc.entity_names)
        self.assertEqual(doc.subject_hints[1], "lion")

    def test_mid_sentence_lowercase_abstract_noun_is_not_an_entity(self):
        doc = resolve_document([
            "The fox felt a great anger.",
            "It could not reach the grapes."])
        self.assertNotIn("anger", doc.entity_names)
        # "it" resolves to the animate/neuter fox, not to "anger" or "grapes"
        self.assertEqual(doc.subject_hints[1], "fox")

    def test_proper_noun_without_a_determiner_is_an_entity(self):
        doc = resolve_document([
            "Pygmalion carved a statue.",
            "He admired his work."])
        self.assertEqual(doc.subject_hints[0], "pygmalion")
        self.assertEqual(doc.subject_hints[1], "pygmalion")

    def test_object_pronoun_resolves_to_a_different_entity_than_the_subject(self):
        doc = resolve_document([
            "The wolf saw a lamb.",
            "The lamb drank.",
            "The wolf chased her."])
        self.assertEqual(doc.object_hints[2], "lamb")

    def test_possessive_is_stripped_from_the_entity_head(self):
        doc = resolve_document([
            "Tilda's heart raced.",
            "She ran."])
        self.assertIn("tilda", doc.entity_names)
        self.assertEqual(doc.subject_hints[1], "tilda")

    def test_recency_window_expires_an_old_antecedent(self):
        doc = resolve_document([
            "The fox spoke.",
            "The weather turned.",
            "The road bent north.",
            "The market opened early.",
            "He smiled."])
        # "fox" is now 4 sentences back, past the default window -> no hint
        self.assertIsNone(doc.subject_hints[4])


if __name__ == "__main__":
    unittest.main()
