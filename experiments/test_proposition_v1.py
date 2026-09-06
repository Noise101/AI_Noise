import unittest

from proposition_v1 import extract_document_propositions, extract_proposition


class PropositionTest(unittest.TestCase):
    def test_copular_property(self):
        p = extract_proposition("The fox was hungry.")
        self.assertEqual((p.subject, p.relation, p.value, p.polarity),
                         ("fox", "is", "hungry", "positive"))

    def test_negated_property(self):
        p = extract_proposition("The grapes were not ripe.")
        self.assertEqual((p.subject, p.relation, p.value, p.polarity),
                         ("grapes", "is-not", "ripe", "negative"))

    def test_possession(self):
        p = extract_proposition("The fox had a cunning plan.")
        self.assertEqual((p.subject, p.relation, p.value), ("fox", "has", "plan"))

    def test_spatial_relation(self):
        p = extract_proposition("The nest was high in the tree.")
        self.assertEqual((p.subject, p.relation, p.value), ("nest", "in", "tree"))

    def test_passive_is_not_a_property(self):
        self.assertIsNone(extract_proposition("The fox was seen by the farmer."))
        self.assertIsNone(extract_proposition("The house was built last year."))

    def test_auxiliary_perfect_is_not_possession(self):
        self.assertIsNone(extract_proposition("The fox had gone away."))

    def test_action_clause_is_left_to_the_event_parser(self):
        self.assertIsNone(extract_proposition("The fox jumped at the grapes."))

    def test_existential_and_pronoun_subjects_are_skipped_without_a_hint(self):
        self.assertIsNone(extract_proposition("There was a great noise."))
        self.assertIsNone(extract_proposition("It was cold."))

    def test_pronoun_subject_uses_the_coreference_hint(self):
        p = extract_proposition("She was afraid.", subject_hint="hen")
        self.assertEqual((p.subject, p.relation, p.value), ("hen", "is", "afraid"))

    def test_document_pass_threads_hints_by_index(self):
        props = extract_document_propositions(
            ["The lion was old.", "He was still strong."],
            subject_hints=["lion", "lion"])
        self.assertEqual([p.key for p in props], ["lion|is|old", "lion|is|strong"])


if __name__ == "__main__":
    unittest.main()
