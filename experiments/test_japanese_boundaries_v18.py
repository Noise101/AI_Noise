import unittest

from japanese_boundaries_v18 import BoundaryInducer, JapaneseBoundaryAgent, JapaneseStory


class FakeStorySource:
    def search(self, query):
        return JapaneseStory("話", "https://story.example",
                             "きつねはぶどうをみた。きつねはつるをみた。つるは歩いた。きつねは走った。", query)


class FakeReference:
    def __init__(self, name, known):
        self.name, self.known = name, known

    def lookup(self, form):
        if form not in self.known:
            return None
        return {"source": self.name, "url": f"https://{self.name}.example/{form}"}


class AmbiguousReference(FakeReference):
    def lookup(self, form):
        result = super().lookup(form)
        if result:
            result["ambiguous"] = True
        return result


class JapaneseBoundaryTest(unittest.TestCase):
    def test_discovers_repeated_chunks_without_given_tokenizer(self):
        inducer = BoundaryInducer()
        inducer.observe("きつねはぶどうをみた。きつねはつるをみた。")
        forms = {candidate.form for candidate in inducer.candidates()}
        self.assertIn("きつね", forms)
        self.assertIn("みた", forms)

    def test_web_references_separate_words_from_frequent_fragments(self):
        agent = JapaneseBoundaryAgent(FakeStorySource(), [
            FakeReference("dictionary", {"きつね"}), FakeReference("encyclopedia", {"きつね"}),
        ])
        report = agent.learn("きつね つる", candidate_limit=30)
        fox = next(item for item in report["accepted_words"] if item["form"] == "きつね")
        self.assertEqual(fox["status"], "corroborated_boundary")
        rejected = [item for item in report["checked_candidates"] if item["form"] == "つね"]
        if rejected:
            self.assertEqual(rejected[0]["status"], "unvalidated_chunk")
        self.assertTrue(all(item["status"] == "corroborated_boundary" for item in report["accepted_words"]))

    def test_valid_boundary_can_keep_ambiguous_meaning(self):
        agent = JapaneseBoundaryAgent(FakeStorySource(), [
            FakeReference("dictionary", {"つる"}), AmbiguousReference("encyclopedia", {"つる"}),
        ])
        report = agent.learn("きつね つる", candidate_limit=30)
        crane = next(item for item in report["accepted_words"] if item["form"] == "つる")
        self.assertEqual(crane["status"], "corroborated_boundary")
        self.assertEqual(crane["meaning_status"], "ambiguous_reference")

    def test_search_query_comes_from_seed_not_fixed_page(self):
        agent = JapaneseBoundaryAgent(FakeStorySource(), [])
        report = agent.learn("ねこ ねずみ", candidate_limit=2)
        self.assertEqual(report["generated_query"], "ねこ ねずみ")
        self.assertNotIn("http", report["generated_query"])


if __name__ == "__main__":
    unittest.main()
