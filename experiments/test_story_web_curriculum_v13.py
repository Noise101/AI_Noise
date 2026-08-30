import unittest

from story_learning_v12 import StoryLearner
from story_web_curriculum_v13 import StoryCurriculumAgent, StoryDocument


class FakeSource:
    def __init__(self, name, text):
        self.name, self.text = name, text

    def search(self, query):
        return StoryDocument(self.name, f"{self.name} tale", f"https://{self.name}.example/tale",
                             self.text, query, 0.8, "public domain fixture")


class StoryWebCurriculumTest(unittest.TestCase):
    def test_generates_query_from_detected_gap(self):
        agent = StoryCurriculumAgent([FakeSource("one", "Fox sees grapes. Fox jumps high.")])
        report = agent.investigate("sharing food")
        self.assertEqual(report["gap"]["kind"], "new_child_concept")
        self.assertIn("sharing", report["generated_query"])
        self.assertNotIn("https://", report["generated_query"])

    def test_requires_two_sources_for_provisional_conclusion(self):
        one = StoryCurriculumAgent([FakeSource("one", "Fox sees grapes. Fox jumps high.")])
        self.assertEqual(one.investigate("fox grapes")["conclusion"]["status"], "uncertain")
        two = StoryCurriculumAgent([
            FakeSource("one", "Fox sees grapes. Fox jumps high."),
            FakeSource("two", "Fox sees grapes. Fox waits quietly."),
        ])
        report = two.investigate("fox grapes")
        self.assertEqual(report["conclusion"]["status"], "provisional")
        self.assertEqual(report["independent_sources"], 2)
        self.assertTrue(all(item["sha256"] for item in report["sources_found"]))

    def test_unexplained_surprise_becomes_next_search_gap(self):
        learner = StoryLearner()
        learner.observe_story(["Fox sees grapes.", "Fox jumps high."])
        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        agent = StoryCurriculumAgent([], learner)
        gap = agent.detect_gap("ignored seed")
        self.assertEqual(gap["kind"], "unexplained_surprise")
        self.assertIn("waits", agent.make_query(gap))

    def test_curriculum_repeats_from_its_new_gap_and_keeps_provenance(self):
        agent = StoryCurriculumAgent([
            FakeSource("one", "Fox sees grapes. Fox jumps high."),
            FakeSource("two", "Fox sees grapes. Fox waits quietly."),
        ])
        report = agent.run_curriculum("fox grapes", rounds=2)
        self.assertEqual(len(report["cycles"]), 2)
        self.assertEqual(report["cycles"][1]["gap"]["kind"], "unexplained_surprise")
        first_sources = report["cycles"][0]["sources_found"]
        self.assertTrue(first_sources[0]["learned_events"])
        self.assertTrue(first_sources[0]["passage_sha256"])
        self.assertTrue(any(item["status"] == "duplicate_skipped" for item in report["search_history"]))

    def test_rejects_license_boilerplate_and_irrelevant_text(self):
        text = ("Project Gutenberg ebook license. Public domain false false. "
                "Mouse eats cheese. Fox sees grapes. Fox jumps toward grapes.")
        passage = StoryCurriculumAgent.select_passage(text, "fox grapes fable")
        self.assertTrue(passage)
        self.assertTrue(all("Gutenberg" not in sentence for sentence in passage))
        self.assertTrue(all("public domain" not in sentence.lower() for sentence in passage))
        self.assertEqual(StoryCurriculumAgent.select_passage("Mouse eats cheese.", "fox grapes fable"), [])

    def test_transparent_web_parser_rejects_titles_and_finds_action(self):
        self.assertIsNone(StoryCurriculumAgent.parse_child_event("Three Hundred Aesop's Fables."))
        event = StoryCurriculumAgent.parse_child_event("A famished fox saw ripe grapes on a vine.")
        self.assertEqual(event.key, "fox|saw|ripe_grapes_on_vine")

    def test_passage_stops_at_next_story_heading(self):
        text = ("The Fox and the Grapes. Fox saw ripe grapes. Fox jumped and missed. "
                "The Peacock and Juno. Peacock asked for a voice. Juno refused him.")
        passage = StoryCurriculumAgent.select_passage(text, "fox grapes fable")
        self.assertIn("Fox jumped and missed.", passage)
        self.assertNotIn("Peacock asked for a voice.", passage)


if __name__ == "__main__":
    unittest.main()
