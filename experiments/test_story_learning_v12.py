import unittest

from story_learning_v12 import StoryLearner


class StoryLearningTest(unittest.TestCase):
    def test_learns_a_next_event_without_pretrained_model(self):
        learner = StoryLearner()
        learner.observe_story(["Cat sees ball.", "Cat pushes ball."])
        context = learner.parser.parse("Cat sees ball.")
        self.assertEqual(learner.predict(context), "cat|pushes|ball")

    def test_surprise_is_recorded_and_changes_confidence(self):
        learner = StoryLearner()
        familiar = ["Fox sees grapes.", "Fox jumps high."]
        learner.observe_story(familiar)
        learner.observe_story(familiar)
        context = learner.parser.parse("Fox sees grapes.")
        self.assertEqual(learner.confidence(context.key, "fox|jumps|high"), 1.0)

        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        self.assertEqual(len(learner.report()["revisions"]), 1)
        self.assertAlmostEqual(learner.confidence(context.key, "fox|jumps|high"), 2 / 3)

    def test_repeated_counterexample_can_replace_old_prediction(self):
        learner = StoryLearner()
        learner.observe_story(["Fox sees grapes.", "Fox jumps high."])
        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        context = learner.parser.parse("Fox sees grapes.")
        self.assertEqual(learner.predict(context), "fox|waits|quietly")

    def test_surprise_creates_a_why_question(self):
        learner = StoryLearner()
        learner.observe_story(["Fox sees grapes.", "Fox jumps high."])
        learner.observe_story(["Fox sees grapes.", "Fox waits quietly."])
        questions = learner.report()["why_questions"]
        self.assertEqual(len(questions), 1)
        self.assertIn("Why did", questions[0]["question"])

    def test_why_requires_contrastive_support(self):
        learner = StoryLearner()
        learner.observe_story(["Rain falls softly.", "Seed grows tall."])
        self.assertEqual(learner.ask_why("seed|grows|tall")["answer"], "unknown")

        learner.observe_story(["Rain falls softly.", "Seed grows tall."])
        learner.observe_story(["Sun shines brightly.", "Bird sings loudly."])
        answer = learner.ask_why("seed|grows|tall")
        self.assertEqual(answer["candidate_cause"], "rain|falls|softly")
        self.assertGreater(answer["lift_over_baseline"], 0)
        self.assertIn("not yet a proven cause", answer["warning"])


if __name__ == "__main__":
    unittest.main()
