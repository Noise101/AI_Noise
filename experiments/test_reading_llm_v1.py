import unittest

import reading_llm_v1 as rl


class FakeWorker:
    def __init__(self, sentences):
        self._sentences = sentences

    def available(self):
        return True

    def simplify(self, story):
        return list(self._sentences)


GOOD = [
    "こうもりが あそびまわって いました。",
    "こうもりが 地べたに おちました。",
    "いたちが こうもりを つかまえました。",
    "こうもりが いたちに ねずみだと いいました。",
    "いたちが こうもりを たすけました。",
]

HARD_STORY = (
    "あるとき、あそびまわっていたこうもりが、あやまって地べたにおちて、"
    "そこにいたいたちに、つかまってしまいました。"
    "いたちは、こうもりをはなしてやりました。"
)


class ReadingLlmTest(unittest.TestCase):
    def test_verified_simplification_turns_an_unparseable_story_into_events(self):
        result = rl.simplify_story(HARD_STORY, FakeWorker(GOOD))
        self.assertEqual(result["status"], "simplified")
        self.assertTrue(result["verified"])
        self.assertGreaterEqual(result["event_count"], rl.MIN_EVENTS)
        subjects = {e["subject"] for e in result["events"]}
        self.assertTrue({"こうもり", "いたち"} & subjects)

    def test_spacing_is_harvested_as_word_boundaries(self):
        known = rl._harvest_known_words(GOOD)
        self.assertIn("こうもり", known)
        self.assertIn("いたち", known)

    def test_over_compressed_rephrasing_is_rejected(self):
        result = rl.simplify_story(HARD_STORY * 3, FakeWorker(["ねこが ねました。"]))
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["parses_to_events"])

    def test_no_response_is_handled(self):
        result = rl.simplify_story(HARD_STORY, FakeWorker([]))
        self.assertEqual(result["status"], "no_response")

    def test_content_preservation_check_rejects_a_rewrite_that_drops_the_nouns(self):
        # keeps the SOV shape but about entirely different things
        off_topic = ["いぬが ボールを なげました。", "ねこが みずを のみました。",
                     "とりが そらを とびました。"]
        source = ("こうもりが地べたにおちました。いたちがこうもりをつかまえました。"
                  "こうもりがいたちに命乞いをしました。いたちがこうもりをたすけました。")
        result = rl.simplify_story(source, FakeWorker(off_topic))
        self.assertFalse(result["checks"]["keeps_content"])
        self.assertEqual(result["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
