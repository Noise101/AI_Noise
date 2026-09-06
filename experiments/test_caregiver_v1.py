import unittest

import caregiver_v1 as cg


def story(url, subjects, verbs):
    events = [{"subject": s, "verb": v, "obj": "", "confidence": 0.9}
              for s, v in zip(subjects, verbs)]
    return {"title": url, "url": f"http://b/{url}", "events": events}


class CaregiverTest(unittest.TestCase):
    def test_due_respects_the_interval_and_no_double_batch(self):
        state = cg.empty_state()
        self.assertFalse(cg.due(state, cycle=cg.CAREGIVER_INTERVAL - 1))
        self.assertTrue(cg.due(state, cycle=cg.CAREGIVER_INTERVAL))
        cg.open_batch(state, [{"id": "x", "kind": "order", "options": ["はい", "いいえ"],
                               "answer_story": 1, "cycle": 1, "prompt": "?"}], cycle=cg.CAREGIVER_INTERVAL)
        self.assertFalse(cg.due(state, cycle=cg.CAREGIVER_INTERVAL * 3))

    def test_generates_a_gradeable_protagonist_question(self):
        recent = [story("kitsune", ["きつね", "きつね", "からす", "きつね"],
                        ["みつける", "ほしがる", "とぶ", "あきらめる"])]
        qs = cg.generate_questions(recent, cycle=15)
        self.assertEqual(qs[0]["kind"], "protagonist")
        self.assertIn("きつね", qs[0]["options"])
        # the story answer points at the most frequent subject
        self.assertEqual(qs[0]["options"][qs[0]["answer_story"] - 1], "きつね")

    def test_apply_answers_is_positional_and_grades_against_the_story(self):
        recent = [story("a", ["ねこ", "ねこ", "いぬ"], ["あるく", "みる", "はしる"]),
                  story("b", ["とり", "とり", "むし"], ["とぶ", "たべる", "にげる"])]
        state = cg.empty_state()
        qs = cg.generate_questions(recent, cycle=15)
        cg.open_batch(state, qs, cycle=15)
        right = qs[0]["answer_story"]
        result = cg.apply_answers(state, f"{right} 9")   # 2nd answer invalid -> stays pending
        self.assertEqual(result["graded"], 1)
        self.assertEqual(result["matched_story"], 1)
        self.assertEqual(len(state["pending"]), 1)
        self.assertEqual(cg.summary(state)["human_story_agreement"], 1.0)

    def test_yes_no_question_accepts_japanese_and_ascii(self):
        q = {"id": "q", "kind": "order", "options": ["はい", "いいえ"],
             "answer_story": 1, "cycle": 1, "prompt": "?"}
        state = cg.empty_state()
        cg.open_batch(state, [q], cycle=15)
        self.assertEqual(cg.apply_answers(state, "はい")["matched_story"], 1)

    def test_stale_batch_expires(self):
        state = cg.empty_state()
        cg.open_batch(state, [{"id": "q", "kind": "order", "options": ["はい", "いいえ"],
                               "answer_story": 1, "cycle": 1, "prompt": "?"}], cycle=1)
        self.assertTrue(cg.expire_if_stale(state, cycle=1 + cg.QUESTION_TTL_CYCLES + 1))
        self.assertEqual(state["pending"], [])


if __name__ == "__main__":
    unittest.main()
