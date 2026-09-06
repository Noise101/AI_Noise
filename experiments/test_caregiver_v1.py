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

    def test_questions_are_self_contained_and_show_noises_retelling(self):
        recent = [story("kitsune", ["きつね", "きつね", "からす", "きつね"],
                        ["みつける", "ほしがる", "とぶ", "あきらめる"])]
        qs = cg.generate_questions(recent, cycle=15)
        self.assertTrue(qs)
        self.assertIn(qs[0]["kind"], ("protagonist_from_text", "retelling_coherent"))
        self.assertIn("Noise", qs[0]["prompt"])          # judge Noise's output, not recall a book
        self.assertIn("みつけました", qs[0]["prompt"])   # the retelling text is in the question
        if qs[0]["kind"] == "protagonist_from_text":
            self.assertEqual(qs[0]["options"][qs[0]["answer_story"] - 1], "きつね")

    def test_apply_answers_is_positional_and_grades(self):
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

    def test_coherence_question_is_graded_against_the_fidelity_band(self):
        recent = [{"title": "t", "url": "http://b/t", "fidelity": 0.1,
                   "events": [{"subject": "き", "verb": "みる", "obj": ""},
                              {"subject": "き", "verb": "とぶ", "obj": ""},
                              {"subject": "き", "verb": "なく", "obj": ""}]}]
        qs = cg.generate_questions(recent, cycle=15)
        coherence = [q for q in qs if q["kind"] == "retelling_coherent"]
        self.assertTrue(coherence)
        self.assertEqual(coherence[0]["answer_story"], 3)   # low fidelity -> "意味不明"
        state = cg.empty_state()
        cg.open_batch(state, coherence, cycle=15)
        self.assertEqual(cg.apply_answers(state, "3")["matched_story"], 1)

    def test_stale_batch_expires(self):
        state = cg.empty_state()
        cg.open_batch(state, [{"id": "q", "kind": "order", "options": ["はい", "いいえ"],
                               "answer_story": 1, "cycle": 1, "prompt": "?"}], cycle=1)
        self.assertTrue(cg.expire_if_stale(state, cycle=1 + cg.QUESTION_TTL_CYCLES + 1))
        self.assertEqual(state["pending"], [])


if __name__ == "__main__":
    unittest.main()
