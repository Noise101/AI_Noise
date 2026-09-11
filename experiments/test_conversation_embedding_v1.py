import unittest

import conversation_embedding_v1 as assoc


def _claims(**subjects):
    """subjects: {subject: [predicate, ...]}"""
    return {s: [{"predicate": p, "retracted": False} for p in preds]
            for s, preds in subjects.items()}


class LearnTest(unittest.TestCase):
    def test_shared_predicate_content_pulls_two_subjects_together(self):
        claims = _claims(献="道具です", 鍵盤="道具です")
        state = None
        for _ in range(40):
            state = assoc.learn(state, claims, [])
        self.assertIn("献", state["word_vectors"])
        self.assertIn("鍵盤", state["word_vectors"])

    def test_topic_stack_proximity_creates_an_association_without_claims(self):
        stack = ["犬", "猫", "鳥"]
        state = None
        for _ in range(40):
            state = assoc.learn(state, {}, stack)
        self.assertIn("犬", state["word_vectors"])
        self.assertIn("猫", state["word_vectors"])

    def test_replaying_the_same_turn_id_does_not_inflate_counts(self):
        claims = _claims(献="道具です")
        state = assoc.learn(None, claims, [], turn_id="t1")
        pairs = state["pairs_seen"]
        again = assoc.learn(state, claims, [], turn_id="t1")
        self.assertEqual(again["pairs_seen"], pairs)

    def test_version_change_resets(self):
        state = assoc.learn(None, _claims(献="道具"), [])
        state["version"] = 0
        reset = assoc.learn(state, {}, [])
        self.assertEqual(reset["version"], assoc.VERSION)
        self.assertEqual(reset["word_vectors"], {})

    def test_no_content_words_produces_no_pairs(self):
        # は/です alone have no kanji/katakana content -- nothing to learn
        state = assoc.learn(None, _claims(あ="いう"), [])
        self.assertEqual(state["pairs_seen"], 0)


class RecallTest(unittest.TestCase):
    def _trained(self):
        state = None
        claims = _claims(献="道具です", 鍵盤="道具です", 電灯="道具です")
        for _ in range(60):
            state = assoc.learn(state, claims, [])
        return state

    def test_recall_finds_a_related_previously_discussed_topic(self):
        state = self._trained()
        results = assoc.recall("献", state, known_topics={"鍵盤", "電灯"})
        self.assertTrue(results)
        self.assertIn(results[0]["topic"], {"鍵盤", "電灯"})

    def test_recall_never_returns_the_query_itself(self):
        state = self._trained()
        results = assoc.recall("献", state, known_topics={"献", "鍵盤", "電灯"})
        self.assertNotIn("献", [r["topic"] for r in results])

    def test_recall_is_restricted_to_known_topics(self):
        state = self._trained()
        results = assoc.recall("献", state, known_topics={"鍵盤"})
        self.assertTrue(all(r["topic"] == "鍵盤" for r in results))

    def test_recall_on_an_unseen_topic_is_empty(self):
        state = self._trained()
        self.assertEqual(assoc.recall("未知語", state, known_topics={"鍵盤"}), [])

    def test_recall_with_no_state_is_empty(self):
        self.assertEqual(assoc.recall("献", None), [])

    def test_similarity_is_a_bounded_score_not_a_claim(self):
        state = self._trained()
        results = assoc.recall("献", state, known_topics={"鍵盤", "電灯"})
        for r in results:
            self.assertGreaterEqual(r["similarity"], assoc.MIN_SIMILARITY)
            self.assertLessEqual(r["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
