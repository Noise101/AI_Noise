import unittest

import noise_chat_v1 as chat


class NoiseChatTests(unittest.TestCase):
    def test_claim_is_remembered_as_testimony_not_world_fact(self):
        reply, state = chat.converse("レモンは果物だよ", None)
        self.assertIn("まだ私自身では確かめていません", reply)
        claim = state["claims"]["レモン"][0]
        self.assertEqual(claim["source"], "owner_testimony")
        self.assertEqual(claim["evidence_role"], "conversation_memory_not_world_fact")
        recalled, _ = chat.converse("レモンって何？", state)
        self.assertIn("果物", recalled)
        self.assertIn("会話の記憶", recalled)

    def test_unknown_is_admitted_and_asked_about(self):
        reply, state = chat.converse("クオリアって何？", None)
        self.assertIn("まだよく分かりません", reply)
        self.assertEqual(state["unknown_topics"]["クオリア"], 1)

    def test_preference_is_person_bound_and_recalled(self):
        reply, state = chat.converse("私はレモンが好きです", None)
        self.assertIn("覚えておきます", reply)
        recalled, state = chat.converse("私の好きなもの覚えてる？", state)
        self.assertIn("レモンが好き", recalled)
        self.assertEqual(state["preferences"]["レモン"]["source"], "owner_self_report")

    def test_explicit_correction_marks_the_previous_response_wrong(self):
        _, state = chat.converse("レモンは野菜だよ", None)
        reply, state = chat.converse("違う、レモンは果物だよ", state)
        self.assertIn("間違いとして残しました", reply)
        self.assertEqual(state["errors"][0]["reason"], "違う")
        self.assertTrue(state["claims"]["レモン"][0]["retracted"])
        self.assertEqual(state["claims"]["レモン"][-1]["predicate"], "果物")

    def test_known_reading_belief_is_expressed_with_uncertainty(self):
        wm = {"beliefs": {"椅子": {"understood": True, "genus": "道具"}}}
        reply, _ = chat.converse("椅子って何？", None, wm)
        self.assertIn("道具", reply)
        self.assertIn("間違っていたら", reply)

    def test_natural_definition_without_copula_is_learned(self):
        _, state = chat.converse("美味しい食べ物は？", None)
        reply, state = chat.converse(
            "美味とは、人間が食物を食べて感じるもので、とても良く感じるもの。", state)
        self.assertIn("聞きました", reply)
        self.assertEqual(
            state["claims"]["美味"][-1]["predicate"],
            "人間が食物を食べて感じるもので、とても良く感じるもの")
        recalled, _ = chat.converse("美味って何？", state)
        self.assertIn("人間が食物を食べて感じる", recalled)

    def test_bare_definition_is_learned(self):
        reply, state = chat.converse("美味は、人間の感覚", None)
        self.assertIn("聞きました", reply)
        self.assertEqual(state["claims"]["美味"][-1]["predicate"], "人間の感覚")

    def test_unknown_question_changes_after_repetition(self):
        first, state = chat.converse("クオリアって何？", None)
        second, state = chat.converse("クオリアって何？", state)
        third, state = chat.converse("クオリアって何？", state)
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(len(state["question_history"]["クオリア"]), 3)

    def test_can_recall_what_human_taught(self):
        _, state = chat.converse("美味は人間の感覚", None)
        reply, _ = chat.converse("何を知っている？", state)
        self.assertIn("美味は人間の感覚", reply)

    def test_old_parser_miss_is_recovered_from_conversation(self):
        old = chat._blank()
        old["turns"] = [{
            "turn_id": "legacy-1", "user": "美味は、人間の感覚",
            "noise": "美味について、何を覚えればよいですか。", "learned": [],
        }]
        reply, state = chat.converse("何を知っている？", old)
        self.assertIn("美味は人間の感覚", reply)
        self.assertEqual(state["stats"]["claims_heard"], 1)
        _, state = chat.converse("こんにちは", state)
        self.assertEqual(state["stats"]["claims_heard"], 1)

    def test_real_failed_dialogue_now_tracks_topic_and_understanding(self):
        state = None
        _, state = chat.converse("美味は、人間の感覚", state)
        recalled, state = chat.converse("何を覚えた？", state)
        self.assertIn("美味は人間の感覚", recalled)
        understood, state = chat.converse("美味はわかった？", state)
        self.assertIn("覚えたことと理解したことは別", understood)
        self.assertEqual(state["last_turn"]["interpretation"]["topic"], "美味")

    def test_greeting_changes_with_conversation_state(self):
        first, state = chat.converse("こんにちは", None)
        _, state = chat.converse("レモンは果物だよ", state)
        second, state = chat.converse("こんにちは", state)
        self.assertNotEqual(first, second)
        self.assertIn("レモン", second)

    def test_elliptical_question_separates_modifier_and_head(self):
        reply, state = chat.converse("美味しい食べ物は？", None)
        read = state["last_turn"]["interpretation"]
        self.assertEqual(read["intent"], "elliptical_question")
        self.assertEqual(read["topic"], "食べ物")
        self.assertIn("美味しいに当てはまる食べ物", reply)

    def test_parsing_feedback_is_retained_as_error(self):
        _, state = chat.converse("今使っている", None)
        reply, state = chat.converse("切り方がおかしい、今、使って、いる", state)
        self.assertIn("切り分けを誤り", reply)
        self.assertEqual(state["errors"][-1]["kind"], "parsing")

    def test_learning_status_uses_actual_memory(self):
        memory = {"beliefs": {"椅子": {"understood": True, "genus": "道具",
                                      "last_cycle": 4}}}
        reply, state = chat.converse("今日は何を学んだ？", None, memory)
        self.assertIn("椅子", reply)
        self.assertEqual(chat.summary(state)["last_intent"], "learning_status")


if __name__ == "__main__":
    unittest.main()
