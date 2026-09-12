import os
import unittest

# every call to chat.converse() below that omits `phraser` would otherwise
# default to a real PhrasingModel(); on a machine that actually has Ollama
# running with the model pulled (as this one does), PhrasingModel.available()
# returns True and every such test becomes a real, unbounded network call to
# a 27B model (PARTNER_TIMEOUT-less by design -- see japanese_dialogue_v1) --
# unit tests must not depend on an external service being up.  Tests that
# specifically exercise the phrasing layer use an explicit `_FakePhraser` and
# are unaffected by this flag.
os.environ.setdefault("AI_NOISE_SKIP_LOCAL_LLM", "1")
# same reasoning for _ask_about's default web_lookup=_web_gist -- a real
# Wiktionary/Wikipedia fetch on every "unknown topic" test.  Tests that
# specifically exercise the web-gist path inject a fake lookup instead.
os.environ.setdefault("AI_NOISE_SKIP_WEB_LOOKUP", "1")

import noise_chat_v1 as chat


class _MorphologyIsolatedTestCase(unittest.TestCase):
    """Base for any test whose expectations depend on a real morphological
    analyser being available.  `AI_NOISE_NO_MORPHOLOGY` is a process-wide env
    var another test module in the same full-suite run may set
    (test_japanese_reader_v1 forces it off for its own determinism) --
    checking availability once at class/import time is not enough, since that
    pollution can land between import and execution.  setUp forces a real
    backend for these tests and undoes the pollution afterwards."""

    def setUp(self):
        import morphology_teacher as mt
        self._prev_env = os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
        mt.get_teacher(refresh=True)
        if mt.get_teacher().name == "none":
            self._restore_env()
            self.skipTest("no morphological analyser vendored/installed")

    def tearDown(self):
        self._restore_env()

    def _restore_env(self):
        import morphology_teacher as mt
        if self._prev_env is None:
            os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
        else:
            os.environ["AI_NOISE_NO_MORPHOLOGY"] = self._prev_env
        mt.get_teacher(refresh=True)


class NoiseChatTests(_MorphologyIsolatedTestCase):
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

    def test_the_common_wa_spelling_of_greetings_is_recognised(self):
        # こんにちわ/こんばんわ (わ for は) is an extremely common casual
        # spelling in real Japanese text -- a real user hit this failure live
        for greeting in ("こんばんわ", "こんにちわ"):
            _, state = chat.converse(greeting, None)
            self.assertEqual(state["last_turn"]["interpretation"]["intent"], "greeting")

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

    # -- structural question-form detection (no written ？ required) --------
    def test_desu_ka_question_without_a_question_mark_is_recognised(self):
        reply, state = chat.converse("元気ですか", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_wellbeing")

    def test_bare_ka_ending_is_a_question_via_morphology(self):
        self.assertTrue(chat._is_question_form("これは犬か"))

    def test_ka_ending_content_words_are_not_misread_as_questions(self):
        self.assertFalse(chat._is_question_form("これは静かだ"))
        self.assertFalse(chat._is_question_form("何かある"))

    def test_kana_colloquial_ending_is_a_question(self):
        self.assertTrue(chat._is_question_form("これは何かな"))

    def test_self_identity_question_is_not_read_as_a_vocabulary_lookup(self):
        reply, state = chat.converse("あなたの名前は", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_identity")
        self.assertIn("Noise", reply)

    def test_thanks_and_capability_questions_get_a_direct_reply(self):
        reply, _ = chat.converse("ありがとう", None)
        self.assertIn("どういたしまして", reply)
        reply, _ = chat.converse("何ができるの", None)
        self.assertEqual(len(reply), len(reply.strip()))
        self.assertNotIn("文の役割", reply)

    # -- a remark is not testimony, even mid-context ------------------------
    def test_a_sentence_final_remark_particle_is_not_stored_as_a_claim(self):
        reply, state = chat.converse("今日はいい天気ですね", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "unresolved")
        self.assertEqual(state.get("claims", {}), {})

    def test_a_remark_does_not_get_misfiled_as_the_answer_to_a_pending_question(self):
        _, state = chat.converse("犬とは何ですか", None)
        self.assertEqual(state.get("awaiting", {}).get("topic"), "犬")
        reply, state = chat.converse("今日はいい天気ですね", state)
        self.assertNotIn("犬", state.get("claims", {}))

    def test_a_question_shaped_predicate_is_never_stored_as_testimony(self):
        _, state = chat.converse("犬とは何ですか", None)
        self.assertNotIn("犬", state.get("claims", {}))

    def test_unresolved_fallback_asks_the_user_to_rephrase(self):
        # topic extraction falls back to a regex match when no morphological
        # analyser is available (AI_NOISE_NO_MORPHOLOGY, possibly set process-
        # wide by another test module) -- assert on what both fallback
        # branches share, not the topic-bearing phrasing specifically
        reply, state = chat.converse("静かですね", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "unresolved")
        self.assertIn("もらえますか", reply)
        self.assertNotIn("文の役割を決められませんでした", reply)

    # -- multi-topic conversational memory (topic_stack) --------------------
    def test_follow_up_surfaces_a_new_fact_about_the_current_topic(self):
        _, state = chat.converse("猫は動物です", None)
        _, state = chat.converse("猫は可愛いです", state)
        reply, state = chat.converse("それについてもっと教えて", state)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "follow_up")
        self.assertIn("可愛い", reply)
        # a second follow-up does not repeat the same fact
        reply2, state = chat.converse("他には？", state)
        self.assertNotIn("可愛い", reply2)

    def test_named_go_back_resumes_an_earlier_topic(self):
        _, state = chat.converse("犬とは何ですか", None)
        _, state = chat.converse("猫は動物です", state)
        self.assertEqual(state["current_topic"], "猫")
        reply, state = chat.converse("犬の話に戻って", state)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "go_back_topic")
        self.assertEqual(state["current_topic"], "犬")
        self.assertIn("犬", reply)

    def test_generic_go_back_targets_the_previous_topic_not_a_literal_saki(self):
        _, state = chat.converse("犬とは何ですか", None)
        _, state = chat.converse("猫は動物です", state)
        reply, state = chat.converse("さっきの話に戻って", state)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "go_back_topic")
        self.assertEqual(state["current_topic"], "犬")
        self.assertNotIn("さっき", reply)

    def test_go_back_to_a_never_discussed_topic_says_so(self):
        _, state = chat.converse("犬とは何ですか", None)
        reply, state = chat.converse("象の話に戻って", state)
        self.assertIn("まだ話していません", reply)
        self.assertEqual(state["current_topic"], "犬")     # unchanged

    def test_topic_stack_moves_touched_topics_to_the_front(self):
        _, state = chat.converse("犬とは何ですか", None)
        _, state = chat.converse("猫は動物です", state)
        self.assertEqual(state["topic_stack"], ["犬", "猫"])
        _, state = chat.converse("犬の話に戻って", state)
        self.assertEqual(state["topic_stack"], ["猫", "犬"])

    def test_follow_up_with_no_topic_asks_what_to_expand_on(self):
        reply, state = chat.converse("もっと教えて", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "follow_up")
        self.assertIn("話題を教えてください", reply)


class _FakePhraser:
    def __init__(self, reply=None, avail=True):
        self._reply, self._avail = reply, avail

    def available(self):
        return self._avail

    def rephrase(self, template):
        return self._reply


class StructuralSmallTalkTests(_MorphologyIsolatedTestCase):
    """Small-talk intents keyed by morphological LEMMA + structural shape, not
    a literal surface string -- matching one exact spelling/conjugation/
    politeness level is exactly why "こんばんわ" and "何が出来るの" kept
    failing live; every new ending was its own future patch."""

    def test_greeting_survives_politeness_and_the_common_wa_misspelling(self):
        for text in ("こんばんわ", "こんにちわ", "おはようございます"):
            _, state = chat.converse(text, None)
            self.assertEqual(state["last_turn"]["interpretation"]["intent"], "greeting")

    def test_thanks_survives_politeness_level_and_a_leading_intensifier(self):
        for text in ("ありがとうございました", "ありがとうね", "どうもありがとうございます",
                     "本当にありがとう"):
            _, state = chat.converse(text, None)
            self.assertEqual(state["last_turn"]["interpretation"]["intent"], "thanks")

    def test_wellbeing_question_survives_casual_and_polite_forms(self):
        for text in ("元気かな", "お元気ですか", "調子はどうですか"):
            _, state = chat.converse(text, None)
            self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_wellbeing")

    def test_capability_question_survives_a_bare_no_ending(self):
        # の as a casual 終助詞 question marker, not the ？-anchored regex this
        # used to require
        for text in ("何が出来ますか", "何ができるの"):
            _, state = chat.converse(text, None)
            self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_capability")

    def test_identity_question_covers_pronoun_and_noise_by_name(self):
        for text in ("あなたのお名前は", "君は誰なの", "Noiseとは何ですか"):
            _, state = chat.converse(text, None)
            self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_identity")

    def test_attributive_no_is_not_misread_as_a_question(self):
        # 連体化 の (possessive/nominalising) is not 終助詞 の -- only the
        # analyser tells them apart; this must not become ask_capability or
        # any other question-routed intent
        self.assertFalse(chat._is_question_form("これは私のものの"))

    def test_structural_checks_fall_back_to_the_literal_regex_without_an_analyser(self):
        import os
        prev = os.environ.get("AI_NOISE_NO_MORPHOLOGY")
        os.environ["AI_NOISE_NO_MORPHOLOGY"] = "1"
        try:
            import morphology_teacher as mt
            mt.get_teacher(refresh=True)
            self.assertTrue(chat._is_greeting("こんにちは"))
            self.assertTrue(chat._is_thanks("ありがとうございます"))
            self.assertTrue(chat._is_wellbeing_question("お元気ですか"))
            self.assertTrue(chat._is_capability_question("何ができるの"))
            self.assertTrue(chat._is_identity_question("あなたの名前は"))
        finally:
            if prev is None:
                os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
            else:
                os.environ["AI_NOISE_NO_MORPHOLOGY"] = prev
            import morphology_teacher as mt
            mt.get_teacher(refresh=True)


class PhrasingLayerTests(unittest.TestCase):
    """PhrasingModel is a presentation layer only -- content-verified, never
    a source of new information (ARCHITECTURE.md "Optional local-model
    boundary")."""

    def test_a_verified_rephrase_that_keeps_all_content_words_is_used(self):
        template = "犬はまだよく分かりません。どんなものですか。"
        good = "犬については、まだよく分からないので、どんなものか教えてください。"
        self.assertEqual(chat._phrase_naturally(template, _FakePhraser(good)), good)

    def test_a_rephrase_that_drops_a_content_word_is_rejected(self):
        template = "犬はまだよく分かりません。どんなものですか。"
        dropped = "まだよく分かりません。"           # 犬 is gone
        self.assertEqual(chat._phrase_naturally(template, _FakePhraser(dropped)), template)

    def test_an_unavailable_model_leaves_the_template_unchanged(self):
        template = "犬はまだよく分かりません。どんなものですか。"
        self.assertEqual(
            chat._phrase_naturally(template, _FakePhraser("何か", avail=False)), template)

    def test_empty_or_none_rephrase_leaves_the_template_unchanged(self):
        template = "犬はまだよく分かりません。どんなものですか。"
        self.assertEqual(chat._phrase_naturally(template, _FakePhraser(None)), template)
        self.assertEqual(chat._phrase_naturally(template, _FakePhraser("")), template)

    def test_a_wildly_longer_rephrase_is_rejected_even_if_content_survives(self):
        template = "犬はまだよく分かりません。どんなものですか。"
        bloated = ("犬" * 200)
        self.assertEqual(chat._phrase_naturally(template, _FakePhraser(bloated)), template)

    def test_converse_uses_the_injected_phraser_and_keeps_the_template_for_audit(self):
        good = "猫の話、まだよく分かってないので、どんな子か教えてね。"
        reply, state = chat.converse("猫とは何ですか", None, None, _FakePhraser(good))
        self.assertEqual(reply, good)
        self.assertIn("template_reply", state["last_turn"])
        self.assertNotEqual(state["last_turn"]["template_reply"], good)

    def test_phrasing_disabled_by_env_flag(self):
        import os
        os.environ["AI_NOISE_CHAT_PHRASING"] = "0"
        try:
            template = "犬はまだよく分かりません。どんなものですか。"
            self.assertEqual(
                chat._phrase_naturally(template, _FakePhraser("犬は謎ですね。")), template)
        finally:
            os.environ.pop("AI_NOISE_CHAT_PHRASING", None)


class AssociativeRecallMemoryTests(unittest.TestCase):
    """conversation_embedding_v1: a bounded, score-0 recall PROPOSAL, never a
    claim -- see conversation_embedding_v1.py and _recall_prompt."""

    def _boost(self, state, rounds=200):
        import conversation_embedding_v1 as assoc
        for i in range(rounds):
            state["embedding_memory"] = assoc.learn(
                state.get("embedding_memory"), state.get("claims", {}),
                state.get("topic_stack", []), turn_id=f"boost{i}")
        return state

    def test_asking_about_a_related_earlier_topic_again_offers_a_recall_prompt(self):
        _, state = chat.converse("献とは何ですか", None)
        _, state = chat.converse("鍵盤は道具です", state)
        state = self._boost(state)
        reply, state = chat.converse("献とは何ですか", state)
        self.assertIn("鍵盤", reply)
        self.assertIn("関係ありますか", reply)

    def test_a_first_ever_mention_offers_no_recall_prompt(self):
        # nothing has been discussed yet -- there is no association to recall,
        # and none should be fabricated
        reply, state = chat.converse("献とは何ですか", None)
        self.assertNotIn("以前", reply)

    def test_the_same_recall_is_not_repeated_for_the_same_subject(self):
        _, state = chat.converse("献とは何ですか", None)
        _, state = chat.converse("鍵盤は道具です", state)
        state = self._boost(state)
        first, state = chat.converse("献とは何ですか", state)
        second, state = chat.converse("献とは何ですか", state)
        self.assertIn("鍵盤", first)
        self.assertNotIn("鍵盤", second)

    def test_recall_never_creates_or_changes_a_claim(self):
        _, state = chat.converse("献とは何ですか", None)
        _, state = chat.converse("鍵盤は道具です", state)
        state = self._boost(state)
        _, state = chat.converse("献とは何ですか", state)
        self.assertNotIn("献", state.get("claims", {}))   # still nothing told about 献


class LiveFailureRegressionTests(_MorphologyIsolatedTestCase):
    """Real bugs a live conversation surfaced in one sitting -- each is its
    own regression test so none of them comes back silently."""

    def test_greeting_echoes_the_users_own_time_of_day_not_the_wall_clock(self):
        # replying こんにちは to a こんばんわ greeting because the SERVER's
        # clock disagrees with the user reads as ignoring what they said
        reply, _ = chat.converse("こんばんわ", None)
        self.assertIn("こんばんは", reply)
        reply, _ = chat.converse("おはよう", None)
        self.assertIn("おはよう", reply)

    def test_a_single_kanji_topic_is_not_shadowed_by_an_unrelated_noun(self):
        # "猫の話のほうがいいな" must extract 猫, not the direction noun 方/ほう
        self.assertEqual(chat._topic("猫の話のほうがいいな"), "猫")

    def test_a_te_form_verb_is_not_mistaken_for_the_tte_topic_particle(self):
        # "知って" contains the two characters って in the middle of 知って(いる)
        # -- that is not the quotative って topic marker, and reading it as
        # one produces a garbage topic like "何を知"
        self.assertEqual(chat._topic("何を知っている？"), "")

    def test_recall_knowledge_answers_about_the_asked_subject_only(self):
        _, state = chat.converse("猫は動物です", None)
        _, state = chat.converse("犬は生き物です", state)
        reply, state = chat.converse("猫について何を知っている？", state)
        self.assertIn("猫", reply)
        self.assertNotIn("犬", reply)

    def test_recall_knowledge_with_no_claims_about_the_topic_says_so(self):
        _, state = chat.converse("犬は生き物です", None)
        reply, state = chat.converse("猫について何を知っている？", state)
        self.assertIn("まだありません", reply)

    def test_a_bare_pronoun_is_never_stored_as_a_claim_subject(self):
        for text in ("それは覚えることではない。", "これは違う。", "あれは変だ。"):
            _, state = chat.converse(text, None)
            for pronoun in ("それ", "これ", "あれ"):
                self.assertNotIn(pronoun, state.get("claims", {}))

    def test_first_person_pronoun_is_never_stored_as_a_claim_subject(self):
        # whose "私" it is (the user's, or Noise's) is not resolvable from the
        # string alone -- this is conversational deixis, not a stable subject
        _, state = chat.converse("私は、犬より、猫の話の方が良いです", None)
        self.assertNotIn("私", state.get("claims", {}))

    def test_recall_preference_recognises_the_compound_noun_suki_kirai(self):
        # 好き嫌い ("likes and dislikes") is ONE compound-noun lemma, not the
        # separate 好き/嫌い it decomposes to elsewhere -- checking only the
        # bare words missed this real phrasing
        _, state = chat.converse("私の好き嫌い覚えてる？", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "recall_preference")

    def test_noise_preference_question_is_not_shadowed_by_identity(self):
        # shares あなた+何 with a plain identity question ("Noiseとは何ですか")
        # -- 好き/嫌い present means this asks about a PREFERENCE
        _, state = chat.converse("あなたの好きなものは何？", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_noise_preference")
        _, state = chat.converse("Noiseとは何ですか", None)
        self.assertEqual(state["last_turn"]["interpretation"]["intent"], "ask_identity")


class WebReferenceLookupTests(unittest.TestCase):
    """A read-only reference lookup for an unknown topic -- reuses
    japanese_word_meaning_v1's cached Wiktionary/Wikipedia fetch, stored as
    source: "web_reference" testimony: capped, revisable, never Noise's own
    confirmed understanding and never written into the word-meaning belief
    store (ARCHITECTURE.md "Optional local-model boundary" -- same boundary,
    a different kind of external source)."""

    def test_a_confident_gist_is_offered_and_stored_as_web_reference_testimony(self):
        state = chat._blank()
        reply = chat._ask_about(state, "献", web_lookup=lambda t: {
            "text": f"{t}は道具の一種だと辞書にあります。", "confident": True})
        self.assertIn("道具", reply)
        self.assertEqual(state["claims"]["献"][0]["source"], "web_reference")
        self.assertEqual(state["claims"]["献"][0]["evidence_role"],
                         "conversation_memory_not_world_fact")
        self.assertNotIn("献", state.get("encountered", {}))

    def test_a_missing_gist_falls_back_to_the_plain_question(self):
        state = chat._blank()
        reply = chat._ask_about(state, "献", web_lookup=lambda t: None)
        self.assertNotIn("献", state.get("claims", {}))
        self.assertNotIn("献", state.get("encountered", {}))
        self.assertIn("まだよく分かりません", reply)

    def test_an_unresolved_fetch_is_kept_as_an_encounter_not_discarded(self):
        # a page WAS fetched (something was read) but nothing resolved into a
        # clean genus -- per the owner, that is not the same as never having
        # looked, and must not be thrown away just because it did not parse
        # into a tidy structured claim
        state = chat._blank()
        reply = chat._ask_about(state, "献", web_lookup=lambda t: {
            "text": f"辞書には「{t}は古い言葉である」とありますが、うまく分類できませんでした。",
            "confident": False})
        self.assertIn("古い言葉", reply)
        self.assertNotIn("献", state.get("claims", {}))       # not a confirmed claim
        self.assertEqual(len(state["encountered"]["献"]), 1)
        entry = state["encountered"]["献"][0]
        self.assertEqual(entry["confidence"], "uncertain")
        self.assertEqual(entry["source"], "web_reference")

    def test_encounters_are_capped_and_never_promoted_to_a_claim(self):
        state = chat._blank()
        for i in range(chat.ENCOUNTERS_PER_SUBJECT + 3):
            chat._remember_encounter(state, "献", f"見た内容{i}", "web_reference", f"t{i}")
        self.assertEqual(len(state["encountered"]["献"]), chat.ENCOUNTERS_PER_SUBJECT)
        self.assertNotIn("献", state.get("claims", {}))

    def test_the_lookup_is_attempted_only_once_per_subject(self):
        calls = {"n": 0}

        def counting_lookup(topic):
            calls["n"] += 1
            return None

        state = chat._blank()
        chat._ask_about(state, "献", web_lookup=counting_lookup)
        chat._ask_about(state, "献", web_lookup=counting_lookup)
        chat._ask_about(state, "献", web_lookup=counting_lookup)
        self.assertEqual(calls["n"], 1)

    def test_web_reference_never_overrides_a_live_human_claim(self):
        state = chat._blank()
        chat._remember_claim(state, "献", "楽器の一種", "t1")
        chat._ask_about(state, "献", web_lookup=lambda t: {"text": f"{t}は道具です。",
                                                          "confident": True})
        # only the human's claim is present -- the lookup does not run once a
        # live claim already exists for this subject
        self.assertEqual(len(state["claims"]["献"]), 1)
        self.assertEqual(state["claims"]["献"][0]["source"], "owner_testimony")

    def test_web_lookup_disabled_by_env_flag(self):
        import os
        os.environ.pop("AI_NOISE_SKIP_WEB_LOOKUP", None)
        try:
            os.environ["AI_NOISE_SKIP_WEB_LOOKUP"] = "1"
            self.assertIsNone(chat._web_gist("献"))
        finally:
            os.environ["AI_NOISE_SKIP_WEB_LOOKUP"] = "1"   # restore test default


if __name__ == "__main__":
    unittest.main()
