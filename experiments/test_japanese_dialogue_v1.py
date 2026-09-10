import unittest

import japanese_dialogue_v1 as jd


def _b(genus, understood=True, conf=0.7):
    return {"genus": genus, "understood": understood, "confidence": conf}


WM = {
    "beliefs": {
        "椅子": _b("道具"), "鉛筆": _b("道具"), "病院": _b("場所"),
        "きつね": _b("生き物"), "からす": _b("生き物"),
        "ふとわたし": _b("生き物"),          # parser debris -- must be filtered
    },
    "contexts": {"椅子": {"鉛筆": 4, "つくえ": 3}, "きつね": {"からす": 5, "ぶどう": 2}},
    "entities": {"きつね": 40, "椅子": 12, "鉛筆": 8, "病院": 5, "からす": 30},
    "profiles": {},
}
RULES = [{"status": "reusable", "subject_genus": "生き物", "verb": "はしる", "object_genus": ""}]


class FakePartner:
    def __init__(self, reply="はい、それは面白いですね。", up=True):
        self.reply, self.up = reply, up

    def available(self):
        return self.up

    def respond(self, utt):
        return self.reply


class ComposeTest(unittest.TestCase):
    def test_concrete_concepts_come_first_and_debris_is_dropped(self):
        cs = jd._understood_concepts(WM)
        self.assertNotIn("ふとわたし", cs)
        self.assertLess(cs.index("椅子"), cs.index("きつね"))   # concrete before a creature

    def test_compose_makes_a_grounded_sentence_per_strategy(self):
        self.assertEqual(jd.compose("椅子", WM, RULES, "genus"), "「椅子」は道具です。")
        self.assertEqual(jd.compose("椅子", WM, RULES, "question"), "「椅子」は道具ですか。")
        self.assertIn("はしる", jd.compose("きつね", WM, RULES, "property"))     # from the rule
        self.assertIn("鉛筆", jd.compose("椅子", WM, RULES, "relation"))          # a real neighbour

    def test_compose_empty_for_an_unknown_concept(self):
        self.assertEqual(jd.compose("みらい", WM, RULES, "genus"), "")


def _claim(concept, strategy="genus"):
    return jd._claim(concept, WM, RULES, strategy)


class ScoreTest(unittest.TestCase):
    def test_independent_correct_use_is_an_understood_candidate(self):
        s = jd.score(_claim("椅子"), "椅子は人が座るために使います。木や金属で作られています。", WM)
        self.assertTrue(s["understood"])
        self.assertTrue(s["relevant_new_information"])
        self.assertFalse(s["echo_detected"])

    def test_full_verbatim_echo_fails(self):
        s = jd.score(_claim("椅子"), "「椅子」は道具です。", WM)
        self.assertTrue(s["echo_detected"])
        self.assertFalse(s["understood"])

    def test_leaked_prompt_prefix_echo_fails(self):
        s = jd.score(_claim("椅子", "relation"),
                     "学習者: 「椅子」は「鉛筆」と いっしょに 出てきます。", WM)
        self.assertTrue(s["echo_detected"])
        self.assertFalse(s["understood"])

    def test_partial_echo_with_a_tiny_change_fails(self):
        s = jd.score(_claim("椅子", "relation"), "椅子は鉛筆と一緒に出てきます。", WM)
        self.assertFalse(s["understood"])
        self.assertTrue(s["echo_detected"] or not s["relevant_new_information"])

    def test_bare_agreement_fails(self):
        for r in ("そうですね。", "はい、わかりました。", "なるほど、その通りです。"):
            s = jd.score(_claim("椅子"), r, WM)
            self.assertFalse(s["understood"], r)

    def test_concept_word_in_an_unrelated_sentence_fails(self):
        s = jd.score(_claim("椅子"), "椅子について、今日は良い天気で散歩が楽しいです。", WM)
        self.assertFalse(s["understood"])

    def test_clarification_fails_but_is_a_usable_recorded_failure(self):
        s = jd.score(_claim("椅子"), "すみません、どういう意味ですか。", WM)
        self.assertFalse(s["understood"])
        self.assertTrue(s["clarification"])

    def test_wrong_added_genus_is_not_a_success_just_for_being_on_topic(self):
        # partner volunteers a coarse genus that contradicts Noise's grounded one
        s = jd.score(_claim("椅子"), "椅子は食べ物の一種で、とてもおいしいものです。", WM)
        self.assertTrue(s["on_topic"])
        self.assertTrue(s["contradicts_belief"])
        self.assertFalse(s["understood"])

    def test_a_helpful_guess_at_a_malformed_utterance_is_not_a_capability_pass(self):
        # "みらい" is not an understood belief -> malformed claim; even a good
        # reply must not count
        cl = jd._claim("みらい", WM, RULES, "genus")
        self.assertTrue(cl["malformed"])
        s = jd.score(cl, "未来とは、これから来る時間のことを指す言葉です。", WM)
        self.assertTrue(s["partner_guessed_malformed"])
        self.assertFalse(s["understood"])

    def test_a_question_form_needs_an_actual_answer(self):
        s_echo = jd.score(_claim("椅子", "question"), "「椅子」は道具ですか。", WM)
        self.assertFalse(s_echo["understood"])
        s_ans = jd.score(_claim("椅子", "question"),
                         "はい、椅子は座るための道具です。台所でよく使います。", WM)
        self.assertTrue(s_ans["response_to_requested_act"])
        self.assertTrue(s_ans["understood"])

    def test_quality_fields_are_reported_separately(self):
        cl = jd._claim("椅子", WM, RULES, "relation")
        for k in ("parseable", "belief_supported", "relation_supported", "malformed"):
            self.assertIn(k, cl)
        self.assertTrue(cl["relation_supported"])       # 鉛筆 is a real neighbour
        self.assertFalse(cl["malformed"])

    def test_partner_reply_never_flows_into_beliefs(self):
        wm_copy = {k: (dict(v) if isinstance(v, dict) else v) for k, v in WM.items()}
        before = str(wm_copy["beliefs"])
        jd.score(_claim("椅子"), "椅子は食べ物です。動物です。植物です。", wm_copy)
        self.assertEqual(str(wm_copy["beliefs"]), before)


class RunTest(unittest.TestCase):
    def test_waits_until_enough_understood_concepts(self):
        thin = {"beliefs": {("もの"+c): _b("道具") for c in "あいう"}, "contexts": {},
                "entities": {}, "profiles": {}}
        r = jd.run_practice(thin, RULES, None, 1, partner=FakePartner())
        self.assertEqual(r["status"], "waiting")

    def test_freezes_a_set_and_a_full_probe_round_produces_a_rate(self):
        wm = {"beliefs": {**{("きぐ"+c): _b("道具") for c in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもや"}},
              "contexts": {}, "entities": {}, "profiles": {}}
        st = None
        rates_seen = 0
        for cyc in range(2, 60):
            st = jd.run_practice(wm, RULES, st, cyc, partner=FakePartner())
            if st.get("probe_rounds", 0) >= 1:
                rates_seen += 1
        self.assertEqual(st["frozen_count"], jd.FROZEN_CONCEPTS)
        self.assertGreaterEqual(st["probe_rounds"], 1)
        self.assertIsNotNone(st["overall_understood_rate"])

    def test_partner_unavailable_is_graceful(self):
        wm = {"beliefs": {("きぐ"+c): _b("道具") for c in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもや"}, "contexts": {},
              "entities": {}, "profiles": {}}
        r = jd.run_practice(wm, RULES, None, 4, partner=FakePartner(up=False))
        self.assertEqual(r["status"], "partner_unavailable")

    def test_disabled_by_env(self):
        import os
        os.environ["AI_NOISE_JA_DIALOGUE"] = "0"
        try:
            self.assertFalse(jd.enabled())
        finally:
            del os.environ["AI_NOISE_JA_DIALOGUE"]
        self.assertTrue(jd.enabled())


if __name__ == "__main__":
    unittest.main()
