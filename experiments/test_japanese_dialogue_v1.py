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


class ScoreTest(unittest.TestCase):
    def test_on_topic_coherent_reply_is_understood(self):
        s = jd.score("「椅子」は道具です。", "椅子は座るための道具ですね。", "椅子", WM)
        self.assertTrue(s["understood"])

    def test_a_clarification_request_is_not_understood(self):
        s = jd.score("「椅子」は道具です。", "すみません、どういう意味ですか。", "椅子", WM)
        self.assertFalse(s["understood"])
        self.assertTrue(s["clarification"])

    def test_an_off_topic_reply_is_not_understood(self):
        s = jd.score("「椅子」は道具です。", "今日はいい天気ですね。", "椅子", WM)
        self.assertFalse(s["understood"])


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
