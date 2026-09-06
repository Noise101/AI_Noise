import unittest

from generative_dialogue_v1 import (compose_utterance, run, score_comprehension,
                                    _speakable, _verbalise)


class ScriptedPartner:
    def __init__(self, replies, available=True):
        self.replies = list(replies)
        self._available = available
        self.model = "scripted"
        self.heard = []

    def available(self):
        return self._available

    def respond(self, utterance):
        self.heard.append(utterance)
        return self.replies.pop(0) if self.replies else None


EVENTS = ["fox|saw|grapes", "fox|wanted|grapes", "wolf|chased|sheep",
          "having|become|augustinian", "king|gave|abbot", "rat|ran|off"]


class GenerativeDialogueTest(unittest.TestCase):
    def test_verbalise_drops_the_article_before_a_particle(self):
        self.assertEqual(_verbalise("fox|saw|grapes"), "the fox saw the grapes")
        self.assertEqual(_verbalise("rat|ran|off"), "the rat ran off")
        self.assertEqual(_verbalise("fox|slept|"), "the fox slept")

    def test_speakable_requires_an_animate_subject_and_a_known_verb(self):
        self.assertTrue(_speakable("fox|saw|grapes"))
        self.assertFalse(_speakable("having|become|augustinian"))
        self.assertFalse(_speakable("wisdom|is|rare"))

    def test_compose_prefers_a_speakable_event_over_a_junk_one(self):
        text = compose_utterance("event", EVENTS, None, None, offset=0)
        self.assertNotIn("having", text)
        self.assertIn("fox", text)

    def test_event_pair_joins_two_events_with_the_same_subject(self):
        text = compose_utterance("event_pair", EVENTS, None, None, offset=0)
        self.assertEqual(text, "the fox saw the grapes. the fox wanted the grapes")

    def test_rnn_strategies_fall_back_to_event_without_a_sampler(self):
        report = run(EVENTS, ScriptedPartner(["ok"]), {"rotation": 2})  # STRATEGIES[2] == rnn_extend
        self.assertEqual(report["turns"][-1]["strategy"], "event")

    def test_comprehension_needs_shared_content_and_no_clarification_request(self):
        good = score_comprehension("the fox wanted the grapes",
                                   "The fox really wanted those grapes badly.")
        self.assertTrue(good["understood"])
        clarify = score_comprehension("the fox wanted the grapes",
                                      "I'm not sure what you mean by that.")
        self.assertFalse(clarify["understood"])
        self.assertTrue(clarify["clarification_request"])
        unrelated = score_comprehension("the fox wanted the grapes",
                                        "Nice weather we are having lately.")
        self.assertFalse(unrelated["understood"])

    def test_unavailable_partner_reports_cleanly(self):
        report = run(EVENTS, ScriptedPartner([], available=False), {})
        self.assertEqual(report["status"], "partner_unavailable")
        self.assertEqual(report["turns"], [])

    def test_run_learns_a_per_strategy_comprehension_rate(self):
        partner = ScriptedPartner(["The fox saw the grapes on the vine."])
        first = run(EVENTS, partner, {})
        strat = first["turns"][-1]["strategy"]
        self.assertIn(strat, first["strategy_performance"])
        self.assertIn("comprehension_rate", first["strategy_performance"][strat])
        second = run(EVENTS, ScriptedPartner(["I don't understand."]), first)
        self.assertGreater(
            sum(b["turns"] for b in second["strategy_performance"].values()),
            sum(b["turns"] for b in first["strategy_performance"].values()))

    def test_reply_is_never_treated_as_a_fact(self):
        report = run(EVENTS, ScriptedPartner(["The fox saw grapes."]), {})
        self.assertIn("evidence score zero for facts", report["note"])


if __name__ == "__main__":
    unittest.main()
