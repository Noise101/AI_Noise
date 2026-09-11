import unittest

import cognition_v1 as cog


def _belief(genus, understood=True, conf=0.7):
    return {"genus": genus, "understood": understood, "confidence": conf,
            "support": {}, "sources": [], "revisions": [], "last_cycle": 1}


WM = {
    "beliefs": {
        "きつね": _belief("生き物"), "からす": _belief("生き物"), "うさぎ": _belief("生き物"),
        "つくえ": _belief("道具"), "いす": _belief("道具"),
        "やま": _belief("場所"),
        "ふね": _belief("道具", conf=0.4),          # not confident enough for the pool
    },
    "contexts": {
        "きつね": {"からす": 4, "ぶどう": 3, "つくえ": 1},
        "つくえ": {"いす": 5, "きつね": 1},
    },
    "entities": {"きつね": 20, "つくえ": 8},
}


def _events(rows):
    return [{"subject": s, "verb": v, "obj": o, "provenance": "heuristic_self"} for s, v, o in rows]


STORE = {
    "b1": _events([("きつね", "はしる", ""), ("きつね", "たべる", "ぶどう"),
                   ("からす", "とぶ", ""), ("うさぎ", "たべる", "くさ")]),
    "b2": _events([("きつね", "たべる", "にく"), ("うさぎ", "はねる", ""),
                   ("きつね", "ほえる", ""), ("からす", "たべる", "むし")]),
    "b3": _events([("つくえ", "こわれる", ""), ("だれか", "つくる", "つくえ")]),
}
SHELF = {"b1": {"title": "話1"}, "b2": {"title": "話2"}, "b3": {"title": "話3"}}


class AbstractionTest(unittest.TestCase):
    def test_genus_triples_seen_in_several_books_become_rules(self):
        state = {"version": cog.VERSION, "rules": []}
        cog._abstract_rules(state, STORE, WM, cycle=5)
        forms = {r["verb"]: r for r in state["rules"]}
        self.assertIn("たべる", forms)                       # 生き物 が たべる, in b1 and b2
        self.assertGreaterEqual(forms["たべる"]["support"], 2)
        # vacuous verbs never make a rule
        self.assertNotIn("いる", forms)
        self.assertNotIn("ある", forms)

    def test_a_failed_transfer_adds_a_counterexample_and_can_weaken_a_rule(self):
        state = {"version": cog.VERSION, "rules": [
            {"rule_id": "r1", "form": "「生き物」がとぶ", "subject_genus": "生き物",
             "verb": "とぶ", "object_genus": "", "support": 9, "books": ["x"],
             "confidence": 0.8, "counterexamples": [], "status": "reusable"}]}
        prob = {"pid": "p", "type": "property_transfer", "level": 4, "concept": "きつね",
                "rule_id": "r1", "gold": "はしる", "options": ["とぶ", "はしる"], "grade": "exact",
                "prompt": "?"}
        for _ in range(3):
            cog.record_experience(state, prob, {"answer": "とぶ", "confidence": 0.5, "steps": ["rule r1"]},
                                  correct=False, evaluation={"notes": [], "sufficient_basis": True,
                                                             "contradiction": False}, cycle=7)
        r = state["rules"][0]
        self.assertGreaterEqual(len(r["counterexamples"]), 3)
        self.assertEqual(r["status"], "weakened")


class ReasoningTest(unittest.TestCase):
    def test_semantic_strategy_can_propose_from_learned_usage_neighbours(self):
        wm = {"beliefs": {
            "子犬": _belief("生き物"), "子猫": _belief("生き物"),
            "野猿": _belief("生き物"), "机台": _belief("道具")}}
        sem = {"word_vectors": {
            "こいぬ": [1.0, 0.0], "子犬": [.99, .01], "子猫": [.97, .03],
            "野猿": [.94, .06], "机台": [0.0, 1.0]}}
        prob = {"type": "genus_recall", "concept": "こいぬ",
                "options": list(cog.COARSE), "gold": "生き物", "grade": "coarse", "prompt": "?"}
        sol = cog.solve(prob, wm, [], [], strategy="semantic", semantic_state=sem)
        self.assertEqual(sol["answer"], "生き物")
        self.assertTrue(any("usage-vector" in step for step in sol["steps"]))

    def test_odd_one_out_names_the_minority_class(self):
        prob = {"type": "odd_one_out", "concept": "きつね",
                "options": ["きつね", "からす", "つくえ"], "gold": "つくえ", "grade": "exact",
                "prompt": "?"}
        sol = cog.solve(prob, WM, [], [])
        self.assertEqual(sol["answer"], "つくえ")

    def test_common_property_derives_the_shared_genus(self):
        prob = {"type": "common_property", "concept": "つくえ", "other": "いす",
                "options": ["道具", "生き物", "場所"], "gold": "道具", "grade": "coarse", "prompt": "?"}
        sol = cog.solve(prob, WM, [], [])
        self.assertEqual(sol["answer"], "道具")

    def test_event_recall_reads_the_next_verb_from_stored_events(self):
        prob = {"type": "event_recall", "concept": "きつね", "book_id": "b1",
                "options": ["たべる", "とぶ"], "gold": "たべる", "grade": "exact", "prompt": "?"}
        sol = cog.solve(prob, WM, [], [], heur_store=STORE)
        self.assertEqual(sol["answer"], "たべる")

    def test_unknown_when_nothing_supports_an_answer(self):
        prob = {"type": "genus_recall", "concept": "みずうみ", "options": list(cog.COARSE),
                "gold": "場所", "grade": "coarse", "prompt": "?"}
        sol = cog.solve(prob, WM, [], [])
        self.assertEqual(sol["answer"], "わからない")
        self.assertEqual(sol["confidence"], 0.0)

    def test_compose_mode_ignores_the_targets_own_belief(self):
        wm = {"beliefs": {**WM["beliefs"], "こま": _belief("生き物")},   # wrong belief
              "contexts": {"こま": {"つくえ": 5, "いす": 4}}}
        # compose: infer こま from its道具 neighbours, not its (wrong) belief
        self.assertEqual(cog._infer_genus(wm, "こま", allow_belief=False), "道具")
        self.assertEqual(cog._infer_genus(wm, "こま", allow_belief=True), "生き物")


class HypothesisTest(unittest.TestCase):
    def test_several_hypotheses_are_generated_never_one(self):
        wm = {"beliefs": {"はこ": _belief("生き物", conf=0.6),      # wrong belief
                          "つくえ": _belief("道具"), "いす": _belief("道具")},
              "contexts": {"はこ": {"つくえ": 5, "いす": 4}}, "profiles": {}}
        store = {"b": _events([("だれか", "つくる", "はこ"), ("だれか", "はこぶ", "はこ")])}
        hyps = cog.generate_hypotheses("はこ", wm, store, [])
        genera = {h["genus"] for h in hyps}
        self.assertGreaterEqual(len(genera), 2)                 # not fixed on one answer
        self.assertIn("道具", genera)                            # handled -> a thing

    def test_counterevidence_breaks_a_creature_hypothesis_for_a_never_agent(self):
        wm = {"beliefs": {}, "contexts": {},
              "profiles": {"はこ": {"subj": 0, "obj": 9, "subj_verbs": {}, "obj_verbs": {"つくる": 9}}}}
        ce = cog.seek_counterevidence({"genus": "生き物", "confidence": 0.6}, "はこ", wm, {})
        self.assertTrue(ce["counters"])
        self.assertLess(ce["surviving_confidence"], 0.6)

    def test_deliberation_overrides_a_belief_the_evidence_contradicts(self):
        wm = {"beliefs": {"てつ": _belief("生き物", conf=0.6)},   # wrong: 鉄 is a thing
              "contexts": {},
              "profiles": {"てつ": {"subj": 0, "obj": 7, "subj_verbs": {},
                                    "obj_verbs": {"もつ": 7}}}}
        prob = {"type": "genus_recall", "concept": "てつ", "options": list(cog.COARSE),
                "gold": "道具", "grade": "coarse", "prompt": "「てつ」は？"}
        direct = cog.solve(prob, wm, [], [], {}, deliberate=False)
        delib = cog.solve(prob, wm, [], [], {}, deliberate=True)
        self.assertEqual(direct["answer"], "生き物")             # trusts the wrong belief
        self.assertNotEqual(delib["answer"], "生き物")           # deliberation catches it


class ExperienceTest(unittest.TestCase):
    def test_experience_records_star_fields_and_a_correction(self):
        state = {"version": cog.VERSION, "rules": [], "experiences": [], "corrections_index": {}}
        prob = {"pid": "p1", "type": "common_property", "level": 3, "concept": "きつね",
                "gold": "生き物", "options": ["生き物"], "grade": "coarse", "prompt": "共通は？"}
        rec = cog.record_experience(state, prob, {"answer": "道具", "confidence": 0.3,
                                                  "steps": ["belief[きつね]=生き物"]},
                                    correct=False, evaluation={"notes": ["x"], "sufficient_basis": True,
                                                               "contradiction": False}, cycle=3)
        for f in ("situation", "thought", "action", "result", "evaluation", "correction"):
            self.assertIn(f, rec)
        self.assertIn("wrong", rec["result"])
        self.assertTrue(cog.corrections_for(state, "common_property", "生き物"))

    def test_repeated_failure_is_flagged(self):
        state = {"version": cog.VERSION, "rules": [], "experiences": [], "corrections_index": {}}
        prob = {"pid": "p", "type": "odd_one_out", "level": 2, "concept": "き",
                "gold": "A", "options": ["A", "B"], "grade": "exact", "prompt": "?"}
        sol = {"answer": "B", "confidence": 0.2, "steps": ["classify ..."]}
        ev = {"notes": [], "sufficient_basis": True, "contradiction": False}
        r1 = cog.record_experience(state, prob, sol, correct=False, evaluation=ev, cycle=1)
        r2 = cog.record_experience(state, prob, sol, correct=False, evaluation=ev, cycle=2)
        self.assertFalse(r1["repeated_failure"])
        self.assertTrue(r2["repeated_failure"])


class ControllerTest(unittest.TestCase):
    def test_controller_learns_the_better_strategy_from_outcomes(self):
        state = {"version": cog.VERSION}
        # "compose" always right, "direct" always wrong, for common_property
        for _ in range(12):
            cog.record_strategy_outcome(state, "common_property", "compose", True)
            cog.record_strategy_outcome(state, "common_property", "direct", False)
        self.assertEqual(cog.choose_strategy(state, "common_property", explore=False), "compose")
        s = cog.controller_summary(state)
        self.assertEqual(s["common_property"]["best"], "compose")
        self.assertGreater(s["common_property"]["rate"], 0.8)

    def test_solve_strategy_flag_sets_the_reasoning_mode(self):
        wm = {"beliefs": {"き": _belief("生き物")}, "contexts": {}, "profiles": {}}
        p = {"type": "genus_recall", "concept": "き", "options": list(cog.COARSE),
             "gold": "生き物", "grade": "coarse", "prompt": "?"}
        # direct trusts the belief; the answer is the same here but the path differs
        self.assertEqual(cog.solve(p, wm, [], [], {}, strategy="direct")["answer"], "生き物")
        self.assertIn(cog.solve(p, wm, [], [], {}, strategy="deliberate")["answer"],
                      ("生き物", "わからない"))


class LoopTest(unittest.TestCase):
    def test_v1_state_migrates_without_erasing_experience(self):
        old = {"version": 1, "rules": [{"rule_id": "kept", "status": "candidate",
                "form": "「道具」が光る", "subject_genus": "道具", "verb": "光る",
                "object_genus": "", "support": 3, "books": ["old"],
                "confidence": .4, "counterexamples": []}],
               "experiences": [{"result": "correct"}], "loop_history": []}
        got = cog.run_cognitive_cycle(2, WM, STORE, SHELF, "", old)
        self.assertTrue(any(r.get("rule_id") == "kept" for r in got["rules"]))
        self.assertTrue(got.get("migration_history"))

    def test_full_cycle_runs_and_round_trips(self):
        r1 = cog.run_cognitive_cycle(1, WM, STORE, SHELF, "b1", None)
        self.assertEqual(r1["status"], "ran")
        self.assertGreater(r1["rules_total"], 0)
        r2 = cog.run_cognitive_cycle(2, WM, STORE, SHELF, "b2", dict(r1))
        self.assertEqual(r2["version"], cog.VERSION)
        self.assertGreaterEqual(r2["experiences_total"], r1["experiences_total"])
        self.assertIn("repeated_failure_rate", r2)

    def test_disabled_by_env(self):
        import os
        os.environ["AI_NOISE_COGNITION"] = "0"
        try:
            self.assertFalse(cog.enabled())
        finally:
            del os.environ["AI_NOISE_COGNITION"]
        self.assertTrue(cog.enabled())


if __name__ == "__main__":
    unittest.main()
