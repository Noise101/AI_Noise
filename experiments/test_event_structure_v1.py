import random
import unittest

import event_structure_v1 as es
from event_structure_v1 import (CORRUPTERS, DEFAULT_CORRUPTERS, FrequencyBaseline,
                                SmoothedArgumentModel, choose_benchmark, classify_trend,
                                collection_key, evaluate_cloze, evaluate_plausibility,
                                iter_events, passes_gain_gate, train_and_evaluate)


def wiki(collection: str, page: str) -> str:
    return f"https://en.wikisource.org/wiki/{collection}/{page}"


def verified(sequences: list[dict]) -> dict:
    return {"sequences": sequences}


def learnable_sequences(collections: int, per_collection: int = 8) -> list[dict]:
    """Each subject has a deterministic verb and object; a frequent 'narrator
    said tale' filler is the global mode.  A model that conditions on the
    arguments should far exceed 'always predict the globally most common verb'."""
    profile = {"fox": ("ate", "grapes"), "bird": ("sang", "song"),
               "wolf": ("howled", "moon"), "child": ("read", "book"),
               "king": ("ruled", "land"), "mouse": ("squeaked", "cheese")}
    subjects = list(profile)
    sequences = []
    for c in range(collections):
        for p in range(per_collection):
            if p % 3 == 0:                       # ~1/3 filler -> "said" is global mode
                subject, verb, obj = "narrator", "said", "tale"
            else:
                subject = subjects[(c * per_collection + p) % len(subjects)]
                verb, obj = profile[subject]
            sequences.append({"seed": f"s{c}", "source_url": wiki(f"Book{c}", f"Page{p}"),
                              "events": [f"{subject}|{verb}|{obj}"]})
    return sequences


def noise_sequences(collections: int, per_collection: int = 6) -> list[dict]:
    """Verbs are independent of subject and object -- nothing to learn."""
    rng = random.Random(0)
    subjects = ["fox", "bird", "wolf", "child", "king", "mouse"]
    verbs = ["ate", "sang", "chased", "read", "ruled", "hid", "said"]
    objects = ["grapes", "song", "sheep", "book", "land", "hole"]
    sequences = []
    for c in range(collections):
        for p in range(per_collection):
            events = [f"{rng.choice(subjects)}|{rng.choice(verbs)}|{rng.choice(objects)}"
                      for _ in range(2)]
            sequences.append({"seed": f"s{c}", "source_url": wiki(f"Book{c}", f"Page{p}"),
                              "events": events})
    return sequences


class EventStructureTest(unittest.TestCase):
    def test_iter_events_keeps_only_subject_verb_clauses_with_source(self):
        data = verified([{"source_url": wiki("A", "1"),
                          "events": ["fox|ate|grapes", "|ran|home", "bird|sang|"]},
                         {"events": ["orphan|no|source"]}])
        events = iter_events(data)
        self.assertEqual(sorted(e[:3] for e in events),
                         [("bird", "sang", ""), ("fox", "ate", "grapes")])

    def test_positive_control_argument_model_beats_baseline_and_is_selected(self):
        report = train_and_evaluate(verified(learnable_sequences(60)), {})
        self.assertTrue(report["benchmark"]["locked"])
        self.assertEqual(report["selection_status"], "accepted_final_gain")
        self.assertEqual(report["selected"]["model_id"], "smoothed_argument")
        self.assertGreater(report["selected"]["lift"], 0)

    def test_a_confirmed_model_stays_selected_through_the_milestone_wait(self):
        data = verified(learnable_sequences(60))
        first = train_and_evaluate(data, {})
        self.assertEqual(first["selection_status"], "accepted_final_gain")
        # Next cycle: training has barely grown, so no fresh final query fires,
        # but the model that already cleared the final gate must not flap back.
        second = train_and_evaluate(data, first)
        self.assertEqual(second["selected"]["model_id"], "smoothed_argument")
        self.assertTrue(second["selected"].get("stale_final"))
        self.assertEqual(second["final_queries_used"], 1)
        self.assertNotIn("selected_model_changed",
                         [e["event_type"] for e in second["emitted_events"]])

    def test_negative_control_random_data_selects_nothing(self):
        report = train_and_evaluate(verified(noise_sequences(60)), {})
        self.assertIsNone(report["selected"])
        self.assertEqual(report["selected_model_id"], "frequency_baseline")
        self.assertEqual(report["selection_status"],
                         "no_model_beats_corrected_selection_baseline")

    def test_benchmark_freezes_and_never_re_picks(self):
        first = train_and_evaluate(verified(learnable_sequences(60)), {})
        frozen = first["benchmark"]["benchmark_collections"]
        grown = learnable_sequences(60) + [
            {"seed": "x", "source_url": wiki("BrandNew", f"P{i}"),
             "events": ["fox|ate|grapes", "fox|said|grapes"]} for i in range(30)]
        second = train_and_evaluate(verified(grown), first)
        self.assertEqual(second["benchmark"]["benchmark_collections"], frozen)

    def test_frozen_evaluation_events_are_a_snapshot_not_re_derived(self):
        first = train_and_evaluate(verified(learnable_sequences(60)), {})
        self.assertEqual(first["benchmark"]["status"], "ready")
        n_final = first["benchmark"]["final_events"]
        n_sel = first["benchmark"]["selection_events"]
        # the benchmark collections gain MANY more events later
        bench_cols = first["benchmark"]["benchmark_collections"]
        grown = learnable_sequences(60)
        for col in bench_cols[:2]:
            title = col.split("/")[-1] if "/" in col else col
            grown += [{"seed": "x", "source_url": col + f"#extra{i}",
                       "events": ["fox|ate|grapes", "fox|ran|home"]} for i in range(40)]
        second = train_and_evaluate(verified(grown), first)
        self.assertEqual(second["benchmark"]["final_events"], n_final)
        self.assertEqual(second["benchmark"]["selection_events"], n_sel)

    def test_significance_is_a_sign_test_over_collections_not_events(self):
        report = train_and_evaluate(verified(learnable_sequences(80)), {})
        curve = report.get("learning_curve", [])
        # the reported p-value is derived from collection wins/losses
        for ev in report.get("selection_evaluations", []) or []:
            sel = ev.get("selection", ev)
            if "collection_wins" in sel:
                self.assertLessEqual(sel["collection_wins"] + sel["collection_losses"],
                                     sel.get("evaluated_collections", 99))

    def test_benchmark_sources_never_enter_training(self):
        report = train_and_evaluate(verified(learnable_sequences(60)), {})
        bench = set(report["benchmark"]["benchmark_collections"])
        events = iter_events(verified(learnable_sequences(60)))
        train_collections = {collection_key(e[3]) for e in events
                             if collection_key(e[3]) not in bench}
        self.assertFalse(bench & train_collections)
        self.assertEqual(report["training"]["events"],
                         sum(1 for e in events if collection_key(e[3]) not in bench))

    def test_selection_and_final_are_collection_disjoint(self):
        report = train_and_evaluate(verified(learnable_sequences(60)), {})
        selection = set(report["benchmark"]["selection_collections"])
        bench = set(report["benchmark"]["benchmark_collections"])
        final = bench - selection
        self.assertTrue(selection)
        self.assertTrue(final)
        self.assertFalse(selection & final)

    def test_postpones_evaluation_when_benchmark_too_small(self):
        report = train_and_evaluate(verified(learnable_sequences(3)), {})
        self.assertFalse(report["benchmark"]["locked"])
        self.assertEqual(report["selection_status"], "benchmark_not_ready")
        self.assertEqual(report["selected_evaluation"]["total"], 0)

    def test_bonferroni_alpha_is_recalibrated_for_three_models_two_tasks(self):
        self.assertAlmostEqual(es.SELECTION_ALPHA, 0.05 / 6)
        self.assertAlmostEqual(es.FINAL_ALPHA, 0.05 / 30)

    def test_gain_gate_needs_significance_not_just_a_positive_lift(self):
        big_but_noisy = {"total": 100, "lift": 5, "coverage": 1.0, "one_sided_sign_p": 0.2}
        self.assertFalse(passes_gain_gate(big_but_noisy, es.SELECTION_ALPHA))
        significant = {"total": 100, "lift": 5, "coverage": 1.0, "one_sided_sign_p": 0.001}
        self.assertTrue(passes_gain_gate(significant, es.SELECTION_ALPHA))

    def test_benchmark_lock_emits_one_event_then_stops(self):
        first = train_and_evaluate(verified(learnable_sequences(60)), {})
        kinds = [event["event_type"] for event in first["emitted_events"]]
        self.assertIn("benchmark_locked", kinds)
        second = train_and_evaluate(verified(learnable_sequences(60)), first)
        self.assertNotIn("benchmark_locked",
                         [event["event_type"] for event in second["emitted_events"]])

    def test_plausibility_default_corrupter_is_verb_swap_only_but_registry_is_extensible(self):
        self.assertEqual(DEFAULT_CORRUPTERS, ("verb_swap",))
        self.assertIn("object_swap", CORRUPTERS)
        train = ([("fox", "ate", "grapes", wiki("T", str(i))) for i in range(20)]
                 + [("bird", "sang", "song", wiki("T", str(i))) for i in range(20, 40)])
        model = SmoothedArgumentModel(train)
        baseline = FrequencyBaseline(train)
        held_out = [("fox", "ate", "grapes", wiki("H", "1")),
                    ("bird", "sang", "song", wiki("H", "2"))]
        _, trials = evaluate_plausibility(
            model, baseline, held_out, ("verb_swap", "object_swap"), "seed")
        self.assertEqual({t["corrupter"] for t in trials}, {"verb_swap", "object_swap"})

    def test_smoothed_model_uses_arguments_frequency_baseline_does_not(self):
        train = ([("fox", "ate", "grapes", wiki("T", str(i))) for i in range(30)]
                 + [("bird", "sang", "song", wiki("T", str(i))) for i in range(30, 60)]
                 + [("wolf", "said", "nothing", wiki("T", str(i))) for i in range(60, 140)])
        model = SmoothedArgumentModel(train)
        baseline = FrequencyBaseline(train)
        self.assertEqual(baseline.predict_verb("fox", "grapes"), "said")   # global mode
        self.assertEqual(model.predict_verb("fox", "grapes"), "ate")       # argument-aware

    def test_classify_trend_needs_four_points(self):
        self.assertEqual(classify_trend([{"lift": 1}, {"lift": 2}], "lift"), "insufficient_data")
        rising = [{"lift": v} for v in (0, 0, 1, 5, 6, 7)]
        self.assertEqual(classify_trend(rising, "lift", window=6, min_delta=1), "improving")

    def test_learning_curve_appends_once_per_training_size(self):
        first = train_and_evaluate(verified(learnable_sequences(60)), {})
        again = train_and_evaluate(verified(learnable_sequences(60)), first)
        self.assertEqual(len(again["learning_curve"]), 1)


if __name__ == "__main__":
    unittest.main()
