import random
import unittest

from world_model_v51 import (BENCHMARK_REGIME, FINAL_QUERY_BUDGET, build_sequences,
                             choose_benchmark_sources, collection_key, narrative_sequence,
                             narrative_source, parse_frame, passes_gain_gate, train_and_evaluate)


# Positive control: a genuine (noisy) prior-action -> next-action Markov chain, so the
# pipeline's detection power can be checked on data with real learnable structure, not
# only on garbage data it must correctly reject (see diverse_audit_for_sources below).
CYCLE_ACTORS = ("fox", "hare", "wolf", "mouse", "bird", "lion", "deer", "owl", "goat", "crow")
CYCLE_ACTIONS = ("saw", "found", "ate", "left")
CYCLE_OBJECTS = {"saw": "food", "found": "food", "ate": "food", "left": "home"}
CYCLE_NEXT = {"saw": "found", "found": "ate", "ate": "left", "left": "saw"}


def noisy_markov_chain_audit(collections=90, steps=8, noise_rate=0.25, seed=1234):
    """Every collection follows the same prior-action -> next-action rule, with a
    noise_rate chance per step of a uniformly random deviation instead."""
    rng = random.Random(seed)
    records = {}
    for index in range(collections):
        actor = CYCLE_ACTORS[index % len(CYCLE_ACTORS)]
        url = f"https://story.example/Animal_Stories_{index}/chapter"
        action = CYCLE_ACTIONS[index % len(CYCLE_ACTIONS)]
        sentences = [f"The {actor} {action} {CYCLE_OBJECTS[action]}."]
        for _ in range(steps - 1):
            true_next = CYCLE_NEXT[action]
            action = rng.choice(CYCLE_ACTIONS) if rng.random() < noise_rate else true_next
            sentences.append(f"The {actor} {action} {CYCLE_OBJECTS[action]}.")
        for position, sentence in enumerate(sentences):
            records[f"{index}:{position}"] = {"source_url": url, "source_position": position,
                "sentence": sentence, "curriculum_admitted": True}
    return {"records": records}


def audit_for_sources(count=30):
    records = {}
    for source in range(count):
        url = f"https://story.example/Animal_Stories_{source}/chapter"
        sentences = ["The fox was hungry.", "The fox wanted to eat.",
                     "The fox tried to reach food.", "The fox ate food."]
        for position, sentence in enumerate(sentences):
            records[f"{source}:{position}"] = {"source_url": url, "seed": f"story {source}",
                "source_position": position, "sentence": sentence, "curriculum_admitted": True}
    return {"records": records}


def _linear_audit(sentences):
    return {"records": {str(index): {"source_url": "https://story.example/fox/chapter",
                                     "source_position": index, "sentence": sentence,
                                     "curriculum_admitted": True}
                        for index, sentence in enumerate(sentences)}}


def rich_audit_for_sources(count=25, sentences_per_source=12):
    """Fewer, longer collections than audit_for_sources -- enough per-collection
    volume to test readiness by example count rather than collection count."""
    records = {}
    cycle = ("The fox saw food.", "The fox found food.", "The fox ate food.",
             "The fox left home.")
    for source in range(count):
        url = f"https://story.example/Animal_Stories_{source}/chapter"
        sentences = [cycle[index % len(cycle)] for index in range(sentences_per_source)]
        for position, sentence in enumerate(sentences):
            records[f"{source}:{position}"] = {"source_url": url, "source_position": position,
                "sentence": sentence, "curriculum_admitted": True}
    return {"records": records}


def diverse_audit_for_sources(count=30):
    stories = (
        ("fox", "cold", "find", "shelter"), ("hare", "tired", "seek", "water"),
        ("lion", "angry", "catch", "thief"), ("girl", "lost", "return", "home"),
        ("boy", "afraid", "help", "friend"), ("wolf", "hungry", "reach", "bread"),
        ("mouse", "small", "open", "door"), ("bird", "thirsty", "drink", "rain"),
    )
    records = {}
    action_patterns = (("saw", "found", "ate"), ("wanted", "tried", "left"),
                       ("found", "drank", "left"), ("saw", "left", "found"),
                       ("drank", "saw", "ate"), ("tried", "found", "left"))
    for source in range(count):
        actor, state, goal, obj = stories[source % len(stories)]
        url = f"https://story.example/Animal_Tales_{source}/chapter"
        actions = action_patterns[source % len(action_patterns)]
        sentences = (f"The {actor} was {state}.",) + tuple(
            f"The {actor} {action} {obj}." for action in actions)
        # Deliberate conflicts prevent a uniform URL-memorisation fixture.
        if source % 7 == 0:
            sentences = sentences + (f"The {actor} did not find {obj}.",)
        for position, sentence in enumerate(sentences):
            records[f"{source}:{position}"] = {"source_url": url, "source_position": position,
                "sentence": sentence, "curriculum_admitted": True}
    return {"records": records}


class WorldModelV51Test(unittest.TestCase):
    def test_history_is_not_mistaken_for_story(self):
        self.assertFalse(narrative_source("https://example/History_of_Oregon"))
        self.assertTrue(narrative_source("https://example/Animal_Stories/Fox"))

    def test_benchmark_requires_recurrent_actor_and_action_change(self):
        self.assertFalse(narrative_sequence({"frames": [
            {"actor": "a", "action": "said"}, {"actor": "b", "action": "said"},
            {"actor": "c", "action": "said"}, {"actor": "d", "action": "said"}]}))
        self.assertTrue(narrative_sequence({"frames": [
            {"actor": "fox", "action": "saw"}, {"actor": "fox", "action": "wanted"},
            {"actor": "fox", "action": "tried"}, {"actor": "fox", "action": "ate"}]}))

    def test_frame_keeps_state_goal_action_and_source_sentence(self):
        frame = parse_frame("The fox was hungry and wanted to eat food.")
        self.assertIsNotNone(frame)
        self.assertEqual(frame["actor"], "fox")
        self.assertIn("hungry", frame["states"])
        self.assertIn("eat", frame["goals"])
        self.assertEqual(frame["sentence"], "The fox was hungry and wanted to eat food.")

    def test_benchmark_is_locked_and_never_used_for_training(self):
        first = train_and_evaluate(audit_for_sources())
        expanded = audit_for_sources(35)
        second = train_and_evaluate(expanded, first)
        self.assertEqual(first["benchmark"]["source_urls"], second["benchmark"]["source_urls"])
        self.assertTrue(set(second["benchmark"]["source_urls"]).isdisjoint(
            second["training"]["source_urls"]))
        self.assertEqual(second["benchmark"]["fingerprint"], first["benchmark"]["fingerprint"])
        self.assertEqual(first["benchmark"]["selection_regime"], BENCHMARK_REGIME)

    def test_benchmark_waits_instead_of_consuming_a_small_or_ineligible_pool(self):
        small = audit_for_sources(15)
        sequences = build_sequences(small)
        self.assertEqual(choose_benchmark_sources(sequences), [])
        result = train_and_evaluate(small)
        self.assertFalse(result["benchmark"]["locked"])
        self.assertEqual(result["training"]["source_count"], 15)
        self.assertEqual(result["selection_status"], "benchmark_not_ready")
        # Readiness is now reported as example counts, not a bare collection count.
        self.assertIn("candidate_selection_examples", result["benchmark"])
        self.assertIn("minimum_train_examples", result["benchmark"])

    def test_benchmark_unlocks_below_the_old_thirty_collection_floor_when_examples_suffice(self):
        # 25 collections would have failed the old len(groups) < 30 gate outright,
        # regardless of how much evidence each one carried. Readiness must now be
        # judged by whether the resulting selection/final/train split actually has
        # enough predictive pairs (12 sentences/collection here comfortably does).
        audit = rich_audit_for_sources(count=25, sentences_per_source=12)
        sequences = build_sequences(audit)
        self.assertEqual(len({collection_key(url) for url in sequences}), 25)
        self.assertTrue(choose_benchmark_sources(sequences))
        result = train_and_evaluate(audit)
        self.assertTrue(result["benchmark"]["locked"])
        self.assertGreaterEqual(result["benchmark"]["selection_examples"], 15)
        self.assertGreaterEqual(result["benchmark"]["final_examples"], 15)
        self.assertGreaterEqual(result["training"]["examples"], 15)

    def test_same_collection_is_never_split_between_training_and_benchmark(self):
        audit = audit_for_sources(30)
        extra = audit_for_sources(1)["records"]
        for key, row in extra.items():
            row["source_url"] = "https://story.example/Animal_Stories_0/another-chapter"
            audit["records"]["sibling:" + key] = row
        result = train_and_evaluate(audit)
        benchmark_groups = {collection_key(url) for url in result["benchmark"]["source_urls"]}
        train_groups = {collection_key(url) for url in result["training"]["source_urls"]}
        self.assertTrue(benchmark_groups.isdisjoint(train_groups))

    def test_no_candidate_is_adopted_without_fixed_benchmark_gain(self):
        result = train_and_evaluate(audit_for_sources())
        if result["selected_evaluation"]["lift"] <= 0:
            self.assertEqual(result["selected_mode"], "frequency_baseline")
        self.assertIn("benchmark_sources_never_train", result["invariants"])

    def test_diverse_conflicting_sources_do_not_manufacture_generalization(self):
        audit = diverse_audit_for_sources()
        self.assertGreater(len({row["sentence"] for row in audit["records"].values()}), 20)
        result = train_and_evaluate(audit)
        self.assertTrue(result["benchmark"]["locked"])
        self.assertEqual(result["selected_mode"], "frequency_baseline")

    def test_pipeline_detects_a_genuine_noisy_but_learnable_pattern(self):
        # Positive control: every prior-action -> next-action test must clear the
        # strict, Bonferroni-corrected gates on data that actually has structure,
        # not just correctly reject data that doesn't (see the test above). A 25%
        # per-step noise rate keeps this from being a near-deterministic strawman.
        result = train_and_evaluate(noisy_markov_chain_audit())
        self.assertTrue(result["benchmark"]["locked"])
        self.assertNotEqual(result["selected_mode"], "frequency_baseline")
        self.assertEqual(result["selection_status"], "accepted_final_gain")
        selected = result["selected_evaluation"]
        self.assertGreater(selected["lift"], 30)
        self.assertLess(selected["one_sided_sign_p"], 1e-6)
        final = result["final_attempt"]["evaluation"]
        self.assertLess(final["one_sided_sign_p"], 1e-6)

    def test_pipeline_rejects_a_prior_action_rule_when_there_is_no_real_signal(self):
        # Same generator, same collection/example counts, but noise_rate=1.0 means
        # prior action carries no information about the next one: the mechanism
        # must not manufacture a detection out of pure noise at this scale either.
        result = train_and_evaluate(noisy_markov_chain_audit(noise_rate=1.0, seed=42))
        self.assertTrue(result["benchmark"]["locked"])
        self.assertEqual(result["selected_mode"], "frequency_baseline")

    def test_small_or_unpaired_gain_cannot_clear_gate(self):
        self.assertFalse(passes_gain_gate({"total": 30, "lift": 3, "coverage": 1.0,
                                          "one_sided_sign_p": .2}))
        self.assertFalse(passes_gain_gate({"total": 10, "lift": 10, "coverage": 1.0,
                                          "one_sided_sign_p": .01}))
        self.assertFalse(passes_gain_gate({"total": 30, "lift": 4, "coverage": .5,
                                          "one_sided_sign_p": .05}))
        self.assertTrue(passes_gain_gate({"total": 30, "lift": 4, "coverage": .5,
                                         "one_sided_sign_p": .001}))

    def test_state_persists_into_later_actor_experience(self):
        sequences = build_sequences(audit_for_sources(1))
        frames = next(iter(sequences.values()))["frames"]
        eating = next(frame for frame in frames if frame["action"] == "ate")
        self.assertIn("hungry", eating["known_states"])

    def test_multiword_negation_removes_state_without_garbage(self):
        frame = parse_frame("The fox was no longer hungry.")
        self.assertNotIn("longer", frame["states"])
        self.assertIn("hungry", frame["states"])
        self.assertEqual(frame["polarity"], "negative")
        audit = {"records": {
            "0": {"source_url": "https://story.example/fox/chapter", "source_position": 0,
                  "sentence": "The fox was hungry.", "curriculum_admitted": True},
            "1": {"source_url": "https://story.example/fox/chapter", "source_position": 1,
                  "sentence": "The fox was no longer hungry.", "curriculum_admitted": True},
            "2": {"source_url": "https://story.example/fox/chapter", "source_position": 2,
                  "sentence": "The fox ate food.", "curriculum_admitted": True}}}
        frames = next(iter(build_sequences(audit).values()))["frames"]
        self.assertNotIn("hungry", frames[-1]["known_states"])

    def test_frame_subject_is_head_noun_not_leading_adjective(self):
        frame = parse_frame("The hungry fox saw grapes.")
        self.assertIsNotNone(frame)
        self.assertEqual(frame["actor"], "fox")
        self.assertEqual(frame["action"], "saw")

    def test_non_conflicting_conditions_accumulate_instead_of_superseding(self):
        frames = next(iter(build_sequences(_linear_audit([
            "The fox was hungry.", "The fox was brave.", "The fox ate food."])).values()))["frames"]
        self.assertIn("hungry", frames[-1]["known_states"])
        self.assertIn("brave", frames[-1]["known_states"])
        self.assertEqual(frames[1]["superseded_states"], [])

    def test_new_condition_supersedes_old_condition(self):
        audit = {"records": {}}
        for index, sentence in enumerate(("The fox was hungry.", "The fox found food.",
                                           "The fox was full.", "The fox left home.")):
            audit["records"][str(index)] = {"source_url": "https://story.example/fox/chapter",
                "source_position": index, "sentence": sentence, "curriculum_admitted": True}
        frames = next(iter(build_sequences(audit).values()))["frames"]
        self.assertIn("full", frames[-1]["known_states"])
        self.assertNotIn("hungry", frames[-1]["known_states"])

    def test_frozen_benchmark_examples_do_not_change_when_source_grows(self):
        audit = audit_for_sources()
        first = train_and_evaluate(audit)
        benchmark_url = first["benchmark"]["source_urls"][0]
        audit["records"]["late:0"] = {"source_url": benchmark_url, "seed": "late",
            "source_position": 99, "sentence": "The fox ate berries.",
            "curriculum_admitted": True}
        second = train_and_evaluate(audit, first)
        self.assertEqual(first["benchmark_examples"], second["benchmark_examples"])
        self.assertEqual(first["benchmark"]["examples"], second["benchmark"]["examples"])

    def test_final_holdout_is_not_evaluated_during_candidate_search(self):
        result = train_and_evaluate(audit_for_sources())
        self.assertTrue(all("final" not in evaluation for evaluation in result["evaluations"]))
        if result["final_attempt"] is None:
            again = train_and_evaluate(audit_for_sources(35), result)
            self.assertIsNone(again["final_attempt"])

    def test_early_final_miss_is_rechecked_after_training_growth(self):
        audit = audit_for_sources(40)
        baseline = train_and_evaluate(audit)
        model_id = baseline["final_attempt"]["model_id"]
        # A data-starved first final query that failed must not be a permanent verdict.
        starved_miss = {"model_id": model_id, "training_examples": 2, "query_index": 1,
                        "trials": [], "evaluation": {"total": 4, "lift": 0, "coverage": 0.0,
                                                     "one_sided_sign_p": 1.0}}
        previous = {"final_attempt": starved_miss, "final_attempt_history": [starved_miss]}
        rechecked = train_and_evaluate(audit, previous)
        self.assertEqual(rechecked["final_queries_used"], 2)
        self.assertNotEqual(rechecked["selection_status"],
                            "final_holdout_query_budget_exhausted_no_confirmed_gain")

    def test_final_query_is_not_repeated_before_the_next_training_milestone(self):
        audit = audit_for_sources(40)
        baseline = train_and_evaluate(audit)
        model_id = baseline["final_attempt"]["model_id"]
        recent_miss = {"model_id": model_id,
                       "training_examples": baseline["training"]["examples"],
                       "query_index": 1, "trials": [],
                       "evaluation": {"total": 4, "lift": 0, "coverage": 0.0,
                                      "one_sided_sign_p": 1.0}}
        previous = {"final_attempt": recent_miss, "final_attempt_history": [recent_miss]}
        held = train_and_evaluate(audit, previous)
        self.assertEqual(held["final_queries_used"], 1)
        self.assertEqual(held["selection_status"],
                         "awaiting_training_growth_for_next_final_query")

    def test_final_holdout_query_budget_is_finite(self):
        audit = audit_for_sources(40)
        baseline = train_and_evaluate(audit)
        model_id = baseline["final_attempt"]["model_id"]
        miss = {"model_id": model_id, "training_examples": 1, "query_index": 1, "trials": [],
                "evaluation": {"total": 4, "lift": 0, "coverage": 0.0, "one_sided_sign_p": 1.0}}
        previous = {"final_attempt": miss,
                    "final_attempt_history": [dict(miss) for _ in range(FINAL_QUERY_BUDGET)]}
        exhausted = train_and_evaluate(audit, previous)
        self.assertEqual(exhausted["final_queries_used"], FINAL_QUERY_BUDGET)
        self.assertEqual(exhausted["selected_mode"], "frequency_baseline")
        self.assertEqual(exhausted["selection_status"],
                         "final_holdout_query_budget_exhausted_no_confirmed_gain")

    def test_failure_target_rotates_across_patterns(self):
        first = train_and_evaluate(audit_for_sources())
        if first["next_learning_target"]:
            second = train_and_evaluate(audit_for_sources(35), first)
            if len(first.get("counterexample_patterns", [])) > 1 and second["next_learning_target"]:
                self.assertNotEqual(first["next_learning_target"]["failure_pattern"],
                                    second["next_learning_target"]["failure_pattern"])


if __name__ == "__main__":
    unittest.main()
