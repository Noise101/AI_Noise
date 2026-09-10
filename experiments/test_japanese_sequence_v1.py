import random
import unittest

import japanese_sequence_v1 as js


def learnable_texts(n=40):
    line = "むかしむかしおじいさんとおばあさんがいました。おじいさんは山へ行きました。"
    return {f"https://ja.wikisource.org/wiki/本{c}": line * 10 for c in range(n)}


def random_texts(n=40):
    rng = random.Random(0)
    kana = "あいうえおかきくけこさしすせそたちつてと。"
    return {f"https://ja.wikisource.org/wiki/乱{c}":
            "".join(rng.choice(kana) for _ in range(700)) for c in range(n)}


class JapaneseSequenceTest(unittest.TestCase):
    def test_normalise_keeps_kana_kanji_and_core_punctuation_only(self):
        self.assertEqual(js.normalise("Fox「桃」を食べた。\n 123"), "「桃」を食べた。")

    def test_collection_key_is_per_work(self):
        a = js.collection_key("https://www.aozora.gr.jp/cards/000121/files/627_13466.html")
        b = js.collection_key("https://www.aozora.gr.jp/cards/000121/files/4253_10259.html")
        self.assertNotEqual(a, b)
        self.assertEqual(
            js.collection_key("https://ja.wikisource.org/wiki/桃太郎"),
            js.collection_key("https://ja.wikisource.org/wiki/桃太郎"))

    def test_character_benchmark_is_diagnostic_only_never_a_capability(self):
        # re-audit #6 P2-1: repeating the same held-out set never confers a
        # confirmed capability -- this benchmark is a learning signal only.
        texts = learnable_texts()
        first = js.train_and_evaluate(texts, {}, train_seconds=30, max_steps=500)
        self.assertGreater(first["improvement_z"], js.SIGNIFICANCE_Z)
        self.assertEqual(first["capability_status"], "diagnostic_only")
        self.assertFalse(first["beats_char_baseline"])
        second = js.train_and_evaluate(texts, first, train_seconds=30, max_steps=500)
        self.assertFalse(second["beats_char_baseline"])           # even after two significant cycles
        self.assertNotEqual(second["status"], "beats_char_baseline")

    def test_negative_control_random_kana_is_not_significant(self):
        report = js.train_and_evaluate(random_texts(), {}, train_seconds=30, max_steps=500)
        self.assertLess(report["improvement_z"], js.SIGNIFICANCE_Z)
        self.assertFalse(report["beats_char_baseline"])

    def test_insufficient_text_reports_cleanly(self):
        report = js.train_and_evaluate({"https://ja.wikisource.org/wiki/短": "みじかい。"}, {})
        self.assertEqual(report["status"], "insufficient_text")
        self.assertIsNone(report["held_out_bits_per_char"])
        self.assertFalse(report["can_sample"])

    def test_benchmark_split_is_frozen_across_growth(self):
        first = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=100)
        held = set(first["benchmark"]["held_out_collections"])
        self.assertTrue(held)
        second = js.train_and_evaluate(learnable_texts(60), first, train_seconds=30, max_steps=100)
        self.assertEqual(set(second["benchmark"]["held_out_collections"]), held)

    def test_japanese_model_has_more_capacity_than_the_english_default(self):
        from sequence_model_v1 import HIDDEN as EN_HIDDEN
        r = self._clean_state()
        self.assertEqual(len(r["state"]["bh"]), js.JA_HIDDEN)
        self.assertGreater(js.JA_HIDDEN, EN_HIDDEN)
        # a loaded state keeps its own size regardless of JA_HIDDEN
        m = js.TinyRNN(r["state"]["vocab"], r["state"])
        self.assertEqual(m.H, js.JA_HIDDEN)

    def test_generate_returns_japanese_characters(self):
        model = js.TinyRNN(sorted("むかしあおじいん。やまへ行きました"))
        out = js.generate(model, "むかし", length=20)
        self.assertEqual(len(out), 20)
        self.assertTrue(all("぀" <= ch or ch in "。" for ch in out))

    def test_model_fingerprint_moves_with_weights_and_steps(self):
        r1 = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertTrue(r1["model_fingerprint"])
        self.assertEqual(r1["model_fingerprint"], r1["state"]["model_fingerprint"])
        same = js.model_fingerprint(r1["state"], r1["state"]["steps_trained"])
        self.assertEqual(same, r1["model_fingerprint"])
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertNotEqual(r2["model_fingerprint"], r1["model_fingerprint"])
        self.assertGreater(r2["state"]["steps_trained"], r1["state"]["steps_trained"])

    # --- re-audit #5 item 1: contamination reset ------------------------
    def _ctx(self, forbidden=()):
        return {"parser_version": 5, "provenance_policy": js.PROVENANCE_POLICY,
                "read_only": True, "forbidden_collections": list(forbidden)}

    def _clean_state(self):
        r = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=80,
                                  training_context=self._ctx())
        return r

    def test_legacy_state_without_training_regime_is_not_continued(self):
        legacy = {"state": {"vocab": list("あいうえお"), "Wxh": [[0.1] * 5] * 24,
                            "Whh": [[0.1] * 24] * 24, "Why": [[0.1] * 24] * 5,
                            "bh": [0.0] * 24, "by": [0.0] * 5},
                  "steps_trained": 2_380_000, "model_fingerprint": "legacyFP00000000"}
        r = js.train_and_evaluate(learnable_texts(40), legacy, train_seconds=30, max_steps=60,
                                  training_context=self._ctx())
        self.assertEqual(r["reset_reason"], "legacy_state_without_training_regime")
        self.assertEqual(r["contamination_status"], "retired_replaced")
        self.assertLess(r["steps_trained"], 1000)                 # 0-based, not +2.38M
        self.assertEqual(r["parent_model_fingerprint"], "legacyFP00000000")
        self.assertEqual(r["retired_model"]["retired_steps_trained"], 2_380_000)
        self.assertTrue(r["retired_model"]["retired_state"])

    def test_a_clean_state_continues_on_the_same_boundary(self):
        r1 = self._clean_state()
        self.assertIsNone(r1["reset_reason"])
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertIsNone(r2["reset_reason"])
        self.assertEqual(r2["contamination_status"], "clean_continued")
        self.assertGreater(r2["steps_trained"], r1["steps_trained"])

    def test_new_read_books_are_growth_not_a_reset(self):
        r1 = self._clean_state()
        r2 = js.train_and_evaluate(learnable_texts(70), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())      # more training data
        self.assertIsNone(r2["reset_reason"])
        self.assertGreater(r2["steps_trained"], r1["steps_trained"])

    def test_a_new_forbidden_collection_it_never_trained_on_does_NOT_retire(self):
        # re-audit #6 P1-1: a brand-new selection/final collection must not wipe
        # a model that has never trained on it.
        r1 = self._clean_state()
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx(forbidden=["https://ja.wikisource.org/wiki/NEVER_TRAINED"]))
        self.assertIsNone(r2["reset_reason"])
        self.assertGreater(r2["steps_trained"], r1["steps_trained"])

    def test_a_semantic_boundary_change_retires_the_model(self):
        r1 = self._clean_state()
        # a change to a semantic training rule (here the vocabulary method) makes
        # the old weights meaningless -> retire.  Since v3 the parser version is
        # NOT in the boundary (the RNN trains on raw text), so it is tested via a
        # rule that still matters.
        import unittest.mock
        with unittest.mock.patch.object(js, "VOCAB_METHOD", "some_other_vocab_method"):
            r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                       training_context=self._ctx())
        self.assertEqual(r2["reset_reason"], "training_boundary_changed")
        self.assertLess(r2["steps_trained"], r1["steps_trained"])
        self.assertEqual(r2["parent_model_fingerprint"], r1["model_fingerprint"])

    def test_a_parser_version_change_does_NOT_retire_the_raw_text_rnn(self):
        # v3: the RNN trains on raw book text, so a parser upgrade cannot
        # invalidate it -- this used to be a spurious-retirement path.
        r1 = self._clean_state()
        ctx2 = self._ctx(); ctx2["parser_version"] = 99
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=ctx2)
        self.assertIsNone(r2["reset_reason"])
        self.assertGreater(r2["steps_trained"], r1["steps_trained"])

    def test_a_provenance_policy_change_retires_the_model(self):
        r1 = self._clean_state()
        ctx2 = self._ctx(); ctx2["provenance_policy"] = "some_other_policy"
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=ctx2)
        self.assertEqual(r2["reset_reason"], "training_boundary_changed")

    def test_forbidding_a_collection_it_HAS_trained_on_retires_the_model(self):
        r1 = self._clean_state()
        trained_col = js._collection_of(next(iter(r1["ever_trained_sources"])))
        r2 = js.train_and_evaluate(learnable_texts(40), r1, train_seconds=30, max_steps=60,
                                   training_context=self._ctx(forbidden=[trained_col]))
        self.assertTrue(r2["reset_reason"].startswith("trained_data_invalidated"))
        self.assertIn("trained_collection_now_held_out",
                      {c["kind"] for c in r2["reset_collisions"]})
        self.assertLess(r2["steps_trained"], r1["steps_trained"])

    def test_the_retired_state_is_returned_for_archiving_and_streaks_reset(self):
        legacy = {"state": {"vocab": list("あいうえお"), "Wxh": [[0.1] * 5] * 24,
                            "Whh": [[0.1] * 24] * 24, "Why": [[0.1] * 24] * 5,
                            "bh": [0.0] * 24, "by": [0.0] * 5},
                  "steps_trained": 999_999, "model_fingerprint": "x",
                  "significant_streak": 2}
        r = js.train_and_evaluate(learnable_texts(40), legacy, train_seconds=30, max_steps=40,
                                  training_context=self._ctx())
        self.assertTrue(r["retired_model"]["retired_state"])
        self.assertLessEqual(r["significant_streak"], 1)        # not inherited (was 2)
        self.assertFalse(r.get("beats_char_baseline"))
        self.assertEqual(len(r["retirement_log"]), 1)

    def test_an_incompatible_regime_change_retires_and_archives(self):
        # v2 -> v3 is a real representation + corpus change: NOT a compatible
        # predecessor.  The 11.3M-step v2 model (worse than a frequency table) is
        # archived, and training restarts clean.
        r1 = self._clean_state()
        legacy_v2 = {"state": r1["state"], "steps_trained": 11_300_000,
                     "model_fingerprint": "oldv2", "training_regime": "jseq_clean_v2",
                     "ever_trained_sources": dict(r1["ever_trained_sources"]),
                     "significant_streak": 3}
        r2 = js.train_and_evaluate(learnable_texts(40), legacy_v2, train_seconds=30, max_steps=60,
                                   training_context=self._ctx())
        self.assertEqual(r2["reset_reason"], f"training_regime_changed:jseq_clean_v2->{js.TRAINING_REGIME}")
        self.assertEqual(r2["contamination_status"], "retired_replaced")
        self.assertLess(r2["steps_trained"], 1000)
        self.assertEqual(r2["retired_model"]["retired_steps_trained"], 11_300_000)
        self.assertTrue(r2["retired_model"]["retired_state"])
        self.assertEqual(r2["training_regime"], js.TRAINING_REGIME)

    def test_a_compatible_predecessor_regime_migrates_without_retiring(self):
        # the in-place migration path still exists for future compatible bumps;
        # exercise it by declaring v2 compatible for the duration of the test.
        import unittest.mock
        r1 = self._clean_state()
        legacy = {"state": r1["state"], "steps_trained": 356583, "model_fingerprint": "keepme",
                  "training_regime": "jseq_clean_v2",
                  "training_data_fingerprint": {"training_sources": [
                      {"url": u, "text_hash": h} for u, h in r1["ever_trained_sources"].items()],
                      "boundary_fingerprint": "OLD"}}
        with unittest.mock.patch.object(js, "COMPATIBLE_PREDECESSOR_REGIMES", ("jseq_clean_v2",)):
            r2 = js.train_and_evaluate(learnable_texts(40), legacy, train_seconds=30, max_steps=60,
                                       training_context=self._ctx())
        self.assertIsNone(r2["reset_reason"])
        self.assertTrue(r2["boundary_migrated"])
        self.assertGreaterEqual(r2["steps_trained"], 356583)
        self.assertEqual(r2["training_regime"], js.TRAINING_REGIME)

    def test_training_data_fingerprint_is_auditable(self):
        r = self._clean_state()
        tdf = r["training_data_fingerprint"]
        for k in ("boundary_fingerprint", "ever_trained_set_fingerprint", "training_regime",
                  "split_policy_version", "parser_version", "provenance_policy",
                  "normalisation_version", "ever_trained_source_count",
                  "ever_trained_sources", "ever_trained_collections"):
            self.assertIn(k, tdf)
        self.assertEqual(len(tdf["ever_trained_sources"]), tdf["ever_trained_source_count"])
        # the boundary fingerprint does NOT depend on the dynamic forbidden list
        # nor (since v3) on the parser version
        b1 = js.boundary_fingerprint({"parser_version": 5, "forbidden_collections": ["a"]})
        b2 = js.boundary_fingerprint({"parser_version": 99, "forbidden_collections": ["a", "b", "c"]})
        self.assertEqual(b1, b2)

    # --- v3: OOV chars map to UNK instead of skipping the training window ------
    def test_build_vocab_always_includes_unk(self):
        vocab = js.build_vocab({"u": "あいうえお。" * 10})
        self.assertIn(js.UNK, vocab)
        self.assertEqual(js.unk_index(vocab), vocab.index(js.UNK))

    def test_a_window_with_a_rare_char_still_trains(self):
        # a 200-char vocab used to skip any window touching a rare kanji; with UNK
        # the window contributes a real gradient step.
        vocab = sorted("むかしあおじいん。やま") + [js.UNK]
        model = js.TinyRNN(vocab, unk_index=js.unk_index(vocab))
        before = [row[:] for row in model.Why]
        loss = model.train_step("むかし驫驫あじいさん。" + "あ" * 40, js.LEARNING_RATE)
        self.assertGreater(loss, 0.0)
        self.assertNotEqual(model.Why, before)

    def test_oov_positions_are_scored_via_unk_not_dropped(self):
        vocab = sorted("あいうえお。") + [js.UNK]
        model = js.TinyRNN(vocab, unk_index=js.unk_index(vocab))
        bpc, n = model.bits_per_char("あ驫あ驫あ。")     # 2 rare chars
        self.assertEqual(n, 5)                            # every transition counted
        self.assertGreater(bpc, 0.0)

    def test_english_tinyrnn_unchanged_when_no_unk_index(self):
        from sequence_model_v1 import TinyRNN as EnRNN
        model = EnRNN(sorted("the quick brown fox. "))
        self.assertIsNone(model._idx("Z"))               # OOV -> None, as before
        self.assertEqual(model.train_step("Zzz the fox. ", 0.02), 0.0)   # window skipped

    # --- P1-1: general vs dedicated-narrative regime -----------------------
    def test_regime_param_stamps_a_distinct_identity(self):
        gen = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=60,
                                    regime=js.TRAINING_REGIME, training_context=self._ctx())
        narr = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=60,
                                     regime=js.NARRATIVE_REGIME, training_context=self._ctx())
        self.assertEqual(gen["training_regime"], js.TRAINING_REGIME)
        self.assertEqual(narr["training_regime"], js.NARRATIVE_REGIME)
        self.assertEqual(narr["state"]["training_regime"], js.NARRATIVE_REGIME)
        self.assertNotEqual(gen["training_data_fingerprint"]["boundary_fingerprint"],
                            narr["training_data_fingerprint"]["boundary_fingerprint"])

    def test_feeding_a_general_state_into_a_narrative_call_retires_not_continues(self):
        # the two models must never silently share weights: a regime mismatch is
        # a retirement, so a stale file in the wrong slot cannot leak.
        gen = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=80,
                                    regime=js.TRAINING_REGIME, training_context=self._ctx())
        crossed = js.train_and_evaluate(learnable_texts(40), gen, train_seconds=30, max_steps=60,
                                        regime=js.NARRATIVE_REGIME, training_context=self._ctx())
        self.assertEqual(crossed["reset_reason"],
                         f"training_regime_changed:{js.TRAINING_REGIME}->{js.NARRATIVE_REGIME}")
        self.assertLess(crossed["steps_trained"], 1000)
        self.assertEqual(crossed["training_regime"], js.NARRATIVE_REGIME)

    def test_a_narrative_state_continues_on_its_own_regime(self):
        n1 = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=80,
                                   regime=js.NARRATIVE_REGIME, training_context=self._ctx())
        n2 = js.train_and_evaluate(learnable_texts(40), n1, train_seconds=30, max_steps=60,
                                   regime=js.NARRATIVE_REGIME, training_context=self._ctx())
        self.assertIsNone(n2["reset_reason"])
        self.assertGreater(n2["steps_trained"], n1["steps_trained"])

    def test_held_out_split_has_a_min_eval_sources_floor(self):
        r = js.train_and_evaluate(learnable_texts(40), {}, train_seconds=30, max_steps=40,
                                  training_context=self._ctx())
        self.assertGreaterEqual(len(r["benchmark"]["held_out_collections"]), js.MIN_EVAL_SOURCES)
        # the per-source eval cap must leave room to actually score that many
        self.assertGreaterEqual(r["held_out_sources_evaluated"], js.MIN_EVAL_SOURCES)


if __name__ == "__main__":
    unittest.main()
