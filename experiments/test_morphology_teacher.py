import os
import unittest

import morphology_teacher as mt


class NullBackendTest(unittest.TestCase):
    def test_null_backend_is_never_available_and_returns_nothing(self):
        null = mt.NullMorphologyTeacher()
        self.assertFalse(null.available())
        self.assertIsNone(null.analyse("犬が走る。"))

    def test_env_var_forces_the_null_backend(self):
        prev = os.environ.get("AI_NOISE_NO_MORPHOLOGY")
        os.environ["AI_NOISE_NO_MORPHOLOGY"] = "1"
        try:
            teacher = mt.get_teacher(refresh=True)
            self.assertEqual(teacher.name, "none")
            self.assertIsNone(mt.analyse("犬が走る。"))
        finally:
            if prev is None:
                os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
            else:
                os.environ["AI_NOISE_NO_MORPHOLOGY"] = prev
            mt.get_teacher(refresh=True)

    def test_analysis_is_a_proposal_not_evidence(self):
        analysis = mt.MorphAnalysis("test", [mt.Morpheme("犬", "名詞", "一般", "犬")])
        self.assertEqual(analysis.evidence_score, 0.0)
        self.assertFalse(analysis.verified)

    def test_refresh_discards_results_cached_by_the_previous_backend(self):
        previous = os.environ.get("AI_NOISE_NO_MORPHOLOGY")
        os.environ["AI_NOISE_NO_MORPHOLOGY"] = "1"
        try:
            mt.get_teacher(refresh=True)
            self.assertIsNone(mt.analyse("キャッシュ更新確認"))
            self.assertIn("キャッシュ更新確認", mt._ANALYSE_CACHE)
        finally:
            if previous is None:
                os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
            else:
                os.environ["AI_NOISE_NO_MORPHOLOGY"] = previous
            mt.get_teacher(refresh=True)
        self.assertNotIn("キャッシュ更新確認", mt._ANALYSE_CACHE)


def _teacher_available() -> bool:
    prev = os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
    try:
        return mt.get_teacher(refresh=True).name != "none"
    finally:
        if prev is not None:
            os.environ["AI_NOISE_NO_MORPHOLOGY"] = prev
        mt.get_teacher(refresh=True)


@unittest.skipUnless(_teacher_available(), "no morphological analyser installed/vendored")
class BackendTest(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.pop("AI_NOISE_NO_MORPHOLOGY", None)
        self.teacher = mt.get_teacher(refresh=True)

    def tearDown(self):
        if self._prev is not None:
            os.environ["AI_NOISE_NO_MORPHOLOGY"] = self._prev
        mt.get_teacher(refresh=True)

    def test_segments_a_relative_clause_and_tags_the_case_particle(self):
        a = self.teacher.analyse("あそびまわっていたこうもりがつかまった。")
        surfaces = [m.surface for m in a.morphemes]
        self.assertIn("こうもり", surfaces)
        ga = [m for m in a.morphemes if m.surface == "が"]
        self.assertTrue(ga and ga[0].is_case_particle)

    def test_gives_a_verb_its_dictionary_form(self):
        a = self.teacher.analyse("きつねがぶどうを見つけました。")
        self.assertEqual(a.dictionary_form("見つけ"), "見つける")

    def test_refinement_cleans_a_fragment_subject_and_a_broken_verb(self):
        import japanese_event_v1 as jevent
        events = jevent.extract_story(
            "あるとき、あそびまわっていたこうもりが、つかまってしまいました。",
            use_teacher=True)
        self.assertTrue(all(e.subject == "こうもり" for e in events))
        self.assertIn("つかまる", [e.verb for e in events])


if __name__ == "__main__":
    unittest.main()
