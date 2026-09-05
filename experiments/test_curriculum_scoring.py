import unittest

from curriculum_scoring import curriculum_strategy_allowed, learned_curriculum_score


class CurriculumScoringTest(unittest.TestCase):
    def test_deprioritized_strategy_is_not_allowed_but_others_still_are(self):
        curriculum = {"strategy_performance": {
            "broad shelf": {"status": "deprioritized"},
            "targeted counterexample": {"status": "active"}}}
        self.assertFalse(curriculum_strategy_allowed(curriculum, {"reason": "broad shelf"}))
        self.assertTrue(curriculum_strategy_allowed(
            curriculum, {"reason": "targeted counterexample"}))
        # A reason never seen before has no recorded performance yet.
        self.assertTrue(curriculum_strategy_allowed(curriculum, {"reason": "brand new route"}))

    def test_a_route_with_a_higher_admission_rate_scores_higher_at_equal_base_score(self):
        curriculum = {"strategy_performance": {
            "good": {"admitted": 8, "rejected": 2},
            "bad": {"admitted": 1, "rejected": 9}}}
        self.assertGreater(learned_curriculum_score(curriculum, {"reason": "good", "score": 2}),
                           learned_curriculum_score(curriculum, {"reason": "bad", "score": 2}))

    def test_score_never_flips_the_relative_order_of_two_base_scores_at_equal_yield(self):
        curriculum = {"strategy_performance": {
            "route_a": {"admitted": 3, "rejected": 3}, "route_b": {"admitted": 3, "rejected": 3}}}
        low = learned_curriculum_score(curriculum, {"reason": "route_a", "score": 2.5})
        high = learned_curriculum_score(curriculum, {"reason": "route_b", "score": 6.0})
        self.assertLess(low, high)

    def test_an_unseen_route_gets_a_neutral_beta_prior_not_zero(self):
        # (0 admitted + 1) / (0 + 0 + 2) = 0.5 expected_yield -> multiplier 1.0.
        score = learned_curriculum_score({}, {"reason": "never tried", "score": 4.0})
        self.assertAlmostEqual(score, 4.0)


if __name__ == "__main__":
    unittest.main()
