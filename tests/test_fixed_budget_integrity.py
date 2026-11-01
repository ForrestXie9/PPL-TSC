"""Reject silent changes to the registered search and calibration budgets."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from LLM_SR_DISTILLATION_method_v2.evolution import Evolution
from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator


class FixedBudgetIntegrityTests(unittest.TestCase):
    def test_nonfinite_scores_cannot_enter_either_cache(self):
        e = Evolution.__new__(Evolution)
        e._evaluation_cache = {}
        e._raw_evaluation_cache = {}
        for score in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                e._cache_evaluation('offspring', SimpleNamespace(score=score))
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                e._cache_raw_evaluation('code', {'average': {'score': score}}, None)
        self.assertEqual(e._evaluation_cache, {})
        self.assertEqual(e._raw_evaluation_cache, {})

    def test_incomplete_population_cannot_start_fixed_budget_evolution(self):
        for count in (0, 4, 6):
            e = Evolution.__new__(Evolution)
            e.population_size = 5
            e.time_budget_seconds = None
            e.population = [None] * count
            with self.assertRaisesRegex(RuntimeError, 'Incomplete initial population'):
                e._assert_full_population()
        e.population = [None] * 5
        e._assert_full_population()

    def test_explicit_time_budget_can_have_partial_population(self):
        e = Evolution.__new__(Evolution)
        e.population_size = 5
        e.time_budget_seconds = 1.0
        e.population = [None]
        e._assert_full_population()

    def test_invalid_calibration_configuration_fails_before_fallback(self):
        e = Evaluator.__new__(Evaluator)
        for bound in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'bias_bound'):
                e.fit_per_intersection_bias(None, bias_bound=bound)
        for iterations in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'max_iter'):
                e.fit_per_intersection_bias(None, max_iter=iterations)


if __name__ == '__main__':
    unittest.main()
