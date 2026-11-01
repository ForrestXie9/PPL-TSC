from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from LLM_SR_DISTILLATION.numerical_safety import check_division_safety
from LLM_SR_DISTILLATION.constraint_validator import ConstraintValidator
from LLM_SR_DISTILLATION.function_utility import FunctionUtility


class NumericalSafetyTests(unittest.TestCase):
    def check(self, body):
        code = 'def compute_utility(state, phase_id=None, feature_stats=None):\n'
        code += '    d = state.get("pressure", 0.0)\n'
        code += '\n'.join('    ' + line for line in body.splitlines())
        return check_division_safety(code)['valid']

    def test_signed_offsets_and_zero_named_epsilon_are_rejected(self):
        for body in ('return 1 / (d + 1)', 'return 1 / (d + 1e-8)',
                     'epsilon = 0\nreturn 1 / epsilon', 'return 1 / (1 + d + 3)',
                     'return 1 / abs(d)', 'return 1 / (d*d)'):
            with self.subTest(body=body):
                self.assertFalse(self.check(body))

    def test_protected_signed_and_magnitude_ratios_are_accepted(self):
        for body in ('return 1 / max(abs(d), 1e-8)',
                     'return 1 / (abs(d) + 1e-8)',
                     'denom = max(abs(d), 1e-8)\nreturn 1 / denom',
                     'return 1 / d if abs(d) > 1e-8 else 0',
                     'if d != 0:\n    return 1 / d\nreturn 0',
                     'if d == 0:\n    return 0\nelse:\n    return 1 / d'):
            with self.subTest(body=body):
                self.assertTrue(self.check(body))

    def test_guards_do_not_leak_outside_branches_or_across_assignments(self):
        for body in ('if d != 0:\n    pass\nreturn 1/d',
                     'if d != 0:\n    d = 0\n    return 1/d',
                     'if d != 0 or phase_id == 0:\n    return 1/d',
                     'if d != 0:\n    return 1 / (d * 0)',
                     'denom = max(abs(d), 1e-8)\ndenom = 0\nreturn 1/denom'):
            with self.subTest(body=body):
                self.assertFalse(self.check(body))

    def test_guard_in_one_function_does_not_protect_another(self):
        code = 'def other(d):\n    if d != 0:\n        return 1/d\ndef compute_utility(d):\n    return 1/d\n'
        self.assertFalse(check_division_safety(code)['valid'])

    def test_augmented_and_numpy_division_are_checked(self):
        self.assertFalse(self.check('q = 1\nq /= d\nreturn q'))
        self.assertFalse(self.check('import numpy as np\nreturn np.divide(1., d)'))
        self.assertFalse(self.check('return d ** -1'))

    def test_numpy_invalid_intermediates_cannot_be_hidden_by_final_finite_fallback(self):
        for expression in ('np.divide(1.0, 0.0)', 'np.log(-1.0)', 'np.exp(1000.0)'):
            with self.subTest(expression=expression):
                uf = FunctionUtility('def compute_utility(state, phase_id=None, feature_stats=None):\n'
                                     f'    import numpy as np\n    q = {expression}\n'
                                     '    return float(q) if np.isfinite(q) else 0.0\n')
                try:
                    with self.assertRaisesRegex(ValueError, 'FloatingPointError'):
                        uf.compute({})
                finally:
                    uf._compiled_function.close()

    def test_numpy_inverse_and_remainder_need_nonzero_guards(self):
        for expr in ('np.reciprocal(d)', 'numpy.reciprocal(d)', 'np.power(d, -1)',
                     'np.float_power(d, -2)', 'pow(d, -1)', 'np.floor_divide(1., d)',
                     'np.remainder(1., d)', 'numpy.mod(1., d)', 'np.fmod(1., d)'):
            with self.subTest(expr=expr):
                self.assertFalse(self.check('return ' + expr))
                self.assertTrue(self.check('if abs(d) > 1e-8:\n    return ' + expr + '\nreturn 0.0'))
        self.assertFalse(self.check('d **= -1\nreturn d'))
        self.assertTrue(self.check('d = max(abs(d), 1e-8)\nd **= -1\nreturn d'))
        self.assertTrue(self.check('return np.power(d, 2)'))

    def validator(self):
        keys = ['pressure', 'num_waiting_vehicle', 'sum_waiting_time']
        state = {'current_phase': 1, 'controlled_directions': ['a', 'b'],
                 'features_by_direction': {d: {k: 1. for k in keys} for d in ['a', 'b']}}
        loader = SimpleNamespace(intersection_ids=[0], get_action_consistency_data=lambda *a, **k: [({0: state}, 0)])
        return ConstraintValidator('TinyLight', [], '', data_loader=loader,
                actual_feature_keys=keys, feature_stats={k: {'min': 0., 'max': 10.} for k in keys})

    def test_signed_boundaries_find_real_hysteresis_defect_even_with_positive_source_stats(self):
        code = '''def compute_utility(state, phase_id=None, feature_stats=None):
    q = sum(state['features_by_direction'][d]['num_waiting_vehicle'] for d in state['controlled_directions'])
    p = sum(state['features_by_direction'][d]['pressure'] for d in state['controlled_directions'])
    result = 0.7 * p
    if state.get('current_phase') == phase_id:
        result += 0.5 * (q + p) / (1.0 + q + p)
    return result
'''
        uf = FunctionUtility(code)
        try:
            valid, errors = self.validator()._check_boundary_extreme_function_utility(uf, max_samples=1)
            self.assertFalse(valid)
            self.assertTrue(any('ZeroDivisionError' in e and 'active=True' in e for e in errors))
        finally:
            uf._compiled_function.close()

    def test_default_review_executes_runtime_checks_and_reports_failure(self):
        validator = self.validator()
        with patch.object(validator, 'check_feature_key_constraints', return_value={'valid': True}), \
             patch.object(validator, '_validate_on_real_states', return_value=(False, ['boundary overflow'])) as gate:
            result = validator.validate_hybrid('def compute_utility(state):\n    return 1.0\n',
                                              SimpleNamespace(compute=lambda: 1.))
        gate.assert_called_once()
        self.assertFalse(result['valid'])
        self.assertIn('boundary overflow', result['violations'])

    def test_single_pass_review_cannot_disable_numerical_gate(self):
        from LLM_SR_DISTILLATION_method_v2.evolution import Evolution
        evo = Evolution.__new__(Evolution)
        evo.evaluator = SimpleNamespace()
        evo.feature_stats = {}
        evo.action_mode = 'phase_argmax'
        evo.use_constraint_validation = False
        evo.constraint_validator = self.validator()
        with patch.object(evo.constraint_validator, 'check_feature_key_constraints', return_value={'valid': True}):
            with self.assertRaisesRegex(RuntimeError, 'Numerical safety'):
                evo._compile_and_review_candidate('def compute_utility(state, phase_id=None, feature_stats=None):\n    return 1 / (state.get("pressure", 0) + 1)\n',
                                                   verbose=False, mutation_phase='initialization')


if __name__ == '__main__':
    unittest.main()
