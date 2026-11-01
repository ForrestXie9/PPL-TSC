"""
Expression Validator Tests Module
Validate utility function expressions using various test methods
Not dependent on training data, but uses systematic testing approaches
"""

import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Callable
import traceback


class ExpressionValidatorTests:
    """
    Validate expressions through systematic testing
    """

    def __init__(self,
                 feature_ranges: Dict[str, Tuple[float, float]],
                 feature_kinds: Optional[Dict[str, str]] = None,
                 epsilon: float = 0.001):
        """
        Initialize validator

        Args:
            feature_ranges: Dict {feature_name: (min_val, max_val)}
            epsilon: Small perturbation for sensitivity testing
        """
        self.feature_ranges = feature_ranges
        self.feature_kinds = feature_kinds or {}
        self.epsilon = epsilon

    def _range_for_feature(self, feature: str) -> Tuple[float, float]:
        """
        Decide feature range based on its semantic kind.
        Some kinds are non-negative; others may be signed.
        """
        min_val, max_val = self.feature_ranges[feature]
        kind = (self.feature_kinds.get(feature) or "").lower()

        non_negative_kinds = {
            "current_phase",
            "num_vehicle_sum",
            "num_waiting_vehicle_sum",
            "waiting_time_sum",
        }

        if kind == "current_phase":
            return 0.0, 1.0

        if kind in non_negative_kinds or kind.startswith("other:"):
            min_val = max(0.0, float(min_val))
            max_val = float(max_val)
            if max_val < 0.0:
                max_val = 0.0
            if max_val < min_val:
                max_val = min_val
            return min_val, max_val

        # Signed kinds (e.g., pressure_sum) keep original range.
        return float(min_val), float(max_val)

    def _is_discrete_feature(self, feature: str) -> bool:
        """Return whether the feature is discrete."""
        kind = (self.feature_kinds.get(feature) or "").lower()
        return kind == "current_phase"

    def _get_discrete_values(self, feature: str) -> List[float]:
        """Return the possible values of a discrete feature."""
        if self._is_discrete_feature(feature):
            return [0.0, 1.0]  # The current-phase indicator is binary.
        return []

    def run_all_tests(self,
                     utility_function,
                     parameters: Dict[str, float],
                     verbose: bool = False) -> Dict[str, Any]:
        """
        Run all validation tests

        Args:
            utility_function: UtilityFunction instance
            parameters: Parameter dictionary
            verbose: Print detailed results

        Returns:
            Test results dictionary
        """
        test_results = {}

        try:
            test_results['monotonicity'] = self.test_monotonicity(
                utility_function, parameters
            )
        except Exception as e:
            test_results['monotonicity'] = {
                'passed': False,
                'violations': [{'type': 'test_error', 'error': str(e)}]
            }

        try:
            test_results['boundary'] = self.test_boundary_cases(
                utility_function, parameters
            )
        except Exception as e:
            test_results['boundary'] = {
                'passed': False,
                'violations': [{'type': 'test_error', 'error': str(e)}]
            }

        try:
            test_results['extreme'] = self.test_extreme_values(
                utility_function, parameters
            )
        except Exception as e:
            test_results['extreme'] = {
                'passed': False,
                'violations': [{'type': 'test_error', 'error': str(e)}]
            }

        # Summarize results
        all_passed = all(result['passed'] for result in test_results.values())
        total_violations = sum(
            len(result.get('violations', []))
            for result in test_results.values()
        )

        if verbose:
            self._print_test_summary(test_results, all_passed, total_violations)

        return {
            'valid': all_passed,
            'test_results': test_results,
            'total_violations': total_violations,
            'message': 'All tests passed' if all_passed else f'{total_violations} test violations found'
        }

    def test_monotonicity(self,
                         utility_function,
                         parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Test: Each feature should monotonically increase Q value

        For traffic signals:
        - pressure ↑ → Q should ↑
        - num_vehicles ↑ → Q should ↑
        - waiting_time ↑ → Q should ↑
        """
        violations = []

        # Create baseline test point (middle of ranges)
        base_point = {}
        for f in self.feature_ranges.keys():
            if self._is_discrete_feature(f):

                base_point[f] = 0.0
            else:

                min_val, max_val = self._range_for_feature(f)
                base_point[f] = (min_val + max_val) / 2

        try:
            Q_base = utility_function.compute(base_point, parameters)
        except:
            return {
                'passed': False,
                'violations': [{'type': 'baseline_evaluation_error'}]
            }

        if not np.isfinite(Q_base):
            return {
                'passed': False,
                'violations': [{'type': 'baseline_q_invalid', 'q_value': Q_base}]
            }

        # Discrete feature monotonicity: phase_active=1 should not reduce Q
        for feature in self.feature_ranges.keys():
            if not self._is_discrete_feature(feature):
                continue

            test_point_0 = base_point.copy()
            test_point_1 = base_point.copy()
            test_point_0[feature] = 0.0
            test_point_1[feature] = 1.0

            try:
                Q0 = utility_function.compute(test_point_0, parameters)
                Q1 = utility_function.compute(test_point_1, parameters)
            except:
                violations.append({
                    'type': 'monotonicity_evaluation_error',
                    'feature': feature,
                    'error': 'Failed to compute Q value for discrete feature'
                })
                continue

            if not np.isfinite(Q0) or not np.isfinite(Q1):
                violations.append({
                    'type': 'monotonicity_invalid_output',
                    'feature': feature,
                    'q_base': Q0,
                    'q_test': Q1
                })
                continue

            if (Q1 - Q0) < -0.01:
                violations.append({
                    'type': 'monotonicity_violation',
                    'feature': feature,
                    'q_base': Q0,
                    'q_test': Q1,
                    'q_change': Q1 - Q0,
                    'message': f'Discrete feature {feature} switched to 1 but Q decreased by {abs(Q1 - Q0):.4f}'
                })

        # Test each feature
        for feature in self.feature_ranges.keys():
            # Skip continuous monotonicity check for discrete features
            if self._is_discrete_feature(feature):
                continue

            test_point = base_point.copy()

            # Continuous feature: test at max value (90%)
            min_val, max_val = self._range_for_feature(feature)
            test_point[feature] = max_val * 0.9 if max_val > 0 else max_val

            try:
                Q_test = utility_function.compute(test_point, parameters)
            except:
                violations.append({
                    'type': 'monotonicity_evaluation_error',
                    'feature': feature,
                    'error': 'Failed to compute Q value'
                })
                continue

            if not np.isfinite(Q_test):
                violations.append({
                    'type': 'monotonicity_invalid_output',
                    'feature': feature,
                    'q_value': Q_test
                })
                continue

            # Check monotonicity: Q should increase or stay same
            q_change = Q_test - Q_base
            if q_change < -0.01:  # Allow small floating point error
                violations.append({
                    'type': 'monotonicity_violation',
                    'feature': feature,
                    'q_base': Q_base,
                    'q_test': Q_test,
                    'q_change': q_change,
                    'message': f'Feature {feature} increased but Q decreased by {abs(q_change):.4f}'
                })

        return {
            'passed': len(violations) == 0,
            'violations': violations
        }

    def test_boundary_cases(self,
                           utility_function,
                           parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Test: Behavior at boundary conditions
        """
        violations = []

        def _clamp(val: float, lo: float, hi: float) -> float:
            return float(min(max(val, lo), hi))

        # Generate test cases
        test_cases = [
            ('all_zero', {f: _clamp(0.0, *self._range_for_feature(f)) for f in self.feature_ranges.keys()}),
            ('all_min', {f: self._range_for_feature(f)[0] for f in self.feature_ranges.keys()}),
            ('all_max', {f: self._range_for_feature(f)[1] for f in self.feature_ranges.keys()}),
            ('all_mid', {f: _clamp((self._range_for_feature(f)[0] + self._range_for_feature(f)[1]) / 2,
                                   *self._range_for_feature(f)) for f in self.feature_ranges.keys()}),
        ]

        # Add mixed extreme cases
        feature_list = list(self.feature_ranges.keys())
        for i, feature in enumerate(feature_list):
            case_dict = {}
            for j, f in enumerate(feature_list):
                if j == i:
                    case_dict[f] = self._range_for_feature(f)[1]  # max
                else:
                    case_dict[f] = self._range_for_feature(f)[0]  # min
            test_cases.append((f'mixed_extreme_{i}', case_dict))

        for case_name, test_point in test_cases:
            try:
                Q = utility_function.compute(test_point, parameters)

                # Check 1: Q value should be finite
                if not np.isfinite(Q):
                    violations.append({
                        'type': 'boundary_invalid_output',
                        'case': case_name,
                        'q_value': Q,
                        'message': f'Q value is NaN or Inf: {Q}'
                    })
                    continue

                # Only require finite Q in boundary cases

            except Exception as e:
                violations.append({
                    'type': 'boundary_evaluation_error',
                    'case': case_name,
                    'error': str(e),
                    'message': f'Failed to evaluate: {str(e)}'
                })

        return {
            'passed': len(violations) == 0,
            'violations': violations
        }


    def test_extreme_values(self,
                           utility_function,
                           parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Test: Behavior at extreme values
        All feature values are normalized to [0, 1] by UtilityFunction
        """
        violations = []

        def _clamp01(val: float) -> float:
            return float(min(max(val, 0.0), 1.0))

        # Generate extreme cases using normalized values [0, 1]
        extreme_cases = []
        feature_list = list(self.feature_ranges.keys())

        # All zeros
        all_zero = {f: (0.0 if self._is_discrete_feature(f) else _clamp01(0.0)) for f in feature_list}
        extreme_cases.append(('all_zero', all_zero))

        # Single feature at max (1.0), others at min (0.0)
        for feature in feature_list:
            case_dict = {}
            for f in feature_list:
                if self._is_discrete_feature(f):
                    case_dict[f] = 1.0 if f == feature else 0.0
                else:
                    case_dict[f] = _clamp01(1.0) if f == feature else _clamp01(0.0)
            extreme_cases.append((f'only_{feature}_max', case_dict))

        # All at max (1.0)
        all_max = {f: (1.0 if self._is_discrete_feature(f) else _clamp01(1.0)) for f in feature_list}
        extreme_cases.append(('all_max', all_max))

        for case_name, test_point in extreme_cases:
            try:
                Q = utility_function.compute(test_point, parameters)

                # Check 1: Should be finite (not NaN or Inf)
                if not np.isfinite(Q):
                    violations.append({
                        'type': 'extreme_invalid',
                        'case': case_name,
                        'q_value': Q
                    })
                    continue

                # Only require finite Q in extreme cases

            except Exception as e:
                violations.append({
                    'type': 'extreme_error',
                    'case': case_name,
                    'error': str(e)
                })

        return {
            'passed': len(violations) == 0,
            'violations': violations
        }

    def _print_test_summary(self, test_results: Dict, all_passed: bool, total_violations: int):
        """Print summary of test results"""
        print("\n" + "="*60)
        print("Expression Validation Test Results")
        print("="*60)

        for test_name, result in test_results.items():
            status = "[PASS]" if result['passed'] else "[FAIL]"
            n_violations = len(result.get('violations', []))
            print(f"\n{test_name.upper():20} {status}")

            if not result['passed'] and n_violations > 0:
                for i, violation in enumerate(result['violations'][:3]):  # Show first 3
                    print(f"  - {violation.get('type', 'unknown')}")
                    if 'message' in violation:
                        print(f"    {violation['message']}")
                if n_violations > 3:
                    print(f"  ... and {n_violations - 3} more violations")

        print("\n" + "="*60)
        overall = "[PASS] ALL TESTS PASSED" if all_passed else f"[FAIL] {total_violations} VIOLATIONS FOUND"
        print(f"Overall: {overall}")
        print("="*60 + "\n")
