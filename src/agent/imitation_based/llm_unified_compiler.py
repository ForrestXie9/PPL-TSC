"""Compile and fit unified symbolic templates with a phase_idx placeholder."""
import re
import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from agent.imitation_based.llm_expression_compiler import ExpressionCompiler


class UnifiedExpressionCompiler(ExpressionCompiler):
    """Expression compiler supporting the phase_idx placeholder."""

    def compile_unified_expression(self, expr_template: str, phase_idx: int) -> Tuple[Callable, List[str]]:
        """Substitute the physical phase index and compile the resulting expression."""

        expr = expr_template.replace('phase_idx', str(phase_idx))


        return self.compile_expression(expr)

    def fit_coefficients_for_phase(self,
                                   expr_template: str,
                                   phase_idx: int,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   initial_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Fit a template for one phase and return its coefficients and mean squared error."""

        compiled_func, coeff_names = self.compile_unified_expression(expr_template, phase_idx)
        num_coeffs = len(coeff_names)


        def loss(coeffs):
            try:
                y_pred = compiled_func(X, *coeffs)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return 1e10
                mse = mean_squared_error(y, y_pred)
                return mse
            except Exception:
                return 1e10


        if initial_coeffs is None:
            initial_coeffs = np.random.randn(num_coeffs) * 0.1


        try:
            result = minimize(
                loss,
                initial_coeffs,
                method='BFGS',
                options={'maxiter': 1000}
            )

            if result.success:
                fitted_coeffs = result.x
                final_mse = loss(fitted_coeffs)
            else:
                # Try L-BFGS-B if BFGS fails.
                result = minimize(
                    loss,
                    initial_coeffs,
                    method='L-BFGS-B',
                    bounds=[(-10, 10)] * num_coeffs,
                    options={'maxiter': 1000}
                )
                fitted_coeffs = result.x
                final_mse = loss(fitted_coeffs)
        except Exception as e:
            print(f"  Coefficient fitting failed for phase {phase_idx}: {e}, using random initialization")
            fitted_coeffs = initial_coeffs
            final_mse = loss(fitted_coeffs)

        return fitted_coeffs, final_mse

    def create_phase_predictor(self, expr_template: str, phase_idx: int, coeffs: np.ndarray) -> Callable:
        """Create a prediction callable for one phase and its fitted coefficients."""
        compiled_func, _ = self.compile_unified_expression(expr_template, phase_idx)

        def predictor(X):
            return compiled_func(X, *coeffs)

        return predictor


class UnifiedLLMExpressionModel:
    """Pickle-compatible unified template with a separate coefficient vector for each phase."""

    def __init__(self, expr_template: str, phase_coeffs: List[np.ndarray], compiler: UnifiedExpressionCompiler):
        """Store the expression template, per-phase coefficients and compiler feature names."""
        self.expr_template = expr_template
        self.phase_coeffs = phase_coeffs
        self.feature_names = compiler.feature_names  # Retain feature names to rebuild predictors.
        self.num_phase = len(phase_coeffs)
        self._predictors = None  # Build predictors lazily.

    def _get_predictors(self):
        """Lazily create and cache the per-phase prediction callables."""
        if self._predictors is None:
            compiler = UnifiedExpressionCompiler(self.feature_names)
            self._predictors = []
            for phase_idx in range(self.num_phase):
                predictor = compiler.create_phase_predictor(
                    self.expr_template, phase_idx, self.phase_coeffs[phase_idx]
                )
                self._predictors.append(predictor)
        return self._predictors

    def __getstate__(self):
        """Exclude prediction callables from pickle state."""
        state = self.__dict__.copy()
        state['_predictors'] = None
        return state

    def __setstate__(self, state):
        """Restore state and defer rebuilding prediction callables until prediction."""
        self.__dict__.update(state)
        # Rebuild on the first prediction after loading.

    def predict(self, X: np.ndarray, phase_idx: Optional[int] = None) -> np.ndarray:
        """Predict one phase as (n_samples,) or all phases as (n_samples, num_phase)."""
        predictors = self._get_predictors()
        if phase_idx is not None:
            return predictors[phase_idx](X)
        else:

            predictions = [pred(X) for pred in predictors]
            return np.column_stack(predictions)

    def get_expression_for_phase(self, phase_idx: int) -> str:
        """Return a phase expression with its coefficient values substituted."""
        expr = self.expr_template.replace('phase_idx', str(phase_idx))


        coeff_names = re.findall(r'\ba(\d+)\b', expr)
        coeff_names = sorted(set([f'a{num}' for num in coeff_names]),
                           key=lambda x: int(x[1:]))


        for idx, coeff_name in enumerate(coeff_names):
            expr = re.sub(rf'\b{coeff_name}\b', f'{self.phase_coeffs[phase_idx][idx]:.6f}', expr)

        return expr

    def get_all_expressions(self) -> List[str]:
        """Return the substituted expressions for all phases."""
        return [self.get_expression_for_phase(i) for i in range(self.num_phase)]

