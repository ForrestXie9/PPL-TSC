"""Compile LLM-generated symbolic expressions and fit their numerical coefficients."""
import re
import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class ExpressionCompiler:
    """Convert symbolic expressions into executable prediction functions."""

    def __init__(self, feature_names: List[str]):
        """Initialize the compiler with the ordered feature names."""
        self.feature_names = feature_names
        self.num_features = len(feature_names)

    def _normalize_expression(self, expr: str) -> str:
        """Convert supported non-Python expression syntax into Python syntax."""
        normalized = expr


        #    -> (value_if_true if condition else value_if_false)

        def replace_ternary_operator(text):
            """Replace supported ternary operators in expression text."""
            result = text
            max_iterations = 20  # Bound retries to avoid an infinite replacement loop.
            iteration = 0

            while '?' in result and ':' in result and iteration < max_iterations:
                iteration += 1

                q_positions = [i for i, char in enumerate(result) if char == '?']

                if not q_positions:
                    break

                replaced = False
                # Process right to left so earlier offsets remain valid.
                for q_pos in reversed(q_positions):

                    colon_pos = result.find(':', q_pos)
                    if colon_pos == -1:
                        continue



                    paren_start = result.rfind('(', 0, q_pos)
                    if paren_start == -1:
                        continue


                    paren_end = result.find(')', colon_pos)
                    if paren_end == -1:
                        continue


                    # Require paren_start < q_pos < colon_pos < paren_end.
                    if not (paren_start < q_pos < colon_pos < paren_end):
                        continue


                    condition = result[paren_start + 1:q_pos].strip()
                    true_val = result[q_pos + 1:colon_pos].strip()
                    false_val = result[colon_pos + 1:paren_end].strip()


                    if (condition and true_val and false_val and
                        '?' not in condition and '?' not in true_val and '?' not in false_val and
                        ':' not in condition and ':' not in true_val):

                        python_expr = f'({true_val} if {condition} else {false_val})'
                        result = result[:paren_start] + python_expr + result[paren_end + 1:]
                        replaced = True
                        break  # Restart matching after each replacement.

                if not replaced:

                    break

            return result


        normalized = replace_ternary_operator(normalized)



        if '?' in normalized:


            ternary_pattern = r'\(([^?]+)\s*\?\s*([^:]+)\s*:\s*([^)]+)\)'

            def regex_replace_ternary(match):
                condition = match.group(1).strip()
                true_val = match.group(2).strip()
                false_val = match.group(3).strip()
                return f'({true_val} if {condition} else {false_val})'

            normalized = re.sub(ternary_pattern, regex_replace_ternary, normalized)



        if '?' in normalized:
            # Legacy approximation for a remaining equality-based ternary.
            eq_ternary = r'\(x(\d+)\s*==\s*(\d+)\s*\?\s*1\s*:\s*0\)'
            def eq_replace(match):
                var_idx = match.group(1)
                target_val = match.group(2)

                return f'np.exp(-np.abs(X[:, {var_idx}] - {target_val}))'
            normalized = re.sub(eq_ternary, eq_replace, normalized)

            # Legacy fallback removes a remaining unsupported ternary term.
            if '?' in normalized:
                print(f"  Warning: Could not fully convert ternary operators in: {expr}")
                print(f"  Partially converted: {normalized}")

                normalized = re.sub(r'[+\-]\s*[^+\-]*\?[^+\-]*', '', normalized)

                if not normalized.strip() or normalized.strip() in ['+', '-', '*', '/']:
                    normalized = 'a1 * X[:, 0]'
                    print(f"  Fallback to simple expression: {normalized}")


        normalized = re.sub(r'\b&&\b', 'and', normalized)
        normalized = re.sub(r'\b\|\|\b', 'or', normalized)
        normalized = re.sub(r'!([^=])', r'not \1', normalized)  # Convert logical negation while preserving !=.







        normalized = re.sub(r'\^', '**', normalized)





        return normalized

    def compile_expression(self, expr: str) -> Tuple[Callable, List[str]]:
        """Compile an expression into f(X, *coeffs) and return its ordered coefficient names."""

        expr = self._normalize_expression(expr)


        coeff_pattern = r'\ba(\d+)\b'
        coeff_matches = re.findall(coeff_pattern, expr)
        coeff_names = sorted(set([f'a{num}' for num in coeff_matches]),
                           key=lambda x: int(x[1:]))

        if not coeff_names:
            raise ValueError(f"No coefficients found in expression: {expr}")



        compiled_expr = expr


        # Replace longer variable indices first so x10 does not match x1.
        for i in range(self.num_features - 1, -1, -1):
            compiled_expr = re.sub(rf'\bx{i}\b', f'X[:, {i}]', compiled_expr)


        for idx, coeff_name in enumerate(coeff_names):
            compiled_expr = re.sub(rf'\b{coeff_name}\b', f'coeffs[{idx}]', compiled_expr)


        math_replacements = {
            'sqrt': 'np.sqrt',
            'log': 'np.log',
            'exp': 'np.exp',
            'abs': 'np.abs',
            'max': 'np.maximum',
            'min': 'np.minimum',
        }

        for old, new in math_replacements.items():
            compiled_expr = re.sub(rf'\b{old}\s*\(', f'{new}(', compiled_expr)


        # Escape braces in generated f-strings or use string concatenation.
        func_code = f"""
def compiled_func(X, *coeffs):
    import numpy as np
    try:
        result = {compiled_expr}
        # Ensure a one-dimensional result.
        if isinstance(result, (int, float)):
            result = np.full(X.shape[0], result)
        # Convert to a one-dimensional array.
        result = np.asarray(result).flatten()
        # Check the sample count.
        if len(result) != X.shape[0]:
            if len(result.shape) > 1:
                result = result.flatten()
            if len(result) != X.shape[0]:
                # Broadcast a scalar result if necessary.
                if len(result) == 1:
                    result = np.full(X.shape[0], result[0])
                else:
                    # Use concatenation to avoid nested f-string parsing.
                    result_len = len(result)
                    x_shape_0 = X.shape[0]
                    raise ValueError("Result length " + str(result_len) + " doesn't match X.shape[0] " + str(x_shape_0))
        return result
    except Exception as e:
        return np.full(X.shape[0], np.nan)
"""

        namespace = {'np': np}
        try:
            exec(func_code, namespace)
            compiled_func = namespace['compiled_func']
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression syntax: {expr}\n"
                            f"Compiled expression: {compiled_expr}\n"
                            f"Error: {e}") from e

        return compiled_func, coeff_names

    def fit_coefficients(self,
                         expr: str,
                         X: np.ndarray,
                         y: np.ndarray,
                         initial_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Fit coefficients to X and y; return the fitted coefficients and mean squared error."""

        compiled_func, coeff_names = self.compile_expression(expr)
        num_coeffs = len(coeff_names)


        last_error = None


        def loss(coeffs):
            nonlocal last_error
            try:
                y_pred = compiled_func(X, *coeffs)

                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    last_error = "Expression produces NaN or Inf values"
                    return 1e10
                mse = mean_squared_error(y, y_pred)
                return mse
            except Exception as e:
                last_error = str(e)
                return 1e10


        if initial_coeffs is None:
            initial_coeffs = np.random.randn(num_coeffs) * 0.1


        test_mse = loss(initial_coeffs)
        if test_mse >= 1e10:
            print(f"  Warning: Expression evaluation failed with initial coeffs")
            if last_error:
                print(f"    Error: {last_error}")
            print(f"    Expression: {expr}")


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


            if final_mse >= 1e10:
                print(f"  Warning: Final MSE is extremely high ({final_mse})")
                if last_error:
                    print(f"    Last error: {last_error}")
                print(f"    This may indicate the expression cannot be properly evaluated")

        except Exception as e:
            print(f"  Coefficient fitting failed: {e}, using random initialization")
            fitted_coeffs = initial_coeffs
            final_mse = loss(fitted_coeffs)

        return fitted_coeffs, final_mse

    def create_predictor(self, expr: str, coeffs: np.ndarray) -> Callable:
        """Bind fitted coefficients to an expression and return a prediction callable."""
        compiled_func, _ = self.compile_expression(expr)

        def predictor(X):
            return compiled_func(X, *coeffs)

        return predictor


class LLMExpressionModel:
    """Pickle-compatible model wrapping a symbolic expression and fitted coefficients."""

    def __init__(self, expr: str, coeffs: np.ndarray, compiler: ExpressionCompiler):
        """Store the expression, fitted coefficients and compiler feature names."""
        self.expr = expr
        self.coeffs = coeffs
        self.feature_names = compiler.feature_names  # Retain feature names to rebuild the predictor.
        self._predictor = None  # Build the predictor lazily.

    def _get_predictor(self):
        """Lazily create and cache the prediction callable."""
        if self._predictor is None:
            compiler = ExpressionCompiler(self.feature_names)
            self._predictor = compiler.create_predictor(self.expr, self.coeffs)
        return self._predictor

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict Q-values."""
        return self._get_predictor()(X)

    def __getstate__(self):
        """Exclude the local prediction callable from pickle state."""
        state = self.__dict__.copy()
        # Local callables are not pickle-compatible.
        state['_predictor'] = None
        return state

    def __setstate__(self, state):
        """Restore stored state; recreate the predictor lazily on its next use."""
        self.__dict__.update(state)
        # Rebuild on the first prediction after loading.

    def get_expression_with_coeffs(self) -> str:
        """Return the expression with numerical coefficient values substituted."""
        expr = self.expr
        coeff_names = re.findall(r'\ba(\d+)\b', expr)
        coeff_names = sorted(set([f'a{num}' for num in coeff_names]),
                           key=lambda x: int(x[1:]))

        for idx, coeff_name in enumerate(coeff_names):
            expr = re.sub(rf'\b{coeff_name}\b', f'{self.coeffs[idx]:.6f}', expr)

        return expr

