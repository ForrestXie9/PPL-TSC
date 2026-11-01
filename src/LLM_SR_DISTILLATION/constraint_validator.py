"""
约束验证模块
验证生成的效用函数是否满足边界条件和变化趋势约束
支持两种验证方式：
1. 基于训练数据的约束检查
2. 基于系统测试的表达式验证
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import ast
import copy
import os
from utils.constraint_checker import check_generated_function_for_teacher_model
from LLM_SR_DISTILLATION.expression_validator_tests import ExpressionValidatorTests
from LLM_SR_DISTILLATION.proved_numeric_bounds import DivisionCheck, check_division_safety
from LLM_SR_DISTILLATION.numerical_safety import signed_boundary_states
from LLM_SR_DISTILLATION.semantic_checks import phase_monotonicity, binary_monotonicity, demand_discrimination
from LLM_SR_DISTILLATION.phase_label_checks import phase_label_consistency
from LLM_SR_DISTILLATION.direction_order_checks import direction_order_consistency


class ConstraintValidator:
    """约束验证器 - 支持多种验证方式"""
    
    def __init__(
        self,
        teacher_model: str,
        observation_feature_list: List[str],
        json_file_path: str,
        intersection_id: Optional[int] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        feature_kinds: Optional[Dict[str, str]] = None,
        actual_feature_keys: Optional[List[str]] = None,
        data_loader: Optional[Any] = None,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        action_mode: str = "phase_argmax",
        binary_min_margin_span: float = 0.0,
    ):
        """
        初始化约束验证器
        
        Args:
            teacher_model: teacher模型名称（如 'MPLight', 'FRAP'等）
            observation_feature_list: 观察特征列表
            json_file_path: teacher数据JSON文件路径
            intersection_id: 如果指定，只检查该路口的数据
            feature_ranges: 特征范围字典 {feature_name: (min, max)}，用于测试验证
            feature_kinds: 特征类型标记字典
            actual_feature_keys: 实际存在于数据中的feature keys（从feature_stats提取）
            data_loader: TeacherDataLoader，用于真实state采样验证（FunctionUtility）
            feature_stats: 特征统计 {name: {min,max,mean,std}}，用于 compute(state, phase_id, feature_stats)
        """
        self.teacher_model = teacher_model
        self.observation_feature_list = observation_feature_list
        self.json_file_path = json_file_path
        self.intersection_id = intersection_id
        self.feature_ranges = feature_ranges or self._build_default_feature_ranges()
        self.feature_kinds = feature_kinds or self._build_default_feature_kinds()
        self.actual_feature_keys = list(actual_feature_keys or [])
        self.data_loader = data_loader
        self.feature_stats = feature_stats or {}
        self.action_mode = str(action_mode or "phase_argmax")
        self.binary_min_margin_span = float(binary_min_margin_span)
        
        # 初始化测试验证器
        self.test_validator = ExpressionValidatorTests(
            self.feature_ranges,
            feature_kinds=self.feature_kinds
        )

    @staticmethod
    def _call_name(func_node: ast.AST) -> str:
        """Return dotted call name, e.g., np.sqrt / max / abs."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            parts = []
            cur = func_node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""

    @staticmethod
    def _literal_float(node: ast.AST) -> Optional[float]:
        """Extract float literal from AST node if possible."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                return -float(node.operand.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = ConstraintValidator._literal_float(node.left)
            right = ConstraintValidator._literal_float(node.right)
            if left is not None and right is not None and abs(right) > 1e-15:
                return left / right
        return None

    def _is_fractional_exponent(self, node: ast.AST) -> bool:
        """True if exponent is fractional (0.5, 1/2, etc.) - negative base would produce complex."""
        val = self._literal_float(node)
        if val is None:
            return False
        return val > 0 and abs(val - int(val)) > 1e-12

    def _has_guarded_lower_bound(self, expr: ast.AST, strictly_positive: bool = False) -> bool:
        """Prove an explicit lower bound without assuming traffic is unsigned."""
        lo, _ = DivisionCheck(ast.Module(body=[], type_ignores=[])).bounds(expr, {})
        return lo > 0 if strictly_positive else lo >= 0

    def _is_safe_denominator(self, denom: ast.AST) -> bool:
        checker = DivisionCheck(ast.Module(body=[], type_ignores=[]))
        return checker.nonzero(denom, {}, set())

    def _denominator_contains_guarded_var(
        self, denom: ast.AST, guarded_vars: set, safe_denom_vars: set
    ) -> bool:
        """Check if denominator expr contains a guarded variable (e.g. denom * len(x) where denom>1e-8)."""
        if isinstance(denom, ast.Name):
            return denom.id in guarded_vars or denom.id in safe_denom_vars
        if isinstance(denom, ast.BinOp) and isinstance(denom.op, (ast.Mult, ast.MatMult)):
            return (
                self._denominator_contains_guarded_var(denom.left, guarded_vars, safe_denom_vars)
                or self._denominator_contains_guarded_var(denom.right, guarded_vars, safe_denom_vars)
            )
        return False

    def _ifexp_guarantees_positive(self, expr: ast.IfExp) -> bool:
        """
        Detect ternary pattern that guarantees positive denominator:
        denom = A if A > c else k, where c>0 and k!=0.
        """
        # Else branch must be non-zero literal (e.g., 1.0)
        else_lit = self._literal_float(expr.orelse)
        if else_lit is None or abs(else_lit) <= 1e-15:
            return False

        # Test should be a simple compare: <body_expr> > positive_literal
        test = expr.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            return False

        op = test.ops[0]
        comp_lit = self._literal_float(test.comparators[0])
        if comp_lit is None or comp_lit <= 0:
            return False
        if not isinstance(op, (ast.Gt, ast.GtE)):
            return False

        # Body must be exactly the guarded expression in test.left
        return ast.dump(expr.body, include_attributes=False) == ast.dump(test.left, include_attributes=False)

    def _extract_guarded_positive_vars_from_test(self, test: ast.AST) -> set:
        """
        Extract variable names that are explicitly guarded as positive/non-zero in an if-test.
        Examples:
        - if std_p > 1e-6:
        - if abs(std_p) > 1e-6:
        - if std_p != 0:
        - if (std_p > 0) and (...):
        """
        vars_set = set()

        def _name_from_abs_call(node: ast.AST) -> Optional[str]:
            if not isinstance(node, ast.Call):
                return None
            name = self._call_name(node.func)
            if name not in {"abs", "np.abs", "numpy.abs"} or len(node.args) != 1:
                return None
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                return arg.id
            return None

        def _handle_compare(node: ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return
            op = node.ops[0]
            left = node.left
            right = node.comparators[0]
            r_lit = self._literal_float(right)
            l_lit = self._literal_float(left)

            # var > c / var >= c / var != 0  (var > 0 also guards as positive)
            if isinstance(left, ast.Name) and r_lit is not None:
                if isinstance(op, ast.Gt) and r_lit >= 0:
                    vars_set.add(left.id)  # var > 0 or var > 1e-8
                elif isinstance(op, ast.GtE) and r_lit > 0:
                    vars_set.add(left.id)
                elif isinstance(op, ast.NotEq) and abs(r_lit) <= 1e-15:
                    vars_set.add(left.id)

            # c < var / c <= var / 0 != var
            if isinstance(right, ast.Name) and l_lit is not None:
                if isinstance(op, ast.Lt) and l_lit <= 0:
                    vars_set.add(right.id)  # 0 < var means var > 0
                elif isinstance(op, (ast.Lt, ast.LtE)) and l_lit > 0:
                    vars_set.add(right.id)  # 1e-8 < var
                elif isinstance(op, ast.NotEq) and abs(l_lit) <= 1e-15:
                    vars_set.add(right.id)

            # abs(var) > c / abs(var) >= c / abs(var) != 0
            abs_var = _name_from_abs_call(left)
            if abs_var is not None and r_lit is not None:
                if isinstance(op, (ast.Gt, ast.GtE)) and r_lit > 0:
                    vars_set.add(abs_var)
                elif isinstance(op, ast.NotEq) and abs(r_lit) <= 1e-15:
                    vars_set.add(abs_var)

        def _walk(node: ast.AST):
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                for v in node.values:
                    _walk(v)
                return
            if isinstance(node, ast.Compare):
                _handle_compare(node)

        _walk(test)
        return vars_set

    def check_dangerous_operator_constraints(self, function_code: str) -> Dict[str, Any]:
        """
        Check for risky numeric operators without obvious guarding:
        - sqrt(x) where x may be negative
        - log(x) where x may be non-positive
        - division by denominator that may be zero
        """
        violations: List[str] = []
        seen = set()

        try:
            tree = ast.parse(function_code)
        except SyntaxError as e:
            return {
                "valid": False,
                "violations": [f"Cannot parse function for dangerous-op check: {e}"],
                "message": "Dangerous operator check failed (parse error)"
            }

        # Division now uses a separate, scope-aware check. The inherited
        # '+epsilon' and global-guard heuristics are not admission criteria.
        division_result = check_division_safety(function_code)
        violations.extend(division_result['violations'])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node.func)
                lineno = getattr(node, "lineno", -1)

                if name.endswith("sqrt") and node.args:
                    if not self._has_guarded_lower_bound(node.args[0], strictly_positive=False):
                        msg = (
                            f"Line {lineno}: sqrt(...) may receive negative input. "
                            f"Guard with clip/max/abs before sqrt."
                        )
                        if msg not in seen:
                            seen.add(msg)
                            violations.append(msg)

                if name.endswith("log") and not name.endswith("log1p") and node.args:
                    if not self._has_guarded_lower_bound(node.args[0], strictly_positive=True):
                        msg = (
                            f"Line {lineno}: log(...) may receive non-positive input. "
                            f"Guard with clip/max and epsilon."
                        )
                        if msg not in seen:
                            seen.add(msg)
                            violations.append(msg)

            # Power: base ** fractional_exponent with negative base produces complex
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                lineno = getattr(node, "lineno", -1)
                if self._is_fractional_exponent(node.right):
                    if not self._has_guarded_lower_bound(node.left, strictly_positive=False):
                        msg = (
                            f"Line {lineno}: base**fractional may receive negative base (produces complex). "
                            f"Use abs/max(0,...) before power."
                        )
                        if msg not in seen:
                            seen.add(msg)
                            violations.append(msg)

        valid = len(violations) == 0
        message = "Dangerous operator constraints satisfied" if valid else f"{len(violations)} dangerous operator risks detected"
        return {
            "valid": valid,
            "violations": violations,
            "message": message,
        }

    def check_numerical_safety(self, function_code: str) -> Dict[str, Any]:
        """Mandatory candidate gate, including when semantic review is disabled."""
        return check_division_safety(function_code)
    
    def _build_default_feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        构建默认的特征范围（原始范围，未归一化）
        实际归一化由 FeatureExtractor 对 phase_features 的 _norm 字段执行；
        未见目标样本可能超出源场景的 [0,1] 范围。
        """
        return {
            'phase_active': (0.0, 1.0),            # Current phase indicator (已是 0 或 1)
            'pressure_norm': (0.0, 500.0),         # 原始：Pressure or num_vehicles
            'vehicles_norm': (0.0, 500.0),         # 原始：Num_vehicles
            'waiting_veh_norm': (0.0, 1000.0),     # 原始：Num_waiting_vehicles
            'waiting_time_norm': (0.0, 5000.0),    # 原始：Waiting time
        }
    
    def _build_default_feature_kinds(self) -> Dict[str, str]:
        """构建默认的特征类型标记"""
        return {
            'phase_active': 'current_phase',         # 离散：只能是 0 或 1
            'pressure_norm': 'pressure_sum',
            'vehicles_norm': 'num_vehicle_sum',
            'waiting_veh_norm': 'num_waiting_vehicle_sum',
            'waiting_time_norm': 'waiting_time_sum',
        }
    
    def check_feature_key_constraints(self, function_code: str) -> Dict[str, Any]:
        """
        检查生成的函数代码中使用的feature keys是否都在actual_feature_keys中存在。
        这确保生成的函数不会尝试访问数据中不存在的keys。
        
        Args:
            function_code: 生成的Python函数代码
        
        Returns:
            {'valid': bool, 'violations': List[str], 'message': str}
        """
        violations = []
        
        # 如果没有实际的feature keys列表，跳过这个检查
        if not self.actual_feature_keys:
            return {'valid': True, 'violations': [], 'message': 'No actual_feature_keys provided for validation'}
        
        # Statistics and raw feature fields have different namespaces. Mask
        # only proven AST reads, preserving checks on raw fields with the same name.
        from LLM_SR_DISTILLATION.statistics_namespace import mask_statistic_keys
        try:
            function_code, statistics_violations, _ = mask_statistic_keys(function_code, self.feature_stats)
        except SyntaxError as exc:
            return {'valid': False, 'violations': [f'Invalid Python syntax: {exc}'],
                    'message': 'Cannot inspect feature/statistic namespaces'}
        violations.extend(statistics_violations)

        # 检查常见的非法key访问模式
        # 需要检查的非法keys：queue_length, delay, num_waiting_time（应该是waiting_time）等
        illegal_key_patterns = [
            'queue_length',
            'delay',
            'queue',
            'num_waiting_time',  # 应该是 waiting_time
            'vehicle_delay',
            'queue_len',
        ]
        
        # 检查代码中是否出现这些非法的key访问
        import re
        for illegal_key in illegal_key_patterns:
            # 匹配 .get('illegal_key') 或 .get('illegal_key', 0.0) 或 ["illegal_key"] 或 ['illegal_key']
            patterns = [
                rf"\.get\(['\"]({illegal_key})['\"]\s*(?:,\s*[^)]+)?\)",
                rf"\[['\"]{illegal_key}['\"]",
                rf"['\"]({illegal_key})['\"]\s*:",
            ]
            for pattern in patterns:
                if re.search(pattern, function_code, re.IGNORECASE):
                    violations.append(
                        f"Function tries to access illegal feature key '{illegal_key}' which doesn't exist in actual data. "
                        f"Actual available keys: {self.actual_feature_keys}"
                    )
                    break
        
        # 检查代码中引用的feature keys是否在actual_feature_keys中
        # 提取 .get('key') 或 .get('key', default) 或 fbd['key'] 的key
        key_pattern = r"\.get\(['\"]([^'\"]+)['\"]\s*(?:,\s*[^)]+)?\)|\[['\"](\w+)['\"]\]"
        matches = re.findall(key_pattern, function_code)
        
        # matches is a list of tuples like [('pressure', ''), ('', 'num_vehicle')]
        used_keys = set()
        # Keys used in feature_stats (normalization dict), not state/features_by_direction
        allowed_non_feature_keys = {
            'current_phase', 'controlled_directions', 'features_by_direction', 'state',
            'min', 'max', 'mean', 'std',  # feature_stats entry keys
            # EcoLight-native aligned binary state.
            'next_phase', 'binary_features', 'green_vehicle', 'red_vehicle',
            'exact_binary_features',
        }
        for match in matches:
            key = match[0] if match[0] else match[1]
            if key and key not in allowed_non_feature_keys:
                used_keys.add(key)
        
        # 检查used_keys是否都在actual_feature_keys中（跳过已在 illegal 中报告的 key）
        illegal_set = set(illegal_key_patterns)
        for used_key in used_keys:
            if used_key in illegal_set:
                continue  # Already reported above
            if used_key not in self.actual_feature_keys or used_key.endswith(('_raw', '_norm')):
                violations.append(
                    f"Function tries to access feature key '{used_key}' which doesn't exist in actual data. "
                    f"Actual available keys: {self.actual_feature_keys}"
                )
        
        # Deduplicate by message (illegal and used_keys can produce similar messages for same key)
        seen = set()
        violations = [v for v in violations if v not in seen and not seen.add(v)]
        
        valid = len(violations) == 0
        message = 'Feature key constraints satisfied' if valid else f'{len(violations)} violations detected'
        
        return {
            'valid': valid,
            'violations': violations,
            'message': message,
            'used_keys': list(used_keys),
            'actual_keys': self.actual_feature_keys
        }

    def check_feature_stats_key_constraints(self, function_code: str) -> Dict[str, Any]:
        """Check source statistics aliases and nested fields, including aliases."""
        try:
            tree = ast.parse(function_code)
        except SyntaxError as exc:
            return {'valid': False, 'violations': [f'Invalid Python syntax: {exc}'],
                    'message': 'Cannot inspect feature_stats keys'}
        used = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == 'feature_stats' and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                used.add(node.slice.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == 'feature_stats'
                  and node.func.attr == 'get' and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                used.add(node.args[0].value)
            elif isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
                right = node.comparators[0]
                if (isinstance(node.ops[0], (ast.In, ast.NotIn))
                        and isinstance(node.left, ast.Constant)
                        and isinstance(node.left.value, str)
                        and isinstance(right, ast.Name) and right.id == 'feature_stats'):
                    used.add(node.left.value)
        available = set(self.feature_stats)
        missing = sorted(used - available)
        violations = [
            f"feature_stats key '{key}' is absent in source data; available keys: {sorted(available)}. "
            "Use an available statistics key or compute directly from state['features_by_direction']."
            for key in missing
        ]
        from .statistics_namespace import mask_statistic_keys
        _, nested_violations, _ = mask_statistic_keys(function_code, self.feature_stats)
        violations = sorted(set(violations + nested_violations))
        return {'valid': not violations, 'violations': violations,
                'message': 'Feature statistics keys valid' if not violations else 'Unknown feature statistics key',
                'used_keys': sorted(used), 'available_keys': sorted(available)}
    
    # 以下方法已不再使用（只保留系统测试）
    # 原来的 validate() 和 check_parameter_constraints() 可删除
    
    def validate_with_tests(
        self,
        utility_function: Any,
        test_parameters: Dict[str, float],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        使用系统测试进行表达式验证（使用固定测试参数）
        
        Args:
            utility_function: 编译好的效用函数对象
            test_parameters: 测试用参数（固定为1.0）
            verbose: 是否打印详细信息
        
        Returns:
            测试验证结果字典
        """
        try:
            # run_all_tests() 返回包含 test_results 的字典
            results = self.test_validator.run_all_tests(
                utility_function=utility_function,
                parameters=test_parameters,
                verbose=verbose
            )
            
            return results  # 直接返回，包含 valid, test_results, total_violations, message
            
        except Exception as e:
            return {
                'valid': False,
                'test_results': {},
                'total_violations': 4,  # 4个测试
                'message': f'Test execution failed: {str(e)}'
            }

    def _generate_test_parameters(
        self,
        utility_function: Any,
        n_samples: int = 3
    ) -> List[Dict[str, float]]:
        """
        生成多组正数参数用于结构验证（不依赖优化结果）
        包括极端情况：很小、很大的参数值
        """
        parameter_bounds = utility_function.get_parameter_bounds()
        if not parameter_bounds:
            return [{}]

        test_params_list = []
        
        # 1) 全1参数（会裁剪到边界）
        ones_params: Dict[str, float] = {}
        for name, (lower, upper) in parameter_bounds.items():
            val = 1.0
            if lower is not None and val < lower:
                val = lower
            if upper is not None and val > upper:
                val = upper
            ones_params[name] = float(val)
        test_params_list.append(ones_params)

        # 2) 中点参数
        mid_params: Dict[str, float] = {}
        for name, (lower, upper) in parameter_bounds.items():
            if lower is None or upper is None:
                mid_params[name] = 1.0
            else:
                mid_params[name] = float((lower + upper) / 2.0)
        test_params_list.append(mid_params)

        # 3) 极端参数：很小的值（接近0的非零值）
        # 这很重要，因为有些表达式（如 sqrt(x) * a）在 a 很小时可能产生数值问题
        tiny_params: Dict[str, float] = {}
        for name, (lower, upper) in parameter_bounds.items():
            if lower is None or upper is None:
                tiny_params[name] = 0.01
            else:
                # 使用参数范围的 1% 作为"很小"的值
                tiny_params[name] = float(max(0.01, (lower + upper) / 200.0))
        test_params_list.append(tiny_params)

        # 4) 极端参数：很大的值
        # 这也很重要，因为大参数可能导致溢出或其他数值问题
        large_params: Dict[str, float] = {}
        for name, (lower, upper) in parameter_bounds.items():
            if lower is None or upper is None:
                large_params[name] = 100.0
            else:
                # 使用参数范围的上界附近的大值
                large_params[name] = float(upper * 0.9) if upper else 100.0
        test_params_list.append(large_params)

        # 5) 随机正数参数（可重复）
        rng = np.random.default_rng(42)
        for _ in range(max(0, n_samples - 4)):
            rand_params: Dict[str, float] = {}
            for name, (lower, upper) in parameter_bounds.items():
                if lower is None or upper is None:
                    rand_params[name] = 1.0
                else:
                    rand_params[name] = float(rng.uniform(lower, upper))
            test_params_list.append(rand_params)

        return test_params_list
    
    def _validate_on_real_states(
        self,
        utility_function: Any,
        max_samples: int = 25,
        verbose: bool = False,
        include_boundary: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Run compute() on real teacher state samples + boundary/extreme synthetic states.
        For FunctionUtility only. Returns (valid, violations).
        """
        if self.action_mode == "binary_keep_switch":
            return self._validate_binary_on_real_states(
                utility_function,
                max_samples=max_samples,
                include_boundary=include_boundary,
            )

        violations: List[str] = []
        if include_boundary and self.data_loader and self.feature_stats:
            bdry_ok, bdry_v = self._check_boundary_extreme_function_utility(
                utility_function, max_samples=12, verbose=verbose
            )
            if not bdry_ok:
                violations.extend(bdry_v)
        if violations:
            return False, violations
        samples_tested = 0
        for inter_id in sorted(self.data_loader.intersection_ids)[:3]:  # First 3 intersections
            action_data = self.data_loader.get_action_consistency_data(
                inter_id, max_timesteps=max(50, max_samples // 3)
            )
            for item in action_data[:max(10, max_samples // 3)]:
                state_dict = item[0]
                for phase_id, state in state_dict.items():
                    try:
                        q = utility_function.compute(
                            state,
                            phase_id=phase_id,
                            feature_stats=self.feature_stats,
                            raise_on_error=True,  # Propagate errors so validation fails (no masking)
                        )
                        if not np.isfinite(float(q)):
                            violations.append(
                                f"Phase {phase_id} returned non-finite Q={q} on real state"
                            )
                        samples_tested += 1
                        if samples_tested >= max_samples:
                            break
                    except Exception as e:
                        violations.append(
                            f"Exception on real state (phase={phase_id}): {type(e).__name__}: {e}"
                        )
                if samples_tested >= max_samples:
                    break
            if samples_tested >= max_samples:
                break
        if samples_tested == 0 and not violations:
            violations.append("Runtime validation not evaluated: no source phase states")
        return len(violations) == 0, violations

    def _validate_binary_on_real_states(
        self,
        utility_function: Any,
        max_samples: int = 25,
        include_boundary: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Validate the aligned EcoLight binary head on exact rollout states."""
        violations: List[str] = []
        if include_boundary:
            boundary_ok, boundary_violations = (
                self._check_binary_boundary_extreme(
                    utility_function,
                    max_samples=12,
                )
            )
            if not boundary_ok:
                violations.extend(boundary_violations)
        if violations:
            return False, violations

        tested = 0
        for inter_id in sorted(self.data_loader.intersection_ids)[:3]:
            action_data = self.data_loader.get_action_consistency_data(
                inter_id,
                max_timesteps=max(50, max_samples // 3),
            )
            for item in action_data:
                state_dict = item[0]
                binary_state, _, _ = utility_function._native_binary_context(
                    state_dict
                )
                if not binary_state.get("exact_binary_features", False):
                    violations.append(
                        "EcoLight binary state is missing exact green/red "
                        "vehicle counts"
                    )
                    return False, violations
                try:
                    _, _, keep_q, switch_q = (
                        utility_function.compute_binary_action_values(
                            state_dict,
                            feature_stats=self.feature_stats,
                            raise_on_error=True,
                        )
                    )
                    if not (
                        np.isfinite(float(keep_q))
                        and np.isfinite(float(switch_q))
                    ):
                        violations.append(
                            "Binary action values are non-finite: "
                            f"Q_keep={keep_q}, Q_switch={switch_q}"
                        )
                except Exception as exc:
                    violations.append(
                        "Exception on EcoLight binary state: "
                        f"{type(exc).__name__}: {exc}"
                    )
                tested += 1
                if tested >= max_samples:
                    break
            if tested >= max_samples:
                break
        if tested == 0 and not violations:
            violations.append("Binary runtime validation not evaluated: no source states")
        return len(violations) == 0, violations

    def _check_monotonicity_function_utility(
        self, utility_function: Any, max_samples: int = 100,
        max_feature_checks: int = 256, verbose: bool = False,
    ) -> Tuple[bool, List[str]]:
        return phase_monotonicity(self, utility_function, max_samples, max_feature_checks, verbose)

    @staticmethod
    def _set_binary_feature(
        state_dict: Dict[int, Dict[str, Any]],
        key: str,
        value: float,
    ) -> None:
        for state in state_dict.values():
            binary = state.setdefault("binary_features", {})
            binary[key] = float(value)

    def _check_binary_monotonicity(
        self, utility_function: Any, max_feature_checks: int = 256,
    ) -> Tuple[bool, List[str]]:
        return binary_monotonicity(self, utility_function, max_feature_checks)

    def _check_boundary_extreme_function_utility(
        self,
        utility_function: Any,
        max_samples: int = 15,
        verbose: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        对 FunctionUtility 做边界/极值测试：构造 all_zero、all_max、mixed 等合成 state，
        确保 compute() 不崩溃、返回有限值。用于发现 div-by-zero、NaN 等。
        """
        if self.action_mode == "binary_keep_switch":
            return self._check_binary_boundary_extreme(
                utility_function,
                max_samples=max_samples,
            )

        violations: List[str] = []
        if not (self.data_loader and self.feature_stats):
            return True, []

        keys = list(self.actual_feature_keys or []) or list(self.feature_stats.keys())
        if not keys:
            return True, []

        # 从真实数据取一个模板 state 以获取结构
        template_state = None
        template_phase_id = 0
        for inter_id in sorted(self.data_loader.intersection_ids)[:2]:
            action_data = self.data_loader.get_action_consistency_data(inter_id, max_timesteps=30)
            for item in action_data[:3]:
                state_dict = item[0]
                for pid, st in state_dict.items():
                    if st.get("features_by_direction") and st.get("controlled_directions"):
                        template_state = copy.deepcopy(st)
                        template_phase_id = pid
                        break
                if template_state:
                    break
            if template_state:
                break

        if not template_state:
            return True, []

        fbd = template_state.get("features_by_direction", {})
        controlled = template_state.get("controlled_directions", []) or list(fbd.keys())
        # 为每个 key 取合理范围
        max_vals = {}
        for k in keys:
            stat = self.feature_stats.get(k, {})
            max_vals[k] = float(stat.get("max", 500.0))
            if max_vals[k] < 1e-6:
                max_vals[k] = 1.0

        def _make_fbd(val_by_key: Dict[str, float]) -> Dict:
            out = {}
            for d in controlled:
                out[d] = {k: val_by_key.get(k, 0.0) for k in keys}
                if 'waiting_time' in out[d] and 'sum_waiting_time' in out[d]:
                    value = max(out[d]['waiting_time'], out[d]['sum_waiting_time'])
                    out[d]['waiting_time'] = out[d]['sum_waiting_time'] = value
            return out

        # 边界/极值用例
        cases = [
            ("all_zero", {k: 0.0 for k in keys}),
            ("all_max", {k: max_vals.get(k, 1.0) for k in keys}),
            ("mixed_half", {k: max_vals.get(k, 1.0) * 0.5 for k in keys}),
        ]
        for i, k in enumerate(keys[:5]):  # 单特征极值
            c = {x: 0.0 for x in keys}
            c[k] = max_vals.get(k, 1.0)
            cases.append((f"only_{k}_max", c))

        tested = 0
        for case_name, val_by_key in cases:
            if tested >= max_samples:
                break
            st = copy.deepcopy(template_state)
            st["features_by_direction"] = _make_fbd(val_by_key)
            try:
                q = utility_function.compute(
                    st, phase_id=template_phase_id, feature_stats=self.feature_stats, raise_on_error=True
                )
                if not np.isfinite(float(q)):
                    violations.append(f"Boundary '{case_name}': Q={q} (non-finite)")
            except Exception as e:
                violations.append(f"Boundary '{case_name}': {type(e).__name__}: {e}")
            tested += 1

        # Pressure is signed even if the source trajectory happened to contain
        # only nonnegative values. Do not truncate these mandatory checks by
        # max_samples or assume that current_phase equals the evaluated phase.
        for case_name, st in signed_boundary_states(template_state, template_phase_id, keys):
            try:
                q = utility_function.compute(
                    st, phase_id=template_phase_id, feature_stats=self.feature_stats, raise_on_error=True
                )
                if not np.isfinite(float(q)):
                    violations.append(f"Boundary '{case_name}': Q={q} (non-finite)")
            except Exception as e:
                violations.append(f"Boundary '{case_name}': {type(e).__name__}: {e}")
            if len(violations) >= 8:
                break

        return len(violations) == 0, violations

    def _check_binary_boundary_extreme(
        self,
        utility_function: Any,
        max_samples: int = 15,
    ) -> Tuple[bool, List[str]]:
        if not (self.data_loader and self.feature_stats):
            return True, []

        template = None
        for inter_id in sorted(self.data_loader.intersection_ids)[:2]:
            action_data = self.data_loader.get_action_consistency_data(
                inter_id,
                max_timesteps=10,
            )
            if action_data:
                template = action_data[0][0]
                break
        if template is None:
            return True, []

        green_max = max(
            1.0,
            float(
                self.feature_stats.get("green_vehicle", {}).get(
                    "max", 1.0
                )
            ),
        )
        red_max = max(
            1.0,
            float(
                self.feature_stats.get("red_vehicle", {}).get(
                    "max", 1.0
                )
            ),
        )
        cases = [
            ("all_zero", 0.0, 0.0),
            ("green_max", green_max, 0.0),
            ("red_max", 0.0, red_max),
            ("both_max", green_max, red_max),
            ("mixed_half", green_max * 0.5, red_max * 0.5),
        ][:max_samples]
        violations: List[str] = []
        action_margins: List[float] = []
        for name, green, red in cases:
            state_dict = copy.deepcopy(template)
            self._set_binary_feature(
                state_dict, "green_vehicle", green
            )
            self._set_binary_feature(state_dict, "red_vehicle", red)
            try:
                _, _, keep_q, switch_q = (
                    utility_function.compute_binary_action_values(
                        state_dict,
                        feature_stats=self.feature_stats,
                        raise_on_error=True,
                    )
                )
                if not (
                    np.isfinite(float(keep_q))
                    and np.isfinite(float(switch_q))
                ):
                    violations.append(
                        f"Boundary '{name}' produced non-finite binary Q"
                    )
                else:
                    action_margins.append(float(switch_q) - float(keep_q))
            except Exception as exc:
                violations.append(
                    f"Boundary '{name}': {type(exc).__name__}: {exc}"
                )
        if (
            self.binary_min_margin_span > 0.0
            and len(action_margins) == len(cases)
            and len(action_margins) >= 2
        ):
            span = max(action_margins) - min(action_margins)
            if span < self.binary_min_margin_span:
                violations.append(
                    "Binary action margin has insufficient response across "
                    f"source feature extremes: span={span:.6g}, required="
                    f"{self.binary_min_margin_span:.6g}. A later phase bias "
                    "can dominate KEEP/SWITCH. Preserve traffic sensitivity "
                    "without changing the action interface."
                )
        return len(violations) == 0, violations

    def _check_zero_demand_minimum_function_utility(
        self,
        utility_function: Any,
        max_samples: int = 25,
        verbose: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Compare zeroed and observed demand on the same phase-level input.

        Appendix A.3 says to zero all demand features for one candidate while
        keeping non-demand features fixed. In particular, phase_id and
        current_phase must be identical in both evaluations. Comparing the
        zeroed phase against a different phase would confound demand with
        phase-specific utility offsets.
        """
        if self.action_mode == "binary_keep_switch" or not (self.data_loader and self.feature_stats):
            return True, []

        keys = [
            key for key in (
                "pressure", "num_vehicle", "num_waiting_vehicle",
                "waiting_time", "sum_waiting_time",
            )
            if not self.actual_feature_keys or key in self.actual_feature_keys
        ]
        positive_keys = {
            "num_vehicle", "num_waiting_vehicle", "waiting_time", "sum_waiting_time"
        } & set(keys)
        if not positive_keys and "pressure" not in keys:
            return True, []

        def has_positive_demand(state: Dict[str, Any]) -> bool:
            fbd = state.get("features_by_direction") or {}
            controlled = state.get("controlled_directions") or list(fbd.keys())
            for direction in controlled:
                feats = fbd.get(direction) or {}
                for key in positive_keys:
                    try:
                        if float(feats.get(key, 0.0)) > 1e-6:
                            return True
                    except (TypeError, ValueError):
                        continue
                if "pressure" in keys:
                    try:
                        if float(feats.get("pressure", 0.0)) > 1e-6:
                            return True
                    except (TypeError, ValueError):
                        pass
            return False

        violations: List[str] = []
        checks = 0
        tol = 1e-6
        for inter_id in sorted(self.data_loader.intersection_ids)[:3]:
            action_data = self.data_loader.get_action_consistency_data(
                inter_id, max_timesteps=max(50, max_samples)
            )
            for item in action_data:
                state_dict = item[0]
                for phase_id, source_state in state_dict.items():
                    if not has_positive_demand(source_state):
                        continue
                    observed_state = copy.deepcopy(source_state)
                    zero_state = copy.deepcopy(source_state)
                    controlled = zero_state.get("controlled_directions") or list(
                        (zero_state.get("features_by_direction") or {}).keys()
                    )
                    zero_fbd = zero_state.setdefault("features_by_direction", {})
                    for direction in controlled:
                        feats = zero_fbd.setdefault(direction, {})
                        for key in keys:
                            if key != "pressure" or float(feats.get(key, 0.0)) > 0.0:
                                feats[key] = 0.0
                    try:
                        q_zero = float(utility_function.compute(
                            zero_state,
                            phase_id=phase_id,
                            feature_stats=self.feature_stats,
                            raise_on_error=True,
                        ))
                        q_observed = float(utility_function.compute(
                            observed_state,
                            phase_id=phase_id,
                            feature_stats=self.feature_stats,
                            raise_on_error=True,
                        ))
                    except Exception as exc:
                        violations.append(
                            f"Zero-demand comparison for phase {phase_id} raised "
                            f"{type(exc).__name__}: {exc}"
                        )
                        checks += 1
                        if checks >= max_samples:
                            return False, violations
                        continue
                    if not (np.isfinite(q_zero) and np.isfinite(q_observed)):
                        violations.append(
                            f"Zero-demand comparison for phase {phase_id} produced non-finite Q "
                            f"(zero={q_zero}, observed={q_observed})"
                        )
                    elif q_zero > q_observed + tol:
                        violations.append(
                            f"Zero-demand phase {phase_id} scored above its observed-demand "
                            f"counterfactual (Q_zero={q_zero:.6g}, Q_observed={q_observed:.6g})"
                        )
                    checks += 1
                    if checks >= max_samples:
                        return len(violations) == 0, violations
        return len(violations) == 0, violations

    def prepare_source_checks(self) -> None:
        """Prepare paired source probes before the first candidate/API request."""
        from LLM_SR_DISTILLATION.active_phase_protection import plan, _binding
        from LLM_SR_DISTILLATION.source_preparation import SourcePreparationError
        prepared = getattr(self, '_active_phase_prepared', None)
        if prepared is None or prepared.get('binding') != _binding(self):
            try:
                self._active_phase_prepared = plan(self)
            except (ValueError, TypeError, KeyError) as exc:
                raise SourcePreparationError(f'Cannot prepare active-phase tests: {exc}') from exc

    def _check_active_phase_protection_function_utility(
        self, utility_function: Any, max_samples: int = 25, verbose: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Check every source node/phase; max_samples is a legacy argument."""
        import json
        from LLM_SR_DISTILLATION.active_phase_protection import check
        from LLM_SR_DISTILLATION.source_preparation import SourcePreparationError
        self.prepare_source_checks()
        result = check(self, utility_function, prepared=self._active_phase_prepared)
        if any(v['kind'] == 'coverage_error' for v in result['violations']):
            raise SourcePreparationError(str(result['violations']))
        result['legacy_sample_budget'] = max_samples
        result['budget_policy'] = 'four_contexts_per_source_node_and_physical_phase'
        self._last_active_phase_protection = result
        if result['status'] == 'not_applicable':
            # The Boolean is gate eligibility, not evidence of a passed test.
            # validate_hybrid records status/applicable explicitly below.
            return True, ['NOT_APPLICABLE: ' + result['reason']]
        return result['status'] == 'passed', [json.dumps(v, sort_keys=True) for v in result['violations']]

    def validate_hybrid(
        self,
        expression_template: str,
        utility_function: Any,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        验证表达式结构合理性（不依赖具体参数值）
        
        Args:
            expression_template: 效用函数表达式模板
            utility_function: 编译好的效用函数对象
            verbose: 是否打印详细信息
        
        Returns:
            验证结果 - 4项约束条件的检查结果
        """
        from .policy_identity import check_process_identity
        identity = check_process_identity(expression_template)
        if not identity['valid']:
            return dict(valid=False, score=0.0, violations=identity['violations'],
                        message=identity['message'], constraints={'process_identity': {
                            'name': 'Process-independent utility primitives', 'satisfied': False,
                            'details': identity}})
        stats_keys = self.check_feature_stats_key_constraints(expression_template)
        if not stats_keys['valid']:
            return {'valid': False, 'score': 0.0, 'violations': stats_keys['violations'],
                    'message': stats_keys['message'],
                    'constraints': {'feature_stats_keys': {
                        'name': 'Feature statistics key usage', 'satisfied': False,
                        'details': stats_keys}}}
        numeric = self.check_numerical_safety(expression_template)
        if not numeric['valid']:
            return {'valid': False, 'score': 0.0, 'violations': numeric['violations'],
                    'message': numeric['message'],
                    'constraints': {'numerical_safety': {'name': 'Numerical safety',
                        'satisfied': False, 'details': numeric}}}
        # ============ Semantic Constraint: Feature Key Usage ============
        # Check that the generated function only uses feature keys that actually exist in data
        feature_key_check = self.check_feature_key_constraints(expression_template)
        if not feature_key_check['valid']:
            # Early exit if feature key constraint is violated
            violations = feature_key_check.get('violations', [])
            if verbose:
                for violation in violations:
                    print(f"  [FAIL] Feature Key Constraint Violated: {violation}")
            return {
                'valid': False,
                'score': 0.0,
                'constraints': {
                    'feature_keys': {
                        'name': 'Feature Key Usage',
                        'description': 'Function can only use feature keys that exist in actual data',
                        'satisfied': False,
                        'details': feature_key_check
                    }
                },
                'violations': violations,
                'message': f'Invalid - Feature key constraint violated: {violations[0] if violations else "Unknown"}'
            }
        
        if verbose and self.actual_feature_keys:
            print(f"  [OK] Feature key constraint satisfied (uses keys from: {feature_key_check.get('used_keys', [])})")

        constraints = {'numerical_safety': {'name': 'Numerical safety',
                       'description': 'Explicit nonzero denominator guards',
                       'satisfied': True, 'details': numeric}}
        # 凡实现了 compute(state, phase_id, feature_stats) 且具备 data_loader+feature_stats 的控制器，
        # 均走与 LLM（FunctionUtility）相同的真实 state 单调性检查，保证对比方法与 LLM 同一套物理语义约束
        has_compute = callable(getattr(utility_function, "compute", None))
        has_data = bool(self.data_loader and self.feature_stats)
        full_source=None
        if os.environ.get("PPL_METHOD_PROFILE")=="full_100_256_v1":
            from ppl_tsc_current.source_execution import validate_all_source
            full_source=validate_all_source(self,utility_function)
            if not full_source['satisfied']:
                return dict(valid=False,score=0.,message='Complete source execution failed',violations=full_source['details'],constraints={**constraints,'complete_source_execution':full_source})
            constraints['complete_source_execution']=full_source

        use_real_state_monotonicity = has_compute and has_data

        if use_real_state_monotonicity:
            runtime_valid, runtime_violations = self._validate_on_real_states(
                utility_function, max_samples=25, include_boundary=True, verbose=verbose)
            if not runtime_valid:
                return {'valid': False, 'score': 0.0, 'violations': runtime_violations,
                        'message': 'Real-state or boundary execution failed',
                        'constraints': {**constraints, 'real_state': {'name': 'Real-state and boundary execution',
                            'satisfied': False, 'violations': runtime_violations,
                            'details': {'violations': runtime_violations}}}}
            # ==== 与 LLM 相同的单调性检查：在真实 teacher state 上做弱单调性 ====
            mono_valid, mono_violations = self._check_monotonicity_function_utility(
                utility_function, max_samples=100, max_feature_checks=256, verbose=verbose
            )
            if verbose:
                if mono_valid:
                    print("  [OK] Monotonicity check on real states passed")
                else:
                    for v in mono_violations:
                        print(f"  [FAIL] Monotonicity (real states): {v}")

            constraints['monotonicity'] = {
                'name': 'Monotonicity',
                'description': 'Weak monotonicity on real teacher states: increasing traffic feature should not significantly decrease Q',
                'satisfied': mono_valid,
                'details': mono_violations
            }
            if os.environ.get("PPL_METHOD_PROFILE")=="full_100_256_v1":
                zero_valid,zero_details=self._check_zero_demand_minimum_function_utility(utility_function,max_samples=25,verbose=verbose)
                active_valid,active_details=self._check_active_phase_protection_function_utility(utility_function,max_samples=25,verbose=verbose)
                constraints['zero_demand_minimum']={'name':'Zero-positive-demand minimum','satisfied':zero_valid,'details':zero_details}
                active_report = self._last_active_phase_protection
                constraints['active_phase_protection'] = {
                    'name': 'Active-phase protection', 'satisfied': active_valid,
                    'status': active_report['status'], 'applicable': active_report['applicable'],
                    'comparison_satisfied': active_report['satisfied'],
                    'planned_pairs': active_report['planned_pairs'],
                    'executed_pairs': active_report['executed_pairs'],
                    'node_coverage': active_report['node_coverage'],
                    'phase_coverage': active_report['phase_coverage'],
                    'inactive_coverage': active_report['inactive_coverage'],
                    'complete': active_report['complete'], 'details': active_details,
                }
            constraints['demand_discrimination'] = demand_discrimination(self, utility_function)
            constraints['phase_label_consistency'] = phase_label_consistency(
                self, utility_function, max_samples=25, max_checks=40)
            constraints['direction_order_consistency'] = direction_order_consistency(
                self, utility_function, max_samples=25, max_checks=40)
            constraints['boundary'] = {
                'name': 'Boundary conditions',
                'description': 'Executed source-state and synthetic boundary checks, including signed pressure and phase activity',
                'satisfied': True,
                'details': []
            }
            constraints['extreme'] = {
                'name': 'Extreme values',
                'description': 'Executed synthetic extremes in runtime validation',
                'satisfied': True,
                'details': []
            }
        else:
            # UtilityFunction (expression-based): use flat-feature system tests
            test_parameter_sets = self._generate_test_parameters(
                utility_function=utility_function,
                n_samples=3
            )

            test_validations: List[Dict[str, Any]] = []
            for params in test_parameter_sets:
                test_validations.append(
                    self.validate_with_tests(
                        utility_function=utility_function,
                        test_parameters=params,
                        verbose=verbose
                    )
                )

            def _passed_for(test_key: str) -> bool:
                for validation in test_validations:
                    test_results = validation.get('test_results', {})
                    if not test_results.get(test_key, {}).get('passed', False):
                        return False
                return True

            def _details_for(test_key: str) -> List[Dict[str, Any]]:
                return [
                    validation.get('test_results', {}).get(test_key, {})
                    for validation in test_validations
                ]
            
            constraints['monotonicity'] = {
                'name': 'Monotonicity',
                'description': 'Each feature increase should not decrease Q',
                'satisfied': _passed_for('monotonicity'),
                'details': _details_for('monotonicity')
            }
            
            constraints['boundary'] = {
                'name': 'Boundary conditions',
                'description': 'Q must be finite for all_zero, all_min, all_max, mixed_extreme',
                'satisfied': _passed_for('boundary'),
                'details': _details_for('boundary')
            }
            
            constraints['extreme'] = {
                'name': 'Extreme values',
                'description': 'Q must be finite when a single feature is at its extreme',
                'satisfied': _passed_for('extreme'),
                'details': _details_for('extreme')
            }
        
        # 综合判定：3项都通过才有效
        valid = all(c['satisfied'] for c in constraints.values())
        
        # 收集所有未满足的约束
        violations = []
        for constraint in constraints.values():
            if constraint['satisfied']:
                continue
            violations.append(f"{constraint['name']}: {constraint.get('description', 'See concrete source-state counterexamples below')}")
            details = constraint.get('details', [])
            if isinstance(details, list):
                violations.extend(detail for detail in details if isinstance(detail, str))
        
        return {
            'valid': valid,
            'score': 1.0 if valid else 0.0,
            'constraints': constraints,  # 4项约束的详细信息
            'violations': violations,  # 违反的约束列表
            'message': f'Valid - all constraints satisfied' if valid else f'Invalid - violations: {", ".join([c["name"] for c in constraints.values() if not c["satisfied"]])}'
        }
    # _compute_validation_score 已删除（改为纯二元判定）
