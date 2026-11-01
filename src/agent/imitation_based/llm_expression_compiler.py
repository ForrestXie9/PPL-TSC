"""
LLM生成的符号表达式编译和系数拟合模块
将字符串表达式转换为可执行的Python函数，并拟合系数
"""
import re
import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class ExpressionCompiler:
    """表达式编译器，将符号表达式转换为可执行函数"""
    
    def __init__(self, feature_names: List[str]):
        """
        Args:
            feature_names: 特征名列表
        """
        self.feature_names = feature_names
        self.num_features = len(feature_names)
    
    def _normalize_expression(self, expr: str) -> str:
        """
        规范化表达式，将非Python语法转换为Python语法
        
        Args:
            expr: 原始表达式字符串
        
        Returns:
            规范化后的表达式字符串
        """
        normalized = expr
        
        # 1. 转换三元运算符: (condition ? value_if_true : value_if_false) 
        #    -> (value_if_true if condition else value_if_false)
        
        def replace_ternary_operator(text):
            """替换文本中的三元运算符"""
            result = text
            max_iterations = 20  # 防止无限循环
            iteration = 0
            
            while '?' in result and ':' in result and iteration < max_iterations:
                iteration += 1
                # 找到所有 ? 的位置
                q_positions = [i for i, char in enumerate(result) if char == '?']
                
                if not q_positions:
                    break
                
                replaced = False
                # 从右到左处理，避免位置偏移
                for q_pos in reversed(q_positions):
                    # 找到对应的 : 位置（在 ? 之后）
                    colon_pos = result.find(':', q_pos)
                    if colon_pos == -1:
                        continue
                    
                    # 找到包含这个三元运算符的括号对
                    # 从 ? 向前找最近的 (
                    paren_start = result.rfind('(', 0, q_pos)
                    if paren_start == -1:
                        continue
                    
                    # 从 : 向后找对应的 )
                    paren_end = result.find(')', colon_pos)
                    if paren_end == -1:
                        continue
                    
                    # 验证括号匹配：检查从paren_start到paren_end之间是否有匹配的括号对
                    # 简单检查：确保paren_start < q_pos < colon_pos < paren_end
                    if not (paren_start < q_pos < colon_pos < paren_end):
                        continue
                    
                    # 提取各部分
                    condition = result[paren_start + 1:q_pos].strip()
                    true_val = result[q_pos + 1:colon_pos].strip()
                    false_val = result[colon_pos + 1:paren_end].strip()
                    
                    # 验证提取的内容是否合理（非空且不包含 ? 和 :）
                    if (condition and true_val and false_val and 
                        '?' not in condition and '?' not in true_val and '?' not in false_val and
                        ':' not in condition and ':' not in true_val):
                        # 替换为Python语法
                        python_expr = f'({true_val} if {condition} else {false_val})'
                        result = result[:paren_start] + python_expr + result[paren_end + 1:]
                        replaced = True
                        break  # 替换一个后重新开始查找
                
                if not replaced:
                    # 如果没有找到可替换的，退出
                    break
            
            return result
        
        # 应用三元运算符替换
        normalized = replace_ternary_operator(normalized)
        
        # 如果还有残留的三元运算符，使用正则表达式尝试处理
        # 匹配模式: (condition ? true_val : false_val) 或 condition ? true_val : false_val
        if '?' in normalized:
            # 使用更宽松的正则表达式尝试匹配
            # 模式: 任意内容 ? 数字或简单表达式 : 数字或简单表达式
            ternary_pattern = r'\(([^?]+)\s*\?\s*([^:]+)\s*:\s*([^)]+)\)'
            
            def regex_replace_ternary(match):
                condition = match.group(1).strip()
                true_val = match.group(2).strip()
                false_val = match.group(3).strip()
                return f'({true_val} if {condition} else {false_val})'
            
            normalized = re.sub(ternary_pattern, regex_replace_ternary, normalized)
        
        # 最后的回退：如果表达式中还有 ? 和 :，将整个有问题的部分替换为安全表达式
        # 这种情况通常是条件表达式如 (x0 == 5 ? 1 : 0)
        if '?' in normalized:
            # 尝试匹配 (x0 == N ? 1 : 0) 或类似模式，替换为 exp(-abs(x0 - N))
            eq_ternary = r'\(x(\d+)\s*==\s*(\d+)\s*\?\s*1\s*:\s*0\)'
            def eq_replace(match):
                var_idx = match.group(1)
                target_val = match.group(2)
                # 用高斯类似的平滑函数替代
                return f'np.exp(-np.abs(X[:, {var_idx}] - {target_val}))'
            normalized = re.sub(eq_ternary, eq_replace, normalized)
            
            # 如果还有残留的?，直接移除有问题的项（替换为0）
            if '?' in normalized:
                print(f"  Warning: Could not fully convert ternary operators in: {expr}")
                print(f"  Partially converted: {normalized}")
                # 移除包含?的整个项（简单处理：将 term * (xxx ? yyy : zzz) 替换为 0）
                normalized = re.sub(r'[+\-]\s*[^+\-]*\?[^+\-]*', '', normalized)
                # 如果整个表达式都被清空了，给一个默认值
                if not normalized.strip() or normalized.strip() in ['+', '-', '*', '/']:
                    normalized = 'a1 * X[:, 0]'
                    print(f"  Fallback to simple expression: {normalized}")
        
        # 2. 转换逻辑运算符: && -> and, || -> or, ! -> not
        normalized = re.sub(r'\b&&\b', 'and', normalized)
        normalized = re.sub(r'\b\|\|\b', 'or', normalized)
        normalized = re.sub(r'!([^=])', r'not \1', normalized)  # !x -> not x，但保留 !=
        
        # 3. 转换比较操作为数值（用于算术表达式）
        # (x0 == N) -> (1.0 * (X[:, 0] == N)) 使其可以参与数值计算
        # 但这可能会导致问题，因为Python中 True/False 可以直接参与算术运算
        
        # 4. 处理可能的其他语法问题
        # 例如：确保幂运算符是 ** 而不是 ^
        normalized = re.sub(r'\^', '**', normalized)
        
        # 5. 处理 (x0 > N) 这类比较表达式，转换为浮点数
        # 这些在NumPy中返回布尔数组，需要确保可以用于算术运算
        # Python会自动将True当作1，False当作0，所以通常不需要额外处理
        
        return normalized
    
    def compile_expression(self, expr: str) -> Tuple[Callable, List[str]]:
        """
        编译表达式为可执行函数
        
        Args:
            expr: 符号表达式字符串，例如 "a1 * x0 + a2 * sqrt(x1 + 1)"
        
        Returns:
            (compiled_func, coefficient_names) 元组
            compiled_func: 可执行函数 f(X, *coeffs) -> y
            coefficient_names: 系数名列表，如 ['a1', 'a2', ...]
        """
        # 首先规范化表达式（转换非Python语法）
        expr = self._normalize_expression(expr)
        
        # 提取所有系数名
        coeff_pattern = r'\ba(\d+)\b'
        coeff_matches = re.findall(coeff_pattern, expr)
        coeff_names = sorted(set([f'a{num}' for num in coeff_matches]), 
                           key=lambda x: int(x[1:]))
        
        if not coeff_names:
            raise ValueError(f"No coefficients found in expression: {expr}")
        
        # 替换变量名：x0, x1, ... -> X[:, 0], X[:, 1], ...
        # 替换系数名：a1, a2, ... -> coeffs[0], coeffs[1], ...
        compiled_expr = expr
        
        # 替换特征变量（需要小心处理，避免替换掉已替换的部分）
        # 从大到小替换，避免x10被替换成X[:, 1]0
        for i in range(self.num_features - 1, -1, -1):
            compiled_expr = re.sub(rf'\bx{i}\b', f'X[:, {i}]', compiled_expr)
        
        # 替换系数变量
        for idx, coeff_name in enumerate(coeff_names):
            compiled_expr = re.sub(rf'\b{coeff_name}\b', f'coeffs[{idx}]', compiled_expr)
        
        # 替换数学函数，确保使用numpy版本
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
        
        # 创建函数
        # 注意：在 f-string 中，需要使用双大括号 {{ 和 }} 来转义，或者使用字符串拼接
        func_code = f"""
def compiled_func(X, *coeffs):
    import numpy as np
    try:
        result = {compiled_expr}
        # 确保返回一维数组
        if isinstance(result, (int, float)):
            result = np.full(X.shape[0], result)
        # 确保是一维数组
        result = np.asarray(result).flatten()
        # 确保长度匹配
        if len(result) != X.shape[0]:
            if len(result.shape) > 1:
                result = result.flatten()
            if len(result) != X.shape[0]:
                # 如果长度不匹配，尝试广播或重复
                if len(result) == 1:
                    result = np.full(X.shape[0], result[0])
                else:
                    # 使用字符串拼接来避免 f-string 解析问题
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
        """
        拟合表达式中的系数
        
        Args:
            expr: 符号表达式字符串
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
            initial_coeffs: 初始系数值
        
        Returns:
            (fitted_coeffs, mse) 元组
        """
        # 编译表达式
        compiled_func, coeff_names = self.compile_expression(expr)
        num_coeffs = len(coeff_names)
        
        # 用于调试的变量
        last_error = None
        
        # 定义损失函数
        def loss(coeffs):
            nonlocal last_error
            try:
                y_pred = compiled_func(X, *coeffs)
                # 检查是否有无效值
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    last_error = "Expression produces NaN or Inf values"
                    return 1e10
                mse = mean_squared_error(y, y_pred)
                return mse
            except Exception as e:
                last_error = str(e)
                return 1e10
        
        # 初始化系数
        if initial_coeffs is None:
            initial_coeffs = np.random.randn(num_coeffs) * 0.1
        
        # 先测试一次表达式是否能正常工作
        test_mse = loss(initial_coeffs)
        if test_mse >= 1e10:
            print(f"  Warning: Expression evaluation failed with initial coeffs")
            if last_error:
                print(f"    Error: {last_error}")
            print(f"    Expression: {expr}")
        
        # 优化系数
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
                # 如果BFGS失败，尝试L-BFGS-B
                result = minimize(
                    loss,
                    initial_coeffs,
                    method='L-BFGS-B',
                    bounds=[(-10, 10)] * num_coeffs,
                    options={'maxiter': 1000}
                )
                fitted_coeffs = result.x
                final_mse = loss(fitted_coeffs)
            
            # 如果最终MSE仍然很高，输出警告
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
        """
        创建预测函数
        
        Args:
            expr: 符号表达式字符串
            coeffs: 拟合的系数值
        
        Returns:
            预测函数 f(X) -> y
        """
        compiled_func, _ = self.compile_expression(expr)
        
        def predictor(X):
            return compiled_func(X, *coeffs)
        
        return predictor


class LLMExpressionModel:
    """LLM生成的表达式模型（支持pickle序列化）"""
    
    def __init__(self, expr: str, coeffs: np.ndarray, compiler: ExpressionCompiler):
        """
        Args:
            expr: 符号表达式字符串
            coeffs: 拟合的系数
            compiler: 表达式编译器
        """
        self.expr = expr
        self.coeffs = coeffs
        self.feature_names = compiler.feature_names  # 保存特征名用于重建
        self._predictor = None  # 延迟初始化
    
    def _get_predictor(self):
        """获取或创建预测器（延迟初始化）"""
        if self._predictor is None:
            compiler = ExpressionCompiler(self.feature_names)
            self._predictor = compiler.create_predictor(self.expr, self.coeffs)
        return self._predictor
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测Q值"""
        return self._get_predictor()(X)
    
    def __getstate__(self):
        """序列化时排除不可pickle的对象"""
        state = self.__dict__.copy()
        # 排除本地函数predictor
        state['_predictor'] = None
        return state
    
    def __setstate__(self, state):
        """反序列化时恢复状态"""
        self.__dict__.update(state)
        # predictor会在首次调用predict时延迟重建
    
    def get_expression_with_coeffs(self) -> str:
        """获取带系数的表达式字符串"""
        expr = self.expr
        coeff_names = re.findall(r'\ba(\d+)\b', expr)
        coeff_names = sorted(set([f'a{num}' for num in coeff_names]), 
                           key=lambda x: int(x[1:]))
        
        for idx, coeff_name in enumerate(coeff_names):
            expr = re.sub(rf'\b{coeff_name}\b', f'{self.coeffs[idx]:.6f}', expr)
        
        return expr

